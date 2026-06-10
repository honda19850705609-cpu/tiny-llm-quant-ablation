"""
Train a fresh TinyLLM on a synthetic task (copy / reverse / sort / kv /
induction / multitask).

This is the Phase 1+2 trainer. Unlike scripts/train.py (which trains the
language model on TinyStories), this trains a small model FROM SCRATCH on a
symbolic task whose answers are checkable, so we can later ask "does the
*compressed* model still get the answer right?".

The checkpoint format matches scripts/train.py (so run_ablation-style loaders
work) and additionally stores the task spec under "task_spec", letting eval and
the task ablation rebuild the exact task.

Examples
--------
    # the long-range retrieval model (this is the one the KV ablation re-runs on)
    python -m scripts.train_task --task kv --n_pairs 32 --n_layer 4 --n_embd 256 \
        --block_size 128 --max_iters 8000 --ckpt_dir ckpt_kv

    # algorithmic warm-ups
    python -m scripts.train_task --task copy    --ckpt_dir ckpt_copy
    python -m scripts.train_task --task induction --seq_len 64 --ckpt_dir ckpt_ind

    # synthetic instruction following
    python -m scripts.train_task --task multitask --ckpt_dir ckpt_multi
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from tllm.model import TinyLLM, ModelConfig
from tllm import tasks as T
from tllm import eval as E


def build_task_from_args(args):
    if args.task in ("copy", "reverse", "sort"):
        cls = {"copy": T.CopyTask, "reverse": T.ReverseTask, "sort": T.SortTask}[args.task]
        return cls(seq_len=args.seq_len, n_symbols=args.n_symbols)
    if args.task == "kv":
        return T.KeyValueTask(n_pairs=args.n_pairs, n_keys=args.n_keys, n_vals=args.n_vals)
    if args.task == "induction":
        return T.InductionTask(seq_len=args.seq_len, n_symbols=args.n_symbols)
    if args.task == "multitask":
        return T.MultiTask(seq_len=args.seq_len, n_symbols=args.n_symbols)
    # --- Tier 2 ---
    if args.task == "addition":
        return T.AdditionTask(n_digits=args.n_digits)
    if args.task == "incontext":
        return T.InContextMappingTask(n_symbols=args.n_symbols, n_shots=args.n_shots,
                                      mode=args.mode)
    if args.task == "multihop":
        return T.MultiHopTask(n_pairs=args.n_pairs, hops=args.hops, n_symbols=args.n_symbols)
    if args.task == "statetrack":
        return T.StateTrackingTask(n_vars=args.n_vars, n_ops=args.n_ops, n_vals=args.n_vals)
    if args.task == "tooluse":
        return T.ToolUseTask(n_digits=args.n_digits)
    raise ValueError(args.task)


def is_tool_task(task):
    return getattr(task, "name", None) == "tooluse"


def eval_task(model, task, device, n_samples):
    """Accuracy for the task, using the agentic loop for tool tasks."""
    if is_tool_task(task):
        acc, called, n = E.tool_use_accuracy(model, task, device, n_samples=n_samples)
        return acc, n, f"acc {acc:.3f} | tool-called {called:.2f}"
    acc, n = E.task_accuracy(model, task, device, n_samples=n_samples)
    return acc, n, f"acc {acc:.3f}"


def lr_lambda(it, warmup, max_iters, min_ratio=0.1):
    if it < warmup:
        return it / max(1, warmup)
    if it > max_iters:
        return min_ratio
    decay = (it - warmup) / (max_iters - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * decay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["copy", "reverse", "sort", "kv", "induction", "multitask",
                             "addition", "incontext", "multihop", "statetrack", "tooluse"])
    ap.add_argument("--ckpt_dir", default="ckpt_task")
    ap.add_argument("--max_iters", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--ckpt_interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    # task knobs
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--n_symbols", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=32)
    ap.add_argument("--n_keys", type=int, default=64)
    ap.add_argument("--n_vals", type=int, default=16)
    # Tier 2 task knobs
    ap.add_argument("--n_digits", type=int, default=3, help="addition / tooluse")
    ap.add_argument("--n_shots", type=int, default=4, help="incontext")
    ap.add_argument("--mode", default="shift", choices=["shift", "permutation"],
                    help="incontext: shift=generalize to unseen query, permutation=recall")
    ap.add_argument("--hops", type=int, default=3, help="multihop")
    ap.add_argument("--n_vars", type=int, default=8, help="statetrack")
    ap.add_argument("--n_ops", type=int, default=12, help="statetrack")
    # model knobs (smaller than the LM by default — these tasks are easy)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_kv_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--block_size", type=int, default=None,
                    help="defaults to the task's sequence length")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    task = build_task_from_args(args)
    # Tool tasks decode in an interactive loop that can wander a few tokens past
    # the canonical length before emitting EOS; give the RoPE cache headroom so
    # eval-time generation isn't truncated. (Training still only uses total_len.)
    headroom = 8 if is_tool_task(task) else 0
    block_size = args.block_size or (task.total_len + headroom)
    assert block_size >= task.total_len, \
        f"block_size {block_size} < task total_len {task.total_len}"
    print(f"task={args.task} vocab={task.vocab_size} total_len={task.total_len} "
          f"prompt_len={task.prompt_len} answer_len={task.answer_len}")

    cfg = ModelConfig(
        vocab_size=task.vocab_size, n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head, n_embd=args.n_embd, block_size=block_size,
    )
    model = TinyLLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    rng = np.random.default_rng(args.seed)
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")

    def save(it):
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "iter": it, "cfg": cfg.__dict__, "task_spec": task.spec(),
        }, ckpt_path)

    model.train()
    t0 = time.time()
    for it in range(args.max_iters):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_lambda(it, args.warmup, args.max_iters)

        x_np, y_np = T.training_batch(task, args.batch_size, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_interval == 0:
            # the tool loop decodes per-sample (slower, esp. before the model
            # learns to emit EOS), so evaluate on fewer samples for tool tasks.
            n_eval = 128 if is_tool_task(task) else 256
            acc, n, msg = eval_task(model, task, device, n_samples=n_eval)
            model.train()
            dt = time.time() - t0
            print(f"iter {it:6d} | loss {loss.item():.4f} | {msg} (n={n}) | {dt:.1f}s",
                  flush=True)

        if it % args.ckpt_interval == 0 and it > 0:
            save(it)

    save(args.max_iters - 1)
    n_final = 512 if is_tool_task(task) else 1024
    acc, n, msg = eval_task(model, task, device, n_samples=n_final)
    print(f"training done. final {msg} (n={n}). saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
