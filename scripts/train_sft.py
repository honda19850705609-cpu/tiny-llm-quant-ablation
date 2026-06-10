"""
Supervised fine-tuning: turn the pretrained TinyStories LM into an instruction
follower.

Unlike scripts/train_task.py (fresh model, symbolic task), this LOADS the
pretrained language-model checkpoint and continues training it on the SFT data
built by scripts/prepare_sft.py, with loss masked to the response tokens only.

Run:
    # 1. data
    python -m scripts.prepare_sft --tokenizer data/tokenizer.json --out_dir data
    # 2. finetune from the pretrained LM checkpoint
    python -m scripts.train_sft --pretrained ckpt/ckpt.pt --data_dir data \
        --ckpt_dir ckpt_sft --max_iters 2000 --lr 1e-4

After training, generate with the chat template via scripts/chat.py.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from tllm.model import TinyLLM, ModelConfig


def load_sft_data(data_dir):
    with open(os.path.join(data_dir, "sft_meta.json")) as f:
        meta = json.load(f)
    max_len, n = meta["max_len"], meta["n"]
    tokens = np.memmap(os.path.join(data_dir, "sft_tokens.bin"), dtype=np.uint16, mode="r")
    mask = np.memmap(os.path.join(data_dir, "sft_mask.bin"), dtype=np.uint8, mode="r")
    tokens = np.asarray(tokens).reshape(n, max_len)
    mask = np.asarray(mask).reshape(n, max_len)
    return tokens, mask, max_len, n


def get_batch(tokens, mask, batch_size, device):
    n = tokens.shape[0]
    ix = np.random.randint(0, n, size=batch_size)
    rows = tokens[ix].astype(np.int64)
    msk = mask[ix].astype(np.int64)
    x = torch.from_numpy(rows[:, :-1]).to(device)
    y_full = rows[:, 1:].copy()
    # loss only where the *target* token is a response token
    y_full[msk[:, 1:] == 0] = -1
    y = torch.from_numpy(y_full).to(device)
    return x, y


def lr_lambda(it, warmup, max_iters, min_ratio=0.1):
    if it < warmup:
        return it / max(1, warmup)
    decay = min(1.0, (it - warmup) / max(1, max_iters - warmup))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * decay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="pretrained LM checkpoint (ckpt.pt)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt_dir", default="ckpt_sft")
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_interval", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    tokens, mask, max_len, n = load_sft_data(args.data_dir)
    print(f"[sft] {n} rows x {max_len} tokens, {int(mask.sum())} response tokens")

    ck = torch.load(args.pretrained, map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    assert cfg.block_size >= max_len, f"model block_size {cfg.block_size} < sft max_len {max_len}"
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"[sft] loaded pretrained from {args.pretrained}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device == "cuda"))
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")

    model.train()
    t0 = time.time()
    for it in range(args.max_iters):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_lambda(it, args.warmup, args.max_iters)
        x, y = get_batch(tokens, mask, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % args.eval_interval == 0:
            print(f"iter {it:5d} | loss {loss.item():.4f} | {time.time()-t0:.1f}s")

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "iter": args.max_iters - 1},
               ckpt_path)
    print(f"[sft] done. saved {ckpt_path}")


if __name__ == "__main__":
    main()
