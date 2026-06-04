"""
Training loop for TinyLLM.

Designed for Colab: checkpoints to a directory you can point at Google Drive,
auto-resumes if a checkpoint exists, uses AMP (bf16/fp16) and gradient
accumulation so it fits on an L4.

Typical Colab run:
    from google.colab import drive; drive.mount('/content/drive')
    !python -m scripts.train \
        --data_dir data \
        --ckpt_dir /content/drive/MyDrive/tiny-llm-ckpt \
        --max_iters 20000

Resume is automatic: re-run the same command after a disconnect.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from tllm.model import TinyLLM, ModelConfig


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=50):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_lambda(it, warmup, max_iters, min_ratio=0.1):
    if it < warmup:
        return it / max(1, warmup)
    if it > max_iters:
        return min_ratio
    decay = (it - warmup) / (max_iters - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * decay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--max_iters", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--ckpt_interval", type=int, default=500)
    # model size knobs
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_kv_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={device} dtype={dtype}")

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # infer vocab size from tokenizer
    tok_json = os.path.join(args.data_dir, "tokenizer.json")
    vocab_size = 8192
    if os.path.exists(tok_json):
        with open(tok_json) as f:
            vocab_size = len(json.load(f)["model"]["vocab"])
    print("vocab_size:", vocab_size)

    cfg = ModelConfig(
        vocab_size=vocab_size, n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head, n_embd=args.n_embd, block_size=args.block_size,
    )
    model = TinyLLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # ---- resume ----
    start_iter = 0
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        print("resuming from", ckpt_path)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        print("resumed at iter", start_iter)

    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_lambda(it, args.warmup, args.max_iters)

        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if it % args.eval_interval == 0:
            vl = estimate_loss(model, val_data, args.block_size, args.batch_size, device)
            dt = time.time() - t0
            print(f"iter {it:6d} | val loss {vl:.4f} | ppl {math.exp(vl):.2f} | {dt:.1f}s")

        if it % args.ckpt_interval == 0 and it > start_iter:
            torch.save({
                "model": model.state_dict(), "opt": opt.state_dict(),
                "iter": it, "cfg": cfg.__dict__,
            }, ckpt_path)

    # final save
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "iter": args.max_iters - 1, "cfg": cfg.__dict__,
    }, ckpt_path)
    print("training done. saved", ckpt_path)


if __name__ == "__main__":
    main()
