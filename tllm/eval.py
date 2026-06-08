"""
Evaluation harness — produces the numbers that go on the README.

Metrics:
  - overall validation perplexity
  - PER-POSITION perplexity: ppl bucketed by token position in the context.
    This is your scalpel for finding the non-obvious result: if KV-cache INT4
    leaves early-position ppl untouched but blows up late-position ppl, that's
    a long-range-dependency degradation story — exactly the kind of finding
    that turns a tutorial repo into a research repo.
  - model size on disk (real bytes if you pack, simulated bits otherwise)
  - generation latency + peak memory (CUDA only)

Run:
    python -m scripts.run_ablation --ckpt_dir /content/drive/MyDrive/tiny-llm-ckpt
"""

import math
import time

import numpy as np
import torch


@torch.no_grad()
def perplexity(model, data, block_size, device, n_batches=100, batch_size=16, seed=1234):
    model.eval()
    # fix the sampling RNG so every config sees the SAME eval windows -> the
    # tiny ppl gaps between configs are signal, not sampling noise.
    torch.manual_seed(seed)
    nll, count = 0.0, 0
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        nll += loss.item()
        count += y.numel()
    return math.exp(nll / count)


@torch.no_grad()
def perplexity_by_position(model, data, block_size, device, n_batches=200, batch_size=16, n_buckets=8, seed=1234):
    """Return list of (bucket_start, bucket_end, ppl) over context positions."""
    model.eval()
    torch.manual_seed(seed)   # same windows across configs (see perplexity())
    bucket_nll = np.zeros(n_buckets)
    bucket_cnt = np.zeros(n_buckets)
    bsz = block_size // n_buckets
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        ).view(batch_size, block_size)
        for b in range(n_buckets):
            seg = loss[:, b * bsz:(b + 1) * bsz]
            bucket_nll[b] += seg.sum().item()
            bucket_cnt[b] += seg.numel()
    out = []
    for b in range(n_buckets):
        ppl = math.exp(bucket_nll[b] / bucket_cnt[b])
        out.append((b * bsz, (b + 1) * bsz, ppl))
    return out


@torch.no_grad()
def measure_latency_memory(model, device, prompt_len=16, gen_len=128, use_cache=True):
    if device == "cpu":
        return {"note": "latency/memory only meaningful on CUDA"}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    idx = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    # warmup
    model.generate(idx, 8, use_cache=use_cache)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(idx, gen_len, use_cache=use_cache)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    return {
        "tokens_per_sec": gen_len / dt,
        "latency_s": dt,
        "peak_mem_mb": peak,
    }


def model_size_bits(model, weight_bits=16):
    """Effective stored size if all Linear weights were `weight_bits`."""
    total = 0
    for n, p in model.named_parameters():
        bits = weight_bits if "weight" in n else 16
        total += p.numel() * bits
    return total / 8 / 1e6  # MB
