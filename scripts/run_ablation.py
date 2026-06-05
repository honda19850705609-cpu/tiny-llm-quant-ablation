"""
Run the full ablation grid and dump results to results.json (+ plots).

This is the script that produces your GitHub-visible findings.

It sweeps:
  - FP16 baseline
  - weight INT8 per-channel
  - weight INT4 grouped (g=128, g=64)
  - KV-cache INT8 / INT4
  - a couple of combinations

For each config it records: overall ppl, per-position ppl, effective size.
Then it plots per-position ppl curves so the long-vs-short-range story is
visible at a glance.

Run:
    python -m scripts.run_ablation --ckpt_dir /content/drive/MyDrive/tiny-llm-ckpt --data_dir data
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

from tllm.model import TinyLLM, ModelConfig
from tllm.quant import (
    WeightQuantConfig, quantize_model_weights,
    KVQuantConfig, patch_kv_quant,
)
from tllm import eval as E


def load_model(ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def fresh(model_state, cfg, device):
    m = TinyLLM(cfg).to(device)
    m.load_state_dict(model_state)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--n_batches", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, cfg = load_model(args.ckpt_dir, device)
    state = copy.deepcopy(base_model.state_dict())
    block_size = cfg.block_size

    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def evaluate(model, kv_cfg=None):
        unpatch = None
        if kv_cfg is not None and kv_cfg.enabled:
            unpatch = patch_kv_quant(model, kv_cfg)
        ppl = E.perplexity(model, val, block_size, device, n_batches=args.n_batches)
        pos = E.perplexity_by_position(model, val, block_size, device, n_batches=args.n_batches)
        lat = E.measure_latency_memory(model, device)
        if unpatch:
            unpatch()
        return {"ppl": ppl, "ppl_by_pos": pos, "latency": lat}

    results = {}

    # 1. baseline
    print("== baseline fp16 ==")
    results["fp16_baseline"] = evaluate(base_model)
    results["fp16_baseline"]["size_mb"] = E.model_size_bits(base_model, 16)

    # 2. weight INT8 per-channel
    print("== weight INT8 ==")
    m = fresh(state, cfg, device)
    err = quantize_model_weights(m, WeightQuantConfig(n_bits=8))
    r = evaluate(m); r["size_mb"] = E.model_size_bits(m, 8); r["mean_weight_err"] = float(np.mean(list(err.values())))
    results["weight_int8"] = r

    # 3. weight INT4 grouped
    for g in (128, 64):
        print(f"== weight INT4 g={g} ==")
        m = fresh(state, cfg, device)
        err = quantize_model_weights(m, WeightQuantConfig(n_bits=4, group_size=g))
        r = evaluate(m); r["size_mb"] = E.model_size_bits(m, 4)
        r["mean_weight_err"] = float(np.mean(list(err.values())))
        results[f"weight_int4_g{g}"] = r

    # 4. KV-cache INT8 / INT4  (weights stay fp16)
    for bits in (8, 4):
        print(f"== kv-cache INT{bits} ==")
        m = fresh(state, cfg, device)
        r = evaluate(m, KVQuantConfig(enabled=True, n_bits=bits))
        r["size_mb"] = E.model_size_bits(m, 16)
        results[f"kv_int{bits}"] = r

    # 5. combo: weight INT8 + KV INT8
    print("== combo w-int8 + kv-int8 ==")
    m = fresh(state, cfg, device)
    quantize_model_weights(m, WeightQuantConfig(n_bits=8))
    r = evaluate(m, KVQuantConfig(enabled=True, n_bits=8))
    r["size_mb"] = E.model_size_bits(m, 8)
    results["combo_w8_kv8"] = r

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)

    # ---- print a quick table ----
    print(f"\n{'config':<20} {'ppl':>8} {'size_MB':>8}")
    for k, v in results.items():
        print(f"{k:<20} {v['ppl']:>8.3f} {v.get('size_mb', 0):>8.2f}")

    # ---- plot per-position ppl ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for k, v in results.items():
            pos = v["ppl_by_pos"]
            xs = [(a + b) / 2 for a, b, _ in pos]
            ys = [p for _, _, p in pos]
            plt.plot(xs, ys, marker="o", label=k)
        plt.xlabel("token position in context")
        plt.ylabel("perplexity")
        plt.title("Per-position perplexity under different compression")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("ppl_by_position.png", dpi=130)
        print("wrote ppl_by_position.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
