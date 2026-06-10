"""
Quantization ablation on a TASK model — the experiment the README's "Future
work" called for.

The original run_ablation.py measured perplexity on TinyStories and found KV-
cache INT4 was nearly free, contradicting the prediction that KV quantization
should hurt long-range most. The stated reason: TinyStories has little genuine
long-range dependency, so the distant KV entries quantization corrupts are
barely used.

This script re-runs the SAME quant grid on a task that FORCES long-range
retrieval (KeyValueTask / InductionTask), and measures ACCURACY BY DISTANCE
instead of perplexity by position. If the original explanation is right, KV-INT4
accuracy should now fall off at large retrieval distances while weight-INT4
degrades roughly uniformly — i.e. the prediction should finally hold *here*.

Run:
    python -m scripts.run_task_ablation --ckpt_dir ckpt_kv
    python -m scripts.run_task_ablation --ckpt_dir ckpt_ind --out results_ind.json
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
from tllm import tasks as T
from tllm import eval as E


def load_model(ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    task = T.build_task(ck["task_spec"])
    return model, cfg, task


def fresh(state, cfg, device):
    m = TinyLLM(cfg).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


def default_distances(task):
    # name-based on purpose: several tasks share an n_pairs attribute, so a
    # hasattr check would misroute them. Only tasks accuracy_by_distance knows
    # how to pin get a distance axis.
    if task.name == "kv":
        n = task.n_pairs
        return sorted(set(int(d) for d in np.linspace(0, n - 1, min(n, 8))))
    if task.name == "induction":
        L = task.seq_len
        return sorted(set(int(d) for d in np.linspace(2, L - 1, 8)))
    if task.name == "statetrack":
        n = task.n_ops
        return sorted(set(int(d) for d in np.linspace(1, n, min(n, 8))))
    return None   # task has no retrieval-distance axis -> accuracy only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="ckpt_kv")
    ap.add_argument("--out", default="results_task.json")
    ap.add_argument("--n_samples", type=int, default=1024)
    ap.add_argument("--n_per_dist", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, cfg, task = load_model(args.ckpt_dir, device)
    state = copy.deepcopy(base_model.state_dict())
    distances = default_distances(task)
    print(f"task={task.name} distances={distances}")

    is_tool = getattr(task, "name", None) == "tooluse"

    def evaluate(model, kv_cfg=None):
        unpatch = None
        if kv_cfg is not None and kv_cfg.enabled:
            unpatch = patch_kv_quant(model, kv_cfg)
        if is_tool:
            acc, called, n = E.tool_use_accuracy(model, task, device, n_samples=args.n_samples)
            out = {"acc": acc, "tool_called": called, "n": n, "acc_by_distance": []}
        else:
            acc, n = E.task_accuracy(model, task, device, n_samples=args.n_samples)
            by_dist = []
            if distances is not None:
                by_dist = E.accuracy_by_distance(model, task, device, distances,
                                                 n_per_dist=args.n_per_dist)
            out = {"acc": acc, "n": n, "acc_by_distance": by_dist}
        if unpatch:
            unpatch()
        return out

    results = {}

    print("== baseline fp16 ==")
    results["fp16_baseline"] = evaluate(base_model)
    results["fp16_baseline"]["size_mb"] = E.model_size_bits(base_model, 16)

    print("== weight INT8 ==")
    m = fresh(state, cfg, device)
    quantize_model_weights(m, WeightQuantConfig(n_bits=8))
    r = evaluate(m); r["size_mb"] = E.model_size_bits(m, 8)
    results["weight_int8"] = r

    for g in (128, 64):
        print(f"== weight INT4 g={g} ==")
        m = fresh(state, cfg, device)
        quantize_model_weights(m, WeightQuantConfig(n_bits=4, group_size=g))
        r = evaluate(m); r["size_mb"] = E.model_size_bits(m, 4)
        results[f"weight_int4_g{g}"] = r

    for bits in (8, 4):
        print(f"== kv-cache INT{bits} ==")
        m = fresh(state, cfg, device)
        r = evaluate(m, KVQuantConfig(enabled=True, n_bits=bits))
        r["size_mb"] = E.model_size_bits(m, 16)
        results[f"kv_int{bits}"] = r

    print("== combo w-int8 + kv-int8 ==")
    m = fresh(state, cfg, device)
    quantize_model_weights(m, WeightQuantConfig(n_bits=8))
    r = evaluate(m, KVQuantConfig(enabled=True, n_bits=8))
    r["size_mb"] = E.model_size_bits(m, 8)
    results["combo_w8_kv8"] = r

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)

    # ---- table ----
    print(f"\n{'config':<20} {'acc':>8} {'size_MB':>8}")
    for k, v in results.items():
        print(f"{k:<20} {v['acc']:>8.3f} {v.get('size_mb', 0):>8.2f}")

    # ---- plot accuracy vs distance (only if the task has a distance axis) ----
    if distances is None:
        print(f"(task '{task.name}' has no retrieval-distance axis; accuracy-only)")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for k, v in results.items():
            pts = v["acc_by_distance"]
            xs = [d for d, _ in pts]
            ys = [a for _, a in pts]
            plt.plot(xs, ys, marker="o", label=k)
        plt.xlabel("retrieval distance (pairs back / induction gap)")
        plt.ylabel("exact-match accuracy")
        plt.title(f"Accuracy vs distance under compression — task={task.name}")
        plt.ylim(-0.02, 1.02)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = args.out.replace(".json", "") + "_acc_by_distance.png"
        plt.savefig(out_png, dpi=130)
        print("wrote", out_png)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
