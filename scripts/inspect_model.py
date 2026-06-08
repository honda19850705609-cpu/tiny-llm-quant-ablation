"""
inspect_model.py  ——  one-shot report card for a trained checkpoint.

After training the "capable" model, run this once to SEE everything at a glance,
then use it to understand and test what each architecture piece is doing:

  1. config summary  — which features (MoE / QK-norm / z-loss / softcap) are on,
                       total vs active params.
  2. perplexity      — overall + per-position (your long-vs-short-range scalpel).
  3. expert load     — for MoE: how evenly tokens are spread across experts in
                       each layer. A healthy router is roughly balanced; a few
                       hot experts + many dead ones means routing collapsed
                       (and tells you the aux/z losses weren't doing their job).
  4. samples         — actual generated text, so you can read the model's output.

Run:
    python -m scripts.inspect_model --ckpt_dir ckpt_capable --data_dir data
    python -m scripts.inspect_model --ckpt_dir ckpt_capable --data_dir data \
        --prompts "Once upon a time" "The little robot"
"""

import argparse
import os

import numpy as np
import torch

from tllm.model import TinyLLM, ModelConfig, MoE
from tllm import eval as E


def load_model(ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("iter", "?")


def print_config(cfg, trained_iter):
    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  layers={cfg.n_layer}  d={cfg.n_embd}  heads={cfg.n_head}/{cfg.n_kv_head}(kv)"
          f"  ctx={cfg.block_size}  trained_iter={trained_iter}")
    feats = []
    if cfg.use_moe:
        feats.append(f"MoE({cfg.n_experts} experts, top-{cfg.n_experts_per_tok}, "
                     f"aux={cfg.moe_aux_loss_coef}, z={cfg.router_z_loss_coef})")
    if cfg.use_qk_norm:
        feats.append("QK-Norm")
    if cfg.logit_softcap and cfg.logit_softcap > 0:
        feats.append(f"logit-softcap({cfg.logit_softcap})")
    if cfg.dropout and cfg.dropout > 0:
        feats.append(f"dropout({cfg.dropout})")
    print(f"  features: {', '.join(feats) if feats else 'dense baseline (no extras)'}")


@torch.no_grad()
def expert_load(model, cfg, data, device, n_batches=50, batch_size=16):
    """Per-layer expert dispatch fractions, via forward hooks on each router."""
    moe_layers = [(i, b) for i, b in enumerate(model.blocks) if isinstance(b.ffn, MoE)]
    if not moe_layers:
        return None
    counts = {i: torch.zeros(cfg.n_experts) for i, _ in moe_layers}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):  # out = router_logits (N, n_experts)
            top = out.topk(cfg.n_experts_per_tok, dim=-1).indices.reshape(-1)
            counts[layer_idx] += torch.bincount(top.cpu(), minlength=cfg.n_experts).float()
        return hook

    for i, b in moe_layers:
        handles.append(b.ffn.gate.register_forward_hook(make_hook(i)))

    torch.manual_seed(0)
    bs = cfg.block_size
    for _ in range(n_batches):
        ix = torch.randint(len(data) - bs - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + bs].astype(np.int64)) for i in ix]).to(device)
        model(x)
    for h in handles:
        h.remove()
    return counts


def print_expert_load(counts, cfg):
    print("=" * 60)
    print("EXPERT LOAD  (fraction of routed tokens per expert, ideal = "
          f"{1/cfg.n_experts:.2f})")
    print("=" * 60)
    ideal = 1 / cfg.n_experts
    for layer, c in counts.items():
        frac = (c / c.sum()).tolist()
        # coefficient of variation: 0 = perfectly balanced
        cv = float(np.std(frac) / (np.mean(frac) + 1e-9))
        bar = " ".join(f"{f:.2f}" for f in frac)
        dead = sum(1 for f in frac if f < 0.2 * ideal)
        print(f"  L{layer:<2} [{bar}]  CV={cv:.2f}  dead={dead}")
    print("  (CV near 0 = balanced; high CV or dead>0 = routing collapsed)")


@torch.no_grad()
def sample_text(model, cfg, data_dir, device, prompts, max_new=80):
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("(no tokenizer.json found; skipping text samples)")
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    print("=" * 60)
    print("SAMPLES")
    print("=" * 60)
    for p in prompts:
        ids = tok.encode(p).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new, temperature=0.8, top_k=200)
        text = tok.decode(out[0].tolist())
        print(f"\n  > {p!r}\n    {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="ckpt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n_batches", type=int, default=100)
    ap.add_argument("--prompts", nargs="*", default=["Once upon a time", "The little robot"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, it = load_model(args.ckpt_dir, device)
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    print_config(cfg, it)

    print("=" * 60)
    print("PERPLEXITY")
    print("=" * 60)
    ppl = E.perplexity(model, val, cfg.block_size, device, n_batches=args.n_batches)
    print(f"  overall val ppl: {ppl:.3f}")
    pos = E.perplexity_by_position(model, val, cfg.block_size, device, n_batches=args.n_batches)
    print("  per-position:")
    for a, b, p in pos:
        print(f"    pos {a:4d}-{b:<4d}: {p:.3f}")

    counts = expert_load(model, cfg, val, device, n_batches=max(20, args.n_batches // 2))
    if counts is not None:
        print_expert_load(counts, cfg)

    sample_text(model, cfg, args.data_dir, device, args.prompts)


if __name__ == "__main__":
    main()
