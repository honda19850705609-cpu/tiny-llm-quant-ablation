"""
Quantization toolkit for the ablation study.

Two independent axes you can ablate:

  (A) WEIGHT quantization  -> quantize_model_weights()
        - INT8 per-channel, INT4 grouped (group size configurable)
        - simulated quantization: weights are quantized then dequantized back
          to fp, so you measure the *accuracy effect* of the precision loss
          without needing a custom INT4 kernel. This is the standard way to
          study quantization sensitivity (it isolates the error, not the
          hardware speedup).

  (B) KV-CACHE quantization -> KVQuantConfig + patched generation
        - quantize keys/values to INT8/INT4 per-token before caching.
        - this is the axis most likely to give you a NON-OBVIOUS finding,
          e.g. long-range dependency degrades faster than short-range.

The research question this file is built to answer:
    "Which capability degrades first, and under which compression axis?"

Keep notes on every run. The number that matters is not 'how small' but
'where does it break, and is that break surprising?'
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# core quant primitives (symmetric, signed)
# ---------------------------------------------------------------------------
def _quantize_dequantize(w: torch.Tensor, n_bits: int, dim: int = -1, group_size: int = None):
    """Symmetric per-channel (or per-group) quantize-then-dequantize.
    Returns a tensor of the same shape/dtype as w, with precision reduced.
    """
    qmax = 2 ** (n_bits - 1) - 1
    orig_shape = w.shape

    if group_size is not None and group_size > 0:
        # group along the INPUT dim (last dim of an nn.Linear weight: [out, in])
        in_dim = w.shape[-1]
        if in_dim % group_size != 0:
            # fall back to per-channel rather than crash on small layers
            return _quantize_dequantize(w, n_bits, dim=0, group_size=None)
        w2 = w.reshape(-1, group_size)
        scale = w2.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)
        q = torch.clamp(torch.round(w2 / scale), -qmax - 1, qmax)
        return (q * scale).reshape(orig_shape)

    # per-channel along `dim`
    reduce_dims = [d for d in range(w.ndim) if d != (dim % w.ndim)]
    scale = w.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale


@dataclass
class WeightQuantConfig:
    n_bits: int = 8           # 8 or 4
    group_size: int = None    # None=per-channel; e.g. 64 or 128 for grouped INT4
    skip_layers: tuple = ()   # names containing any of these substrings are skipped
    only_layers: tuple = ()   # if set, ONLY layers matching these are quantized


def quantize_model_weights(model: nn.Module, cfg: WeightQuantConfig):
    """In-place simulated quantization of all nn.Linear weights.
    Returns dict of per-layer relative error for logging.
    """
    errors = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if cfg.only_layers and not any(s in name for s in cfg.only_layers):
                continue
            if any(s in name for s in cfg.skip_layers):
                continue
            w = mod.weight.data
            wq = _quantize_dequantize(w, cfg.n_bits, dim=0, group_size=cfg.group_size)
            rel_err = (wq - w).norm() / (w.norm() + 1e-8)
            errors[name] = float(rel_err)
            mod.weight.data.copy_(wq)
    return errors


# ---------------------------------------------------------------------------
# KV-cache quantization
# ---------------------------------------------------------------------------
@dataclass
class KVQuantConfig:
    enabled: bool = False
    n_bits: int = 8           # 8 or 4
    quant_keys: bool = True
    quant_values: bool = True


def quantize_kv(tensor: torch.Tensor, n_bits: int):
    """Per-token (last-dim) symmetric quant-dequant of a K or V cache tensor.
    tensor: (B, n_kv_head, T, head_dim)
    """
    qmax = 2 ** (n_bits - 1) - 1
    scale = tensor.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax)
    return q * scale


def patch_kv_quant(model, kv_cfg: KVQuantConfig):
    """Monkey-patch each attention block's forward so that whatever gets stored
    into the KV cache is quantized first. Returns an unpatch() callable.

    This wraps the existing Attention.forward; the model code itself stays clean.
    """
    from tllm.model import Attention
    originals = {}

    def make_wrapper(orig_forward):
        def wrapper(self, x, cos, sin, kv_cache=None):
            out, new_cache = orig_forward(x, cos, sin, kv_cache)
            if kv_cfg.enabled and new_cache is not None:
                k, v = new_cache
                if k is not None and kv_cfg.quant_keys:
                    k = quantize_kv(k, kv_cfg.n_bits)
                if v is not None and kv_cfg.quant_values:
                    v = quantize_kv(v, kv_cfg.n_bits)
                new_cache = (k, v)
            return out, new_cache
        return wrapper

    for name, mod in model.named_modules():
        if isinstance(mod, Attention):
            originals[name] = mod.forward
            mod.forward = make_wrapper(mod.forward).__get__(mod, Attention)

    def unpatch():
        for name, mod in model.named_modules():
            if name in originals:
                mod.forward = originals[name]
    return unpatch
