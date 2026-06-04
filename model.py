"""
Modern small LLM: RoPE + RMSNorm + SwiGLU + KV-Cache.
Decoder-only transformer in the Llama-3 style, kept readable on purpose.

The point of this file is not novelty — every component here is standard in
2026-era models. The point is that YOU understand every line, because the
research value of this repo comes from the ablations you run on top of it,
not from the architecture itself.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4          # GQA: set == n_head for plain MHA
    n_embd: int = 512
    block_size: int = 512       # max context length
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # SwiGLU hidden dim uses the 2/3 * 4 * n_embd rule, rounded to multiple of 64
    ffn_hidden: int = None

    def __post_init__(self):
        if self.ffn_hidden is None:
            h = int(2 / 3 * 4 * self.n_embd)
            self.ffn_hidden = ((h + 63) // 64) * 64
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # compute in fp32 for stability, cast back
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device):
    # frequencies for each pair of dims
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)              # (seq_len, head_dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin                            # each (seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[:T].view(1, 1, T, D // 2)
    sin = sin[:T].view(1, 1, T, D // 2)
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).view(B, H, T, D)


# ---------------------------------------------------------------------------
# Attention with GQA + optional KV cache
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = self.n_head // self.n_kv_head

        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # repeat kv heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = kv_cache is None  # during cached generation we attend to all past
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y), new_cache


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Block + full model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class TinyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters()) - self.lm_head.weight.numel()
        print(f"[TinyLLM] non-embedding params: {n_params/1e6:.2f}M")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, use_cache=True):
        self.eval()
        cos = self.rope_cos.to(idx.device)
        sin = self.rope_sin.to(idx.device)

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size:]
                logits, _ = self(idx_cond)
                idx = self._sample(idx, logits, temperature, top_k)
            return idx

        # cached path
        caches = [None] * len(self.blocks)
        x = self.tok_emb(idx)
        pos_cos, pos_sin = cos[:idx.size(1)], sin[:idx.size(1)]
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, pos_cos, pos_sin, kv_cache=(None, None))
        x = self.norm(x)
        logits = self.lm_head(x)
        idx = self._sample(idx, logits, temperature, top_k)

        for step in range(max_new_tokens - 1):
            cur = idx[:, -1:]
            x = self.tok_emb(cur)
            t = idx.size(1) - 1
            pos_cos, pos_sin = cos[t:t + 1], sin[t:t + 1]
            for i, block in enumerate(self.blocks):
                x, caches[i] = block(x, pos_cos, pos_sin, kv_cache=caches[i])
            x = self.norm(x)
            logits = self.lm_head(x)
            idx = self._sample(idx, logits, temperature, top_k)
        return idx

    @staticmethod
    def _sample(idx, logits, temperature, top_k):
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        return torch.cat([idx, nxt], dim=1)
