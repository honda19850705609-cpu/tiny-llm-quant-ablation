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
    # --- MoE (mixture-of-experts) FFN. Off by default, so dense runs are byte-
    # for-byte unchanged and the dense-vs-MoE ablation is clean. ---
    use_moe: bool = False
    n_experts: int = 8            # total experts in each MoE layer
    n_experts_per_tok: int = 2    # top-k experts actually run per token
    moe_aux_loss_coef: float = 0.01   # load-balancing loss weight (training only)

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
# Mixture-of-Experts FFN (Mixtral / Switch style)
# ---------------------------------------------------------------------------
class MoE(nn.Module):
    """Sparse MoE FFN: replace one SwiGLU with `n_experts` of them, plus a router
    that sends each token to only its top-k experts.

    Why this matters: total parameters (capacity) scale with n_experts, but the
    compute per token stays at ~top_k experts. That decoupling of *capacity* from
    *compute* is the whole point of MoE, and it's exactly what the dense-vs-MoE
    ablation measures.

    forward() returns (output, aux_loss):
      - output: same shape as input, the top-k weighted expert mix.
      - aux_loss: load-balancing loss (Switch Transformer eq. 4). Without it the
        router collapses onto a few experts and the rest go dead. Add it to the
        training loss; it is a no-op at eval (we just don't use it there).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        self.aux_coef = cfg.moe_aux_loss_coef
        # router: one logit per expert, per token
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        # each expert is just a standard SwiGLU FFN
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)                       # (N, C),  N = B*T
        N = x_flat.shape[0]

        router_logits = self.gate(x_flat)              # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # pick top-k experts per token and renormalize their gate weights to sum 1
        topk_w, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)   # (N, k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # flatten the (token, slot) assignments so we can group by expert
        flat_idx = topk_idx.reshape(-1)                # (N*k,)  which expert
        flat_w = topk_w.reshape(-1)                    # (N*k,)  its weight
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        # dispatch: run each expert once on the tokens routed to it, scatter back
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            sel = flat_idx == e
            if not sel.any():
                continue
            tok = token_ids[sel]                       # tokens going to expert e
            w = flat_w[sel].unsqueeze(-1)              # their gate weights
            out.index_add_(0, tok, self.experts[e](x_flat[tok]) * w)

        # ---- load-balancing aux loss: n * sum_i f_i * P_i ----
        # f_i = fraction of assignments dispatched to expert i (counted, no grad)
        # P_i = mean router probability for expert i (differentiable)
        # balanced -> f_i = P_i = 1/n -> loss = aux_coef; imbalance pushes it up.
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device)
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_w))
            f = counts / flat_idx.numel()
        P = router_probs.mean(dim=0)                   # (n_experts,)
        aux_loss = self.aux_coef * self.n_experts * (f * P).sum()

        return out.reshape(B, T, C), aux_loss


# ---------------------------------------------------------------------------
# Block + full model
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.use_moe = cfg.use_moe
        self.ffn = MoE(cfg) if cfg.use_moe else SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        if self.use_moe:
            ffn_out, aux = self.ffn(self.ffn_norm(x))
        else:
            ffn_out, aux = self.ffn(self.ffn_norm(x)), None
        x = x + ffn_out
        return x, new_cache, aux


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
        print(f"[TinyLLM] total non-embedding params: {n_params/1e6:.2f}M")
        if cfg.use_moe:
            # only top_k of n_experts run per token, so "active" params < total
            expert_params = sum(p.numel() for b in self.blocks for p in b.ffn.experts.parameters())
            inactive = expert_params * (cfg.n_experts - cfg.n_experts_per_tok) / cfg.n_experts
            print(f"[TinyLLM] active params/token: {(n_params - inactive)/1e6:.2f}M  "
                  f"({cfg.n_experts} experts, top-{cfg.n_experts_per_tok})")

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
        aux_total = None
        for block in self.blocks:
            x, _, aux = block(x, cos, sin)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            if aux_total is not None:
                loss = loss + aux_total
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
            x, caches[i], _ = block(x, pos_cos, pos_sin, kv_cache=(None, None))
        x = self.norm(x)
        logits = self.lm_head(x)
        idx = self._sample(idx, logits, temperature, top_k)

        for step in range(max_new_tokens - 1):
            cur = idx[:, -1:]
            x = self.tok_emb(cur)
            t = idx.size(1) - 1
            pos_cos, pos_sin = cos[t:t + 1], sin[t:t + 1]
            for i, block in enumerate(self.blocks):
                x, caches[i], _ = block(x, pos_cos, pos_sin, kv_cache=caches[i])
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
