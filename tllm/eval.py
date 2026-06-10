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
def perplexity(model, data, block_size, device, n_batches=100, batch_size=16):
    model.eval()
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
def perplexity_by_position(model, data, block_size, device, n_batches=200, batch_size=16, n_buckets=8):
    """Return list of (bucket_start, bucket_end, ppl) over context positions."""
    model.eval()
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


# ---------------------------------------------------------------------------
# TASK metrics — does the model get the answer RIGHT? (not just "how surprised")
#
# Perplexity measures fluency; these measure task completion. They drive the
# Phase 1/2 experiments (algorithmic tasks, long-range retrieval) where the
# question is "can a *compressed* model still DO the task", and where the
# KV-cache ablation can finally show a long-range tax.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _greedy_answers(model, prompts, answer_len, device):
    """Greedy-decode `answer_len` tokens for a batch of equal-length prompts.
    prompts: int64 tensor (B, P). Returns generated tokens (B, answer_len)."""
    model.eval()
    idx = prompts.to(device)
    P = idx.size(1)
    # top_k=1 + temperature=1 == argmax == greedy, so the metric is deterministic
    out = model.generate(idx, max_new_tokens=answer_len, temperature=1.0,
                         top_k=1, use_cache=True)
    return out[:, P:P + answer_len]


@torch.no_grad()
def task_accuracy(model, task, device, n_samples=512, batch_size=64, seed=1234):
    """Fraction of samples whose greedily-decoded answer exactly matches.

    Returns (accuracy, n_evaluated). Exact-match per sample by default (a task
    may override Task.score, but accuracy here uses full-answer equality which
    is the strict, honest metric for these deterministic tasks).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    answer_len = task.answer_len
    correct, total = 0, 0
    while total < n_samples:
        bs = min(batch_size, n_samples - total)
        samples = [task.sample(rng) for _ in range(bs)]
        prompts = torch.tensor([s.prompt_ids for s in samples], dtype=torch.long)
        gen = _greedy_answers(model, prompts, answer_len, device).cpu().tolist()
        for s, pred in zip(samples, gen):
            correct += int(task.score(pred, s.answer_ids))
            total += 1
    return correct / total, total


@torch.no_grad()
def accuracy_by_distance(model, task, device, distances, n_per_dist=128,
                         batch_size=64, seed=1234):
    """The long-range scalpel: accuracy bucketed by retrieval distance.

    `task` must accept a distance control in sample() — KeyValueTask(query_pos)
    or InductionTask(gap). For each requested distance we generate fresh samples
    pinned to that distance and measure exact-match accuracy. This is the
    accuracy analogue of perplexity_by_position, and it is what would reveal a
    KV-cache INT4 long-range tax if one exists.

    Returns list of (distance, accuracy).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    answer_len = task.answer_len

    # how to pin a sample to a target distance, per task
    def make_sample(dist):
        if task.name == "kv":                              # KeyValueTask
            query_pos = task.n_pairs - 1 - dist
            return task.sample(rng, query_pos=query_pos)
        if task.name == "induction":
            return task.sample(rng, gap=dist)
        if task.name == "statetrack":                      # recency = ops since last set
            return task.sample(rng, recency=dist)
        raise ValueError(f"task {task.name} has no distance control")

    out = []
    for dist in distances:
        correct, total = 0, 0
        while total < n_per_dist:
            bs = min(batch_size, n_per_dist - total)
            samples = [make_sample(dist) for _ in range(bs)]
            prompts = torch.tensor([s.prompt_ids for s in samples], dtype=torch.long)
            gen = _greedy_answers(model, prompts, answer_len, device).cpu().tolist()
            for s, pred in zip(samples, gen):
                correct += int(task.score(pred, s.answer_ids))
                total += 1
        out.append((int(dist), correct / total))
    return out


# ---------------------------------------------------------------------------
# Tool use — the agentic loop: the model emits a tool call, an executor runs it,
# the result is injected back, and the model continues. Plain greedy decoding
# (task_accuracy) cannot evaluate this because it never runs the tool, so tool
# tasks get their own eval that actually closes the generate->execute->inject
# loop. Used for ToolUseTask (exposes CALC/CALL_END/RESULT/ANS/EOS ids and an
# executor(call_ids) method).
# ---------------------------------------------------------------------------
def _span_between(tokens, start_id, end_id):
    """Tokens strictly between the first start_id and the next end_id after it.
    Returns [] if start_id is absent. Runs to list end if end_id is absent."""
    if start_id not in tokens:
        return []
    i = tokens.index(start_id)
    rest = tokens[i + 1:]
    if end_id in rest:
        rest = rest[:rest.index(end_id)]
    return rest


@torch.no_grad()
def tool_use_accuracy(model, task, device, n_samples=256, max_steps=64, seed=1234):
    """Run the agentic loop per sample and score the final ANS span.

    For each sample: greedily decode; when the model emits CALL_END, parse the
    tokens since the last CALC, run task.executor() to get the injected
    [RESULT]+digits, splice them in (these are NOT model-generated), and keep
    decoding until EOS. Score = does the model's ANS..EOS span match the
    sample's ground-truth ANS span. Also reports how often a well-formed tool
    call was actually emitted.

    Returns (accuracy, frac_called, n).
    """
    import numpy as np
    model.eval()
    rng = np.random.default_rng(seed)
    correct = called = 0
    for _ in range(n_samples):
        s = task.sample(rng)
        gt_ans = _span_between(list(s.answer_ids), task.ANS, task.EOS)
        seq = list(s.prompt_ids)
        did_call = False
        bs = model.cfg.block_size
        for _ in range(max_steps):
            # hard cap: an under-trained model can ramble without emitting EOS;
            # never let the sequence exceed the RoPE cache (block_size).
            if len(seq) >= bs:
                break
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(x)
            nxt = int(logits[0, -1].argmax())
            seq.append(nxt)
            if nxt == task.CALL_END and not did_call:
                # call payload = tokens after the last CALC up to this CALL_END
                ci = len(seq) - 1
                cstarts = [i for i in range(ci) if seq[i] == task.CALC]
                if cstarts:
                    payload = seq[cstarts[-1] + 1:ci]
                    seq.extend(task.executor(payload))   # inject [RESULT]+digits
                    did_call = True
            if nxt == task.EOS:
                break
        pred_ans = _span_between(seq[len(s.prompt_ids):], task.ANS, task.EOS)
        correct += int(pred_ans == gt_ans and len(gt_ans) > 0)
        called += int(did_call)
    return correct / n_samples, called / n_samples, n_samples
