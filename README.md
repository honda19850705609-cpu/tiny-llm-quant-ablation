# tiny-llm-quant-ablation

**A small from-scratch LLM, used as a controlled testbed for one question:
when you compress a language model, *which capability breaks first, and is the
break where you'd expect?***

This is not another "I added RoPE to nanoGPT" repo. The architecture
(RoPE + RMSNorm + SwiGLU + GQA + KV-cache, Llama-3 style) is deliberately
standard and ~300 lines — it exists only so that the *ablation* on top of it is
clean and fully understood. The contribution is the experimental design and the
findings, not the model.

---

## TL;DR finding

> _(Fill this in once you've run on TinyStories. The template below is the
> shape of the claim you're looking for — replace with your actual numbers.)_

**Weight INT4 and KV-cache INT4 cost roughly the same in *average* perplexity,
but they degrade *different parts* of the context.** Per-position analysis shows
KV-cache INT4 leaves early-context prediction nearly untouched while late-context
perplexity rises by X%, whereas weight INT4 degrades uniformly. In other words,
**average perplexity hides where the model actually breaks** — KV-cache
quantization is disproportionately a *long-range* tax.

See `ppl_by_position.png` and the table below.

---

## Why per-position perplexity

A single perplexity number tells you *how much* a model degraded, not *what
degraded*. By bucketing perplexity over token position in the context, you can
separate short-range fluency from long-range dependency. Compression methods
that look equivalent on average can diverge sharply here — and that divergence
is the actual result.

This methodology is a direct port of a "precision-first, compression-second"
research workflow I used in my undergraduate thesis on UAV object detection,
where the same logic applied: aggregate mAP hid which object *classes* the
compression hurt.

## Architecture (`tllm/model.py`)

| component | choice |
|---|---|
| positional encoding | RoPE |
| normalization | RMSNorm (pre-norm) |
| MLP | SwiGLU (2/3 · 4d hidden) |
| attention | Grouped-Query Attention + KV-cache |
| tied embeddings | yes |

Default config: 8 layers, d=512, 8 heads / 4 KV heads, ctx 512 — about 30M
non-embedding params. Trains on TinyStories on a single Colab L4 in a few hours.

## Ablation axes (`tllm/quant.py`)

- **Weight quantization** — INT8 per-channel, INT4 grouped (g=128 / g=64),
  simulated (quant→dequant) to isolate the *accuracy effect* of precision loss.
- **KV-cache quantization** — INT8 / INT4, per-token, applied to keys and/or
  values as they enter the cache.
- **Combinations** — e.g. weight-INT8 + KV-INT8.

## Results

| config | ppl | size (MB) | mean weight err |
|---|---|---|---|
| fp16 baseline | – | – | – |
| weight INT8 | – | – | – |
| weight INT4 g128 | – | – | – |
| weight INT4 g64 | – | – | – |
| KV INT8 | – | – | – |
| KV INT4 | – | – | – |
| combo w8+kv8 | – | – | – |

_(populated by `scripts/run_ablation.py` → `results.json`)_

## Reproduce

```bash
# 1. data: download TinyStories, train tokenizer, tokenize to memmap
python -m scripts.prepare_data --out_dir data --vocab_size 8192

# 2. train (auto-resumes; point ckpt_dir at Google Drive on Colab)
python -m scripts.train --data_dir data --ckpt_dir ckpt --max_iters 20000

# 3. run the ablation grid -> results.json + ppl_by_position.png
python -m scripts.run_ablation --ckpt_dir ckpt --data_dir data
```

### Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <your-repo-url> && cd tiny-llm-quant-ablation
!pip install -q torch datasets tokenizers matplotlib
!python -m scripts.prepare_data --out_dir /content/drive/MyDrive/tllm-data
!python -m scripts.train --data_dir /content/drive/MyDrive/tllm-data \
    --ckpt_dir /content/drive/MyDrive/tllm-ckpt --max_iters 20000
```

Checkpoints land in Drive, so a disconnected session resumes by re-running the
same command.

## Project structure

```
tllm/
  model.py     # RoPE / RMSNorm / SwiGLU / GQA / KV-cache transformer
  quant.py     # weight + KV-cache quantization
  eval.py      # perplexity, per-position perplexity, latency/memory
scripts/
  prepare_data.py   # TinyStories -> tokenizer -> memmap
  train.py          # training loop w/ Drive checkpointing + resume
  run_ablation.py   # the experiment grid -> results.json + plot
```

## Notes & honesty

- Weight quantization here is **simulated** (quant→dequant in fp). This measures
  the *accuracy* cost of low precision; it does not produce a real packed INT4
  kernel or wall-clock speedup. That's the correct tool for a *sensitivity*
  study — speedup is a separate, hardware-dependent question.
- TinyStories is a simple domain. Findings about *which* capability degrades are
  suggestive, not a claim about frontier models. The methodology transfers; the
  exact numbers don't.

## License

MIT
