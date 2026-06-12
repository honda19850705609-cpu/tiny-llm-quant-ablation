# Build notes: the full LLM pipeline, end to end (and where it hits a wall)

These notes capture what was learned by taking the same ~300-line architecture
all the way from raw text to a chatting model — and, just as importantly, where
and *why* a small model stops being able to help.

## The pipeline, stage by stage

```
raw text ──①──▶ token ids (.bin) ──②③──▶ pretrained weights ──⑤──▶ instruct model ──⑥──▶ chat
              tokenizer              pretrain                    SFT
```

| stage | what happens | file |
|---|---|---|
| ① data | download text → train byte-level BPE → tokenize to a uint16 memmap | `scripts/prepare_data.py` |
| ② model | embedding → N×(RoPE attention + SwiGLU FFN) → tied LM head | `tllm/model.py` |
| ③ pretrain | the *only* objective is **predict the next token**; AdamW + warmup/cosine LR, grad-accum, bf16/TF32, checkpoint+resume | `scripts/train.py` |
| ④ eval | perplexity; plus task accuracy / accuracy-by-distance | `tllm/eval.py` |
| ⑤ SFT | continue training on (instruction → response) pairs, **loss masked to the response only** | `tllm/sft.py`, `scripts/train_sft.py` |
| ⑥ generate | autoregressive decode with KV-cache + temperature/top-k sampling | `model.generate`, `scripts/chat.py` |

The whole thing is one idea repeated: **next-token prediction**. Pretraining
presses knowledge into the weights with it; generation runs it forward one token
at a time; SFT just re-points it at "answer the instruction."

## Scaling, seen first-hand

Same architecture, two data regimes:

| model | data | "write a story about a robot" |
|---|---|---|
| 78M | TinyStories (children's stories) | *"Ice has been linked in my wert price. This price never realizes…"* — grammatical word-salad |
| 85M | FineWeb-Edu (general web text) | *"Once upon a time, there was a robot named Robotrix who wanted to build a robot. The robot was programmed with an exquisite-looking assistant…"* — coherent, on-topic |

Nearly the same parameter count. The difference is **data**. This is the whole
"scaling" lesson in one comparison: capability is governed far more by *what the
model was trained on* than by the architecture, which never changed.

## The capability wall (why it can chat but can't *do* the task)

The 85M model can produce fluent, on-topic, correctly-*formatted* responses. Ask
it to "write python code" and it emits code-*shaped* text (comments, structure) —
but nothing that runs. Two different capabilities:

- **Fluency + format** — crosses easily at small scale. ✅
- **Correct, open-ended task completion** — needs ~1000× more (params, tokens,
  code/world data, and alignment/RLHF). ❌ at this scale.

The honest ladder:

| capability | needs | reachable solo on Colab? |
|---|---|---|
| plausible, on-topic English | ~100M params, ~1B tokens | ✅ (this repo) |
| reliable simple Q&A (GPT-3.5-ish) | ~7B+, trillions of tokens, RLHF | ❌ |
| reasoning / working code (GPT-4-ish) | frontier scale + compute | ❌ |

## The twist: a small model *can* complete tasks — the trained ones

It is wrong to say "small models can't complete tasks." In the synthetic-task
suite they hit **100% exact-match**: multi-step addition, copy, in-context rule
learning, tool-calling. The distinction is:

- **narrow, well-defined, explicitly trained** → solved perfectly, even tiny;
- **open-ended, world-knowledge, not specifically trained** → needs frontier scale.

Same architecture. The difference is **scale × data × task-specificity**, not the
design. (This is also why content-addressed retrieval — `kv`/`induction` — sits
right at the model's edge and resists training, while positional copy is trivial.)

## What this testbed is actually for

Not for building an assistant (infeasible solo, poor ROI). Its value is doing
*clean, cheap measurements* on the same architecture frontier models use — e.g.
the headline result that [**quantization cost tracks the capability margin, not
the capability**](README.md). The methodology transfers up; only the numbers
don't. That "measure where it breaks, on a small model you fully understand" loop
is the point — and the right starting place for efficient-models research.
