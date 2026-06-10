"""
Build an SFT dataset for instruction tuning.

Reads instruction records (Alpaca format: {"instruction", "input", "output"})
and encodes them with the SAME BPE tokenizer used for pretraining
(data/tokenizer.json), applying the chat template + response-only loss mask from
tllm/sft.py. Writes three files into --out_dir:

    sft_tokens.bin   uint16, shape (N * max_len,)   packed fixed-length rows
    sft_mask.bin     uint8,  shape (N * max_len,)   1 = response token (train on it)
    sft_meta.json    {"max_len": ..., "n": ...}

Input options:
    --jsonl PATH        a local JSONL of Alpaca records (one JSON object per line)
    --hf_dataset NAME   a HuggingFace instruction dataset (default: tatsu-lab/alpaca)
    (neither)           writes + uses a tiny built-in demo set, so the whole SFT
                        path is runnable end-to-end before you bring real data.

Run:
    python -m scripts.prepare_sft --tokenizer data/tokenizer.json --out_dir data
"""

import argparse
import json
import os

import numpy as np

from tllm.sft import build_example


DEMO_RECORDS = [
    {"instruction": "Say hello to the reader.", "input": "",
     "output": "Hello! It is nice to meet you."},
    {"instruction": "Write a one-sentence story about a cat.", "input": "",
     "output": "The little cat curled up by the fire and fell fast asleep."},
    {"instruction": "Repeat the input word three times.", "input": "dog",
     "output": "dog dog dog"},
    {"instruction": "Give a friendly good morning.", "input": "",
     "output": "Good morning! I hope you have a wonderful day."},
]


def load_records(args):
    if args.jsonl:
        with open(args.jsonl) as f:
            return [json.loads(line) for line in f if line.strip()]
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train")
        n = min(args.max_records, len(ds)) if args.max_records else len(ds)
        return [dict(ds[i]) for i in range(n)]
    # fallback: write + use the demo set so the path is runnable
    demo_path = os.path.join(args.out_dir, "sft_demo.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(demo_path, "w") as f:
        for r in DEMO_RECORDS:
            f.write(json.dumps(r) + "\n")
    print(f"[prepare_sft] no dataset given; wrote demo set -> {demo_path}")
    return list(DEMO_RECORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--max_records", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_records(args)
    print(f"[prepare_sft] {len(records)} records")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    encode = lambda s: tok.encode(s).ids

    all_tokens, all_mask, kept = [], [], 0
    for r in records:
        toks, mask = build_example(r, encode, bos_id, eos_id, args.max_len)
        if sum(mask) == 0:        # response got fully truncated away — skip
            continue
        all_tokens.extend(toks)
        all_mask.extend(mask)
        kept += 1

    tokens = np.asarray(all_tokens, dtype=np.uint16)
    mask = np.asarray(all_mask, dtype=np.uint8)
    tokens.tofile(os.path.join(args.out_dir, "sft_tokens.bin"))
    mask.tofile(os.path.join(args.out_dir, "sft_mask.bin"))
    with open(os.path.join(args.out_dir, "sft_meta.json"), "w") as f:
        json.dump({"max_len": args.max_len, "n": kept}, f)
    print(f"[prepare_sft] wrote {kept} rows x {args.max_len} tokens "
          f"({sum(all_mask)} response tokens) -> {args.out_dir}")


if __name__ == "__main__":
    main()
