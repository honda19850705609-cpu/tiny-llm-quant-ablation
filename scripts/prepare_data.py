"""
Prepare a pretraining corpus: download text, train a byte-level BPE tokenizer,
tokenize into a uint16 memmap (.bin) for fast training.

Two regimes:
  - TinyStories (default): a tiny children's-story corpus. Fluent but no world
    knowledge — good for the quant-ablation testbed, useless as an assistant.
  - General web text (fineweb-edu / openwebtext): streamed up to a token budget.
    Gives the model real vocabulary + some world knowledge, so an SFT'd model
    can actually attempt instructions (roughly GPT-2-small territory, scale
    permitting).

Run:
  # tiny testbed corpus
  python -m scripts.prepare_data --out_dir data --vocab_size 8192

  # general corpus for a chat-capable model (needs more vocab + a token budget)
  python -m scripts.prepare_data --dataset fineweb-edu --out_dir data \
      --vocab_size 16384 --target_tokens 300_000_000

Notes for Colab:
  - General corpora are streamed (no full download). With a high-RAM runtime the
    buffered text fits in memory; the .bin files are what you keep.
  - For training, read the .bin from LOCAL disk (/content), not Drive — memmap
    does heavy random I/O and Drive drops under sustained load.
"""

import argparse
import os

import numpy as np

# name -> (hf_repo, config_or_None). All expose a "text" field.
DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "openwebtext": ("Skylion007/openwebtext", None),
}


def collect_texts(args):
    """Return (train_texts, val_texts) as lists of strings.

    TinyStories is loaded fully (it's small). General corpora are STREAMED and
    buffered until ~target_tokens worth of characters have been seen (≈4 chars
    per token), so we never download the whole multi-GB shard.
    """
    from datasets import load_dataset
    repo, config = DATASETS[args.dataset]

    if args.dataset == "tinystories":
        ds = load_dataset(repo, split="train")
        n_train = min(args.n_train_docs, len(ds))
        train = [ds[i]["text"] for i in range(n_train)]
        val = [ds[i]["text"] for i in range(len(ds) - args.n_val_docs, len(ds))]
        return train, val

    # general corpus: stream until the character budget is hit
    ds = load_dataset(repo, config, split="train", streaming=True)
    char_budget = args.target_tokens * 4          # rough chars-per-token
    texts, seen = [], 0
    for ex in ds:
        t = ex.get("text") or ""
        if len(t) < 32:                            # skip empty / tiny fragments
            continue
        texts.append(t)
        seen += len(t)
        if len(texts) % 50_000 == 0:
            print(f"  streamed {len(texts):,} docs, ~{seen/1e6:.0f}M chars")
        if seen >= char_budget:
            break
    print(f"  buffered {len(texts):,} docs (~{seen/1e6:.0f}M chars)")
    n_val = min(args.n_val_docs, max(1, len(texts) // 50))
    return texts[:-n_val], texts[-n_val:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinystories", choices=list(DATASETS))
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--n_train_docs", type=int, default=200_000,
                    help="TinyStories only: how many stories to use")
    ap.add_argument("--n_val_docs", type=int, default=2_000)
    ap.add_argument("--target_tokens", type=int, default=300_000_000,
                    help="general corpora: approx token budget to stream")
    ap.add_argument("--tokenizer_sample_docs", type=int, default=80_000,
                    help="how many docs to train the BPE tokenizer on")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print(f"loading dataset '{args.dataset}' ...")
    train_texts, val_texts = collect_texts(args)
    print(f"train docs: {len(train_texts):,} | val docs: {len(val_texts):,}")

    # ---- train tokenizer ----
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"training BPE tokenizer (vocab={args.vocab_size}) ...")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["<unk>", "<bos>", "<eos>"],
        )
        sample = train_texts[:args.tokenizer_sample_docs]

        def batch_iter():
            for i in range(0, len(sample), 1000):
                yield sample[i:i + 1000]
        tokenizer.train_from_iterator(batch_iter(), trainer=trainer)
        tokenizer.save(tok_path)
    else:
        print("tokenizer.json exists; reusing")
        tokenizer = Tokenizer.from_file(tok_path)
    print("vocab size:", tokenizer.get_vocab_size())

    eos_id = tokenizer.token_to_id("<eos>")

    def tokenize_split(texts, name):
        ids_all = []
        # encode in batches (the Rust tokenizer is fast and releases the GIL)
        B = 1000
        for i in range(0, len(texts), B):
            for enc in tokenizer.encode_batch(texts[i:i + B]):
                ids_all.extend(enc.ids)
                ids_all.append(eos_id)
            if (i // B) % 50 == 0:
                print(f"  {name}: {i:,}/{len(texts):,} docs, {len(ids_all)/1e6:.1f}M tokens")
        arr = np.array(ids_all, dtype=np.uint16)
        path = os.path.join(args.out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"  wrote {path}: {len(arr):,} tokens")

    tokenize_split(train_texts, "train")
    tokenize_split(val_texts, "val")
    print("done.")


if __name__ == "__main__":
    main()
