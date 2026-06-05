"""
Prepare TinyStories for training.

Steps:
  1. download a slice of TinyStories from HuggingFace
  2. train a small byte-level BPE tokenizer on it
  3. tokenize the whole corpus into a single uint16 memmap (.bin) for fast loading

Run once:  python -m scripts.prepare_data --out_dir data --vocab_size 8192

Notes for Colab:
  - the .bin files are small; copy them to your Drive so you don't re-tokenize
    every session.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--n_train_docs", type=int, default=200_000,
                    help="how many stories to use (full set is ~2.1M)")
    ap.add_argument("--n_val_docs", type=int, default=2_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    print("loading TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    train_docs = ds.select(range(min(args.n_train_docs, len(ds))))
    val_docs = ds.select(range(len(ds) - args.n_val_docs, len(ds)))

    # ---- train tokenizer ----
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE tokenizer ...")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["<unk>", "<bos>", "<eos>"],
        )
        def corpus_iter():
            for i in range(0, len(train_docs), 1000):
                yield [train_docs[j]["text"] for j in range(i, min(i + 1000, len(train_docs)))]
        # flatten batches
        def flat_iter():
            for batch in corpus_iter():
                for t in batch:
                    yield t
        tokenizer.train_from_iterator(flat_iter(), trainer=trainer)
        tokenizer.save(tok_path)
    else:
        tokenizer = Tokenizer.from_file(tok_path)
    print("vocab size:", tokenizer.get_vocab_size())

    eos_id = tokenizer.token_to_id("<eos>")

    def tokenize_split(docs, name):
        ids_all = []
        for i in range(len(docs)):
            ids = tokenizer.encode(docs[i]["text"]).ids
            ids.append(eos_id)
            ids_all.extend(ids)
            if i % 20000 == 0:
                print(f"  {name}: {i}/{len(docs)} docs, {len(ids_all)} tokens")
        arr = np.array(ids_all, dtype=np.uint16)
        path = os.path.join(args.out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"  wrote {path}: {len(arr):,} tokens")

    tokenize_split(train_docs, "train")
    tokenize_split(val_docs, "val")
    print("done.")


if __name__ == "__main__":
    main()
