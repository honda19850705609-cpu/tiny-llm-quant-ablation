"""
Chat with an SFT'd TinyLLM using the same template it was trained on.

Run:
    python -m scripts.chat --ckpt ckpt_sft/ckpt.pt --tokenizer data/tokenizer.json \
        --instruction "Write a short story about a robot."

    # interactive
    python -m scripts.chat --ckpt ckpt_sft/ckpt.pt --tokenizer data/tokenizer.json
"""

import argparse

import torch

from tllm.model import TinyLLM, ModelConfig
from tllm.sft import format_prompt


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["cfg"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def respond(model, tok, instruction, input_text, device, max_new=200,
            temperature=0.7, top_k=100):
    bos_id = tok.token_to_id("<bos>")
    eos_id = tok.token_to_id("<eos>")
    prompt = format_prompt(instruction, input_text)
    ids = ([bos_id] if bos_id is not None else []) + tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, use_cache=True)
    gen = out[0, len(ids):].tolist()
    if eos_id is not None and eos_id in gen:
        gen = gen[:gen.index(eos_id)]
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--input", default="")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    model = load(args.ckpt, device)

    if args.instruction is not None:
        print(respond(model, tok, args.instruction, args.input, device,
                      max_new=args.max_new, temperature=args.temperature))
        return

    print("Interactive chat (Ctrl-C to exit).")
    while True:
        try:
            instr = input("\ninstruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not instr:
            continue
        print(respond(model, tok, instr, "", device,
                      max_new=args.max_new, temperature=args.temperature))


if __name__ == "__main__":
    main()
