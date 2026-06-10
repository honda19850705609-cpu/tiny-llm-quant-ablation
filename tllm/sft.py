"""
Supervised fine-tuning (SFT) building blocks — turn the TinyStories *language
model* into something that follows instructions.

This is the natural-language counterpart to tasks.py. Where tasks.py trains
fresh models on symbolic tasks with checkable answers, SFT *continues* training
the pretrained TinyStories checkpoint on (instruction -> response) pairs, using
the existing BPE tokenizer.

The two pieces here are deliberately tokenizer-agnostic and pure-python so they
can be unit-tested without the `tokenizers` library or a real checkpoint:

  - format_prompt / format_full : the Alpaca-style chat template
  - build_example               : encode one record into (tokens, loss_mask),
                                   with the PROMPT masked so loss falls only on
                                   the response (+eos). This response-only
                                   masking is what makes SFT teach "answer the
                                   instruction" rather than "memorize prompts".
"""

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
PROMPT_TEMPLATE_WITH_INPUT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def format_prompt(instruction: str, input_text: str = "") -> str:
    """The part the model is CONDITIONED on (everything up to the response)."""
    if input_text:
        return PROMPT_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_TEMPLATE.format(instruction=instruction)


def format_full(record: dict) -> str:
    """Prompt + response (the full training string, before special tokens)."""
    prompt = format_prompt(record.get("instruction", ""), record.get("input", ""))
    return prompt + record.get("output", "")


def build_example(record, encode, bos_id, eos_id, max_len):
    """Encode one SFT record into fixed-length (tokens, loss_mask).

    Args:
        record: {"instruction", "input"(optional), "output"}
        encode: callable str -> list[int] (e.g. tokenizer.encode(s).ids)
        bos_id, eos_id: special token ids (eos doubles as pad; it is masked)
        max_len: fixed row length (truncate longer, pad shorter)

    Returns (tokens, loss_mask), each a list[int] of length max_len.
    loss_mask[i] == 1 marks a RESPONSE (or eos) token the loss should cover;
    prompt and padding positions are 0.
    """
    prompt_str = format_prompt(record.get("instruction", ""), record.get("input", ""))
    resp_str = record.get("output", "")

    prompt_ids = ([bos_id] if bos_id is not None else []) + list(encode(prompt_str))
    resp_ids = list(encode(resp_str)) + ([eos_id] if eos_id is not None else [])

    tokens = prompt_ids + resp_ids
    mask = [0] * len(prompt_ids) + [1] * len(resp_ids)

    # truncate (keep the start of the prompt and as much response as fits)
    tokens = tokens[:max_len]
    mask = mask[:max_len]

    # pad to fixed length with eos (masked out)
    pad_id = eos_id if eos_id is not None else 0
    if len(tokens) < max_len:
        pad = max_len - len(tokens)
        tokens = tokens + [pad_id] * pad
        mask = mask + [0] * pad

    return tokens, mask
