"""
Synthetic task framework — the missing piece that lets this repo measure
*task completion* (a verifiable right/wrong answer), not just perplexity.

Why this exists
---------------
`eval.py` only knew one metric: perplexity. Perplexity tells you how *surprised*
a model is, never whether it got an answer *right*. To ask "can a compressed
model still DO something" — copy, sort, retrieve a fact from far back — you need
tasks with checkable answers. That is what this module adds.

Design choices (deliberate, so the ablations stay clean)
--------------------------------------------------------
1. SYMBOLIC VOCAB, FRESH MODEL.  Each task family defines its own tiny symbol
   vocabulary and is learned by a model trained from scratch. We do NOT reuse
   the TinyStories checkpoint — it never saw these symbols. This is the standard
   setup in algorithmic / induction-head interpretability work.

2. FIXED-LENGTH SEQUENCES.  Every sample from a task has the same length, so a
   batch is a plain tensor with no padding (and therefore no padding/causal-mask
   foot-guns). "Long range" is created by moving the relevant token to different
   *positions* inside a fixed-length sequence, not by changing the length.

3. LOSS MASKED TO THE ANSWER.  Training targets over the prompt region are set
   to -1, which `TinyLLM.forward` already ignores (`ignore_index=-1`). The model
   is only ever scored/trained on the tokens it is supposed to produce.

A task therefore only has to know how to (a) make one sample and (b) say which
positions are the answer. Everything downstream — batching, training, accuracy,
accuracy-by-distance — is generic.
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# vocab + sample container
# ---------------------------------------------------------------------------
class Vocab:
    """Bidirectional map between human-readable symbols and integer token ids."""

    def __init__(self, symbols):
        self.symbols = list(symbols)
        assert len(set(self.symbols)) == len(self.symbols), "duplicate symbols"
        self.stoi = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.symbols)

    def id(self, sym):
        return self.stoi[sym]

    def ids(self, syms):
        return [self.stoi[s] for s in syms]

    def decode(self, ids):
        return [self.symbols[i] for i in ids]


@dataclass
class Sample:
    """One task instance.

    prompt_ids: the conditioning the model is shown.
    answer_ids: the tokens it must produce (and which we score).
    meta:       per-sample bookkeeping, e.g. {"distance": 37}. Used by
                accuracy-by-distance to bucket results.
    """
    prompt_ids: list
    answer_ids: list
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# base task
# ---------------------------------------------------------------------------
class Task:
    """A task = a vocab + a way to make samples + a way to score them.

    Subclasses set self.vocab and implement sample(). Sequences MUST be fixed
    length (same prompt length and same answer length for every sample) so that
    batching needs no padding.
    """
    name = "task"

    def __init__(self):
        self.vocab = Vocab([])

    @property
    def vocab_size(self):
        return len(self.vocab)

    # --- to be implemented by subclasses ---
    def sample(self, rng) -> Sample:
        raise NotImplementedError

    def spec(self) -> dict:
        """Serializable description so a checkpoint can rebuild this task."""
        raise NotImplementedError

    # --- generic scoring: exact match over the answer tokens ---
    def score(self, pred_ids, answer_ids) -> bool:
        return list(pred_ids) == list(answer_ids)

    @property
    def answer_len(self):
        return len(self.sample(np.random.default_rng(0)).answer_ids)

    @property
    def prompt_len(self):
        return len(self.sample(np.random.default_rng(0)).prompt_ids)

    @property
    def total_len(self):
        """Total tokens per sample (prompt + answer). The data-length knob that
        some tasks expose as `self.seq_len` is a different, task-specific thing."""
        s = self.sample(np.random.default_rng(0))
        return len(s.prompt_ids) + len(s.answer_ids)


# ---------------------------------------------------------------------------
# Phase 1 — algorithmic tasks (cheap signal that the model can DO something)
# ---------------------------------------------------------------------------
class _SeqTransformTask(Task):
    """Shared machinery for copy / reverse / sort: show a sequence, a SEP, then
    the model must emit some deterministic transform of the sequence."""
    SEP = "|"

    def __init__(self, seq_len=16, n_symbols=10):
        self.seq_len = seq_len
        self.n_symbols = n_symbols
        digits = [str(i) for i in range(n_symbols)]
        self.vocab = Vocab(digits + [self.SEP])
        self.sep_id = self.vocab.id(self.SEP)

    def _transform(self, data):
        raise NotImplementedError

    def sample(self, rng):
        # digit values 0..n_symbols-1 ARE their own token ids (digits are laid
        # out first in the vocab), so no remapping is needed.
        data = rng.integers(0, self.n_symbols, self.seq_len).tolist()
        prompt = data + [self.sep_id]
        answer = list(self._transform(data))
        return Sample(prompt_ids=prompt, answer_ids=answer)

    def spec(self):
        return {"task": self.name, "seq_len": self.seq_len, "n_symbols": self.n_symbols}


class CopyTask(_SeqTransformTask):
    """Reproduce the input verbatim. Tests induction / verbatim recall."""
    name = "copy"

    def _transform(self, data):
        return list(data)


class ReverseTask(_SeqTransformTask):
    """Reproduce the input reversed."""
    name = "reverse"

    def _transform(self, data):
        return list(data)[::-1]


class SortTask(_SeqTransformTask):
    """Emit the input sorted ascending."""
    name = "sort"

    def _transform(self, data):
        return sorted(data)


# ---------------------------------------------------------------------------
# Phase 2 — long-range retrieval (the experiment that tests the failed bet)
# ---------------------------------------------------------------------------
class KeyValueTask(Task):
    """Key->value lookup, a.k.a. needle-in-a-haystack.

    Layout (fixed length):
        k v k v ... k v  ?  k_query   ->   v_query

    The model sees `n_pairs` distinct (key, value) pairs, then a QUERY marker and
    one of the keys; it must emit that key's value. Because the sequence length
    is fixed, we create long-range dependency purely by choosing WHICH pair is
    queried: a pair near the start sits far behind the query (large distance),
    one near the end is close (small distance).

    `distance` (stored in meta) = number of pairs between the queried pair and
    the query marker. This is the axis the KV-cache ablation sweeps over.
    """
    name = "kv"
    QUERY = "?"

    def __init__(self, n_pairs=32, n_keys=64, n_vals=16):
        assert n_keys >= n_pairs, "need enough distinct keys for one per pair"
        self.n_pairs = n_pairs
        self.n_keys = n_keys
        self.n_vals = n_vals
        keys = [f"k{i}" for i in range(n_keys)]
        vals = [f"v{i}" for i in range(n_vals)]
        self.vocab = Vocab(keys + vals + [self.QUERY])
        self.key_base = 0
        self.val_base = n_keys
        self.query_id = self.vocab.id(self.QUERY)

    def sample(self, rng, query_pos=None):
        # distinct keys, random values
        key_idx = rng.choice(self.n_keys, size=self.n_pairs, replace=False)
        val_idx = rng.integers(0, self.n_vals, size=self.n_pairs)
        if query_pos is None:
            query_pos = int(rng.integers(0, self.n_pairs))
        prompt = []
        for k, v in zip(key_idx, val_idx):
            prompt.append(self.key_base + int(k))
            prompt.append(self.val_base + int(v))
        prompt.append(self.query_id)
        prompt.append(self.key_base + int(key_idx[query_pos]))
        answer = [self.val_base + int(val_idx[query_pos])]
        distance = self.n_pairs - 1 - query_pos  # 0 = queried pair is last
        return Sample(prompt_ids=prompt, answer_ids=answer, meta={"distance": distance})

    def spec(self):
        return {"task": self.name, "n_pairs": self.n_pairs,
                "n_keys": self.n_keys, "n_vals": self.n_vals}


class InductionTask(Task):
    """Classic induction-head probe with a controllable gap.

    A trigger token T appears with a follower F early in the sequence:
        ... T F ... (filler) ... T   ->   F
    The sequence ends on a second T; the model must predict F. `gap` (= distance
    between the two T's) is the long-range axis. Fixed length; we move the first
    T/F pair to vary the gap.
    """
    name = "induction"

    def __init__(self, seq_len=64, n_symbols=32):
        assert n_symbols >= 4
        self.seq_len = seq_len
        self.n_symbols = n_symbols
        self.vocab = Vocab([str(i) for i in range(n_symbols)])

    def sample(self, rng, gap=None):
        L = self.seq_len
        # second trigger sits at the final prompt position
        if gap is None:
            gap = int(rng.integers(2, L - 1))
        gap = max(2, min(gap, L - 1))
        i0 = (L - 1) - gap                 # position of first trigger
        trigger = int(rng.integers(0, self.n_symbols))
        follower = int(rng.integers(0, self.n_symbols))
        # fillers: anything; we re-derive the answer from i0+1 so collisions are
        # harmless (the model must use the *nearest preceding* T->next rule, and
        # the planted pair is the salient one). Keep it simple: random fillers.
        seq = rng.integers(0, self.n_symbols, L).tolist()
        seq[i0] = trigger
        seq[i0 + 1] = follower
        seq[L - 1] = trigger
        prompt = seq
        answer = [follower]
        return Sample(prompt_ids=prompt, answer_ids=answer, meta={"distance": gap})

    def spec(self):
        return {"task": self.name, "seq_len": self.seq_len, "n_symbols": self.n_symbols}


# ---------------------------------------------------------------------------
# Phase 3 bridge — synthetic instruction following (one model, many tasks)
# ---------------------------------------------------------------------------
class MultiTask(Task):
    """One model, several digit transforms, selected by a leading instruction
    token. This is "instruction following" in the synthetic regime: the prompt
    is [INSTR] data [SEP], and the model must apply the requested transform.

    A clean warm-up for real SFT: same idea (a directive selects behavior),
    but with a verifiable answer and no tokenizer needed.
    """
    name = "multitask"
    SEP = "|"
    _OPS = {
        "copy": lambda d: list(d),
        "reverse": lambda d: list(d)[::-1],
        "sort": lambda d: sorted(d),
    }

    def __init__(self, seq_len=16, n_symbols=10, ops=("copy", "reverse", "sort")):
        self.seq_len = seq_len
        self.n_symbols = n_symbols
        self.ops = list(ops)
        digits = [str(i) for i in range(n_symbols)]
        instr = [f"<{o}>" for o in self.ops]
        self.vocab = Vocab(digits + [self.SEP] + instr)
        self.sep_id = self.vocab.id(self.SEP)
        self.instr_id = {o: self.vocab.id(f"<{o}>") for o in self.ops}

    def sample(self, rng, op=None):
        if op is None:
            op = self.ops[int(rng.integers(0, len(self.ops)))]
        data = rng.integers(0, self.n_symbols, self.seq_len).tolist()
        prompt = [self.instr_id[op]] + data + [self.sep_id]
        answer = list(self._OPS[op](data))
        return Sample(prompt_ids=prompt, answer_ids=answer, meta={"op": op})

    def spec(self):
        return {"task": self.name, "seq_len": self.seq_len,
                "n_symbols": self.n_symbols, "ops": self.ops}


# ---------------------------------------------------------------------------
# registry — rebuild a task from a checkpoint's stored spec
# ---------------------------------------------------------------------------
_REGISTRY = {
    "copy": CopyTask,
    "reverse": ReverseTask,
    "sort": SortTask,
    "kv": KeyValueTask,
    "induction": InductionTask,
    "multitask": MultiTask,
}


def build_task(spec: dict) -> Task:
    """Reconstruct a Task from spec() output (e.g. stored in a checkpoint)."""
    spec = dict(spec)
    name = spec.pop("task")
    return _REGISTRY[name](**spec)


# ---------------------------------------------------------------------------
# batching — generic over any Task, no padding (fixed-length samples)
# ---------------------------------------------------------------------------
def training_batch(task: Task, batch_size: int, rng):
    """Return (x, y) int64 arrays for next-token training.

    y is the input shifted by one, with the PROMPT region set to -1 so the loss
    (ignore_index=-1) only covers answer tokens.
    """
    xs, ys = [], []
    for _ in range(batch_size):
        s = task.sample(rng)
        full = list(s.prompt_ids) + list(s.answer_ids)
        x = full[:-1]
        y = full[1:]
        keep_from = len(s.prompt_ids) - 1     # first index whose target is an answer token
        y = [t if i >= keep_from else -1 for i, t in enumerate(y)]
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
