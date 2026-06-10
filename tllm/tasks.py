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
    answer_mask: optional 0/1 list over answer_ids. 1 = train on / count this
                answer token, 0 = present in the sequence but NOT a training
                target (e.g. tokens a tool executor injects, which the model
                consumes but should not be taught to predict). None => all-1.
    """
    prompt_ids: list
    answer_ids: list
    meta: dict = field(default_factory=dict)
    answer_mask: list = None


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
# Tier 2 — complex / compositional capabilities
#   addition   : multi-step arithmetic with carry (scratchpad-order output)
#   incontext  : few-shot in-context rule learning (infer mapping from shots)
#   multihop   : compositional retrieval (follow a key->value chain)
#   statetrack : stateful variable tracking with a recency axis
#   tooluse    : function-calling / agentic (emit call -> execute -> consume)
# ---------------------------------------------------------------------------
class AdditionTask(Task):
    """Multi-step integer addition with carry propagation.

    Layout (fixed length):
        a_{d-1} ... a_0  +  b_{d-1} ... b_0  =   ->   s_0 s_1 ... s_d

    The two operands are shown MOST-significant-digit-first (the way humans
    write numbers), each padded to exactly ``n_digits`` digits, separated by a
    "+" token and terminated by an "=" token. The answer is the sum written
    LEAST-significant-digit-FIRST in exactly ``n_digits + 1`` digits.

    Why LSB-first for the answer? Addition's carry flows from the ones place
    upward. Emitting the sum low-digit-first turns carry propagation into a
    plain left-to-right generation: to produce digit ``i`` the model only needs
    operand digits ``i`` and the carry it just produced for digit ``i`` — no
    look-ahead. This is the small-model-friendly "scratchpad" ordering; it is
    the addition analogue of the reversed-output trick used in algorithmic
    transformer work. (The prompt operands stay in the natural MSB-first human
    order; only the produced answer is reversed.)

    Fixed length, for every rng seed and every n_digits:
        prompt = 2 * n_digits + 2   (two n-digit operands + "+" + "=")
        answer = n_digits + 1       (sum of two < 10**n_digits values needs at
                                     most n_digits + 1 digits)

    meta = {"n_digits": n_digits, "n_carries": <number of carry-outs>} where
    n_carries counts, over the per-digit additions ones->up, how many produced a
    carry into the next place (a proxy for arithmetic difficulty).
    """
    name = "addition"
    PLUS = "+"
    EQ = "="

    def __init__(self, n_digits=3):
        assert n_digits >= 1, "need at least one digit"
        self.n_digits = n_digits
        digits = [str(i) for i in range(10)]               # ids 0..9
        self.vocab = Vocab(digits + [self.PLUS, self.EQ])  # "+"=10, "="=11
        self.plus_id = self.vocab.id(self.PLUS)
        self.eq_id = self.vocab.id(self.EQ)

    def _digits_msb(self, value, width):
        """value -> list of `width` digit ids, most-significant first."""
        out = [0] * width
        for i in range(width - 1, -1, -1):   # fill from least-significant up
            out[i] = value % 10
            value //= 10
        return out

    @staticmethod
    def _count_carries(a, b, d):
        """Number of carry-outs when adding a and b digit-by-digit (ones up)."""
        carry = 0
        n = 0
        for _ in range(d):
            col = (a % 10) + (b % 10) + carry
            carry = col // 10
            if carry:
                n += 1
            a //= 10
            b //= 10
        return n

    def sample(self, rng, n_carries=None):
        d = self.n_digits
        hi = 10 ** d
        if n_carries is None:
            a = int(rng.integers(0, hi))
            b = int(rng.integers(0, hi))
        else:
            # control axis: draw until the addition has exactly `n_carries`
            # carry-outs. n_carries in [0, d]. Rejection sampling uses ONLY the
            # passed rng (deterministic) and does not change sequence lengths.
            target = int(n_carries)
            assert 0 <= target <= d, f"n_carries must be in [0, {d}]"
            while True:
                a = int(rng.integers(0, hi))
                b = int(rng.integers(0, hi))
                if self._count_carries(a, b, d) == target:
                    break

        a_digits = self._digits_msb(a, d)            # MSB-first, length d
        b_digits = self._digits_msb(b, d)            # MSB-first, length d
        prompt = a_digits + [self.plus_id] + b_digits + [self.eq_id]

        s = a + b
        # answer: LSB-first, exactly d+1 digits. a+b < 2*10**d <= 10**(d+1),
        # so d+1 digits always suffice and s is exactly 0 after the loop.
        answer = []
        for _ in range(d + 1):
            answer.append(s % 10)
            s //= 10

        carries = self._count_carries(a, b, d)
        return Sample(
            prompt_ids=prompt,
            answer_ids=answer,
            meta={"n_digits": d, "n_carries": carries},
        )

    def spec(self):
        return {"task": self.name, "n_digits": self.n_digits}


class InContextMappingTask(Task):
    """Few-shot in-context rule learning."""
    name = "incontext"

    def __init__(self, n_symbols=20, n_shots=4, mode="shift"):
        assert mode in ("shift", "permutation"), f"bad mode {mode!r}"
        if mode == "shift":
            assert n_symbols >= n_shots + 2, (
                "shift mode needs n_symbols >= n_shots+2 (distinct shot inputs "
                "plus one unseen query)")
        else:  # permutation: query reuses a shot input
            assert n_symbols >= n_shots, "need n_symbols >= n_shots distinct inputs"
        self.n_symbols = n_symbols
        self.n_shots = n_shots
        self.mode = mode
        syms = [str(i) for i in range(n_symbols)]
        self.vocab = Vocab(syms + ["->", ";"])
        self.arrow_id = self.vocab.id("->")   # id == n_symbols
        self.sep_id = self.vocab.id(";")       # id == n_symbols + 1

    def sample(self, rng, mode=None):
        if mode is None:
            mode = self.mode

        if mode == "shift":
            shift = int(rng.integers(1, self.n_symbols))   # [1, n_symbols-1]
            f = lambda x, s=shift: (x + s) % self.n_symbols
            meta_extra = {"shift": shift}
            chosen = rng.choice(self.n_symbols, size=self.n_shots + 1, replace=False)
            shot_inputs = chosen[:self.n_shots]
            x_query = int(chosen[self.n_shots])
        else:  # permutation
            perm = rng.permutation(self.n_symbols)         # perm[x] = f(x)
            f = lambda x, p=perm: int(p[x])
            meta_extra = {}
            shot_inputs = rng.choice(self.n_symbols, size=self.n_shots, replace=False)
            x_query = int(shot_inputs[int(rng.integers(0, self.n_shots))])

        prompt = []
        for x in shot_inputs:
            x = int(x)
            prompt += [self.vocab.id(str(x)), self.arrow_id,
                       self.vocab.id(str(f(x))), self.sep_id]
        prompt += [self.vocab.id(str(x_query)), self.arrow_id]
        answer = [self.vocab.id(str(f(x_query)))]

        meta = {"mode": mode, "n_shots": self.n_shots}
        meta.update(meta_extra)
        return Sample(prompt_ids=prompt, answer_ids=answer, meta=meta)

    def spec(self):
        return {"task": self.name, "n_symbols": self.n_symbols,
                "n_shots": self.n_shots, "mode": self.mode}


class MultiHopTask(Task):
    """Compositional retrieval: follow a key->value chain `hops` times."""
    name = "multihop"
    QUERY = "QUERY"

    def __init__(self, n_pairs=16, hops=3, n_symbols=48):
        assert hops >= 1, "need at least one hop"
        assert n_pairs >= hops, "need at least `hops` pairs for the chain edges"
        assert n_symbols > n_pairs + hops, "n_symbols must be comfortably > n_pairs+hops"
        self.n_pairs = n_pairs
        self.hops = hops
        self.n_symbols = n_symbols
        syms = [str(i) for i in range(n_symbols)]
        self.vocab = Vocab(syms + [self.QUERY])
        self.query_id = self.vocab.id(self.QUERY)

    def sample(self, rng, hops=None):
        H = self.hops if hops is None else int(hops)
        assert 1 <= H <= self.n_pairs, "hops out of range for this n_pairs"

        chain = [int(c) for c in rng.choice(self.n_symbols, size=H + 1, replace=False)]
        chain_sources = set(chain[:H])

        n_distract = self.n_pairs - H
        pool = [s for s in range(self.n_symbols) if s not in chain_sources]
        distract_src = [pool[i] for i in rng.choice(len(pool), size=n_distract, replace=False)]
        distract_tgt = [int(t) for t in rng.integers(0, self.n_symbols, size=n_distract)]

        pairs = [(chain[i], chain[i + 1]) for i in range(H)]
        pairs += [(int(k), int(v)) for k, v in zip(distract_src, distract_tgt)]

        pairs = [pairs[i] for i in rng.permutation(self.n_pairs)]

        prompt = []
        for k, v in pairs:
            prompt.append(k)
            prompt.append(v)
        prompt.append(self.query_id)
        prompt.append(chain[0])

        answer = [chain[H]]
        return Sample(prompt_ids=prompt, answer_ids=answer, meta={"hops": H})

    def spec(self):
        return {"task": self.name, "n_pairs": self.n_pairs,
                "hops": self.hops, "n_symbols": self.n_symbols}


class StateTrackingTask(Task):
    name = "statetrack"
    SET = "SET"
    QUERY = "QUERY"

    def __init__(self, n_vars=8, n_ops=12, n_vals=10):
        assert n_vars >= 1 and n_ops >= 1 and n_vals >= 1
        self.n_vars = n_vars
        self.n_ops = n_ops
        self.n_vals = n_vals
        var_syms = [f"x{i}" for i in range(n_vars)]
        val_syms = [str(i) for i in range(n_vals)]
        self.vocab = Vocab(var_syms + val_syms + [self.SET, self.QUERY])
        self.var_base = 0                 # ids 0 .. n_vars-1
        self.val_base = n_vars            # ids n_vars .. n_vars+n_vals-1
        self.set_id = self.vocab.id(self.SET)      # n_vars + n_vals
        self.query_id = self.vocab.id(self.QUERY)  # n_vars + n_vals + 1

    def sample(self, rng, recency=None):
        n_ops, n_vars, n_vals = self.n_ops, self.n_vars, self.n_vals
        op_vars = rng.integers(0, n_vars, size=n_ops)
        op_vals = rng.integers(0, n_vals, size=n_ops)

        if recency is None:
            qvar = int(op_vars[int(rng.integers(0, n_ops))])
        else:
            r = int(recency)
            r = max(1, min(r, n_ops))
            last_idx = n_ops - r
            qvar = int(rng.integers(0, n_vars))
            op_vars[last_idx] = qvar
            if n_vars > 1:
                for j in range(last_idx + 1, n_ops):
                    if int(op_vars[j]) == qvar:
                        alt = int(rng.integers(0, n_vars - 1))
                        if alt >= qvar:
                            alt += 1
                        op_vars[j] = alt

        hits = np.nonzero(op_vars == qvar)[0]
        last_idx = int(hits[-1])
        answer_val = int(op_vals[last_idx])
        ops_between = n_ops - 1 - last_idx

        prompt = []
        for v, val in zip(op_vars.tolist(), op_vals.tolist()):
            prompt.append(self.set_id)
            prompt.append(self.var_base + int(v))
            prompt.append(self.val_base + int(val))
        prompt.append(self.query_id)
        prompt.append(self.var_base + qvar)
        answer = [self.val_base + answer_val]
        return Sample(prompt_ids=prompt, answer_ids=answer,
                      meta={"recency": ops_between})

    def spec(self):
        return {"task": self.name, "n_vars": self.n_vars,
                "n_ops": self.n_ops, "n_vals": self.n_vals}


class ToolUseTask(Task):
    """Function-calling / agentic addition with a masked tool result."""
    name = "tooluse"

    def __init__(self, n_digits=3):
        self.n_digits = n_digits
        digits = [str(i) for i in range(10)]           # ids 0..9
        symbols = digits + ["+", "CALC", "CALL_END", "RESULT", "ANS", "EOS"]
        self.vocab = Vocab(symbols)
        self.PLUS = self.vocab.id("+")                 # 10
        self.CALC = self.vocab.id("CALC")              # 11
        self.CALL_END = self.vocab.id("CALL_END")      # 12
        self.RESULT = self.vocab.id("RESULT")          # 13
        self.ANS = self.vocab.id("ANS")                # 14
        self.EOS = self.vocab.id("EOS")                # 15

    def _to_digits(self, value, width):
        s = str(int(value)).zfill(width)[-width:]
        return [int(c) for c in s]

    def _from_digits(self, digit_ids):
        return int("".join(str(int(d)) for d in digit_ids))

    def executor(self, call_token_ids):
        nd = self.n_digits
        fallback = [self.RESULT] + [0] * (nd + 1)
        try:
            toks = list(call_token_ids)
            if self.PLUS not in toks:
                return fallback
            p = toks.index(self.PLUS)
            a_digits = toks[:p]
            b_digits = toks[p + 1:]
            if len(a_digits) != nd or len(b_digits) != nd:
                return fallback
            for d in a_digits + b_digits:
                if not (0 <= int(d) <= 9):
                    return fallback
            a = self._from_digits(a_digits)
            b = self._from_digits(b_digits)
            s = a + b
            return [self.RESULT] + self._to_digits(s, nd + 1)
        except Exception:
            return fallback

    def sample(self, rng):
        nd = self.n_digits
        hi = 10 ** nd
        a = int(rng.integers(0, hi))
        b = int(rng.integers(0, hi))
        s = a + b

        a_digits = self._to_digits(a, nd)
        b_digits = self._to_digits(b, nd)
        s_digits = self._to_digits(s, nd + 1)

        prompt = a_digits + [self.PLUS] + b_digits

        call_span = [self.CALC] + a_digits + [self.PLUS] + b_digits + [self.CALL_END]
        result_span = [self.RESULT] + s_digits
        ans_span = [self.ANS] + s_digits + [self.EOS]

        answer = call_span + result_span + ans_span
        mask = [1] * len(call_span) + [0] * len(result_span) + [1] * len(ans_span)

        return Sample(
            prompt_ids=prompt,
            answer_ids=answer,
            meta={"n_digits": nd},
            answer_mask=mask,
        )

    def spec(self):
        return {"task": self.name, "n_digits": self.n_digits}


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
    "addition": AdditionTask,
    "incontext": InContextMappingTask,
    "multihop": MultiHopTask,
    "statetrack": StateTrackingTask,
    "tooluse": ToolUseTask,
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
        prompt, answer = list(s.prompt_ids), list(s.answer_ids)
        amask = s.answer_mask if s.answer_mask is not None else [1] * len(answer)
        full = prompt + answer
        x = full[:-1]
        y = full[1:]
        # y[i] predicts full[i+1]; it's an answer token iff i+1 >= len(prompt).
        # Train only on answer tokens whose answer_mask is 1.
        out_y = []
        for i, t in enumerate(y):
            j = (i + 1) - len(prompt)          # index into answer / amask
            out_y.append(t if (j >= 0 and amask[j]) else -1)
        xs.append(x)
        ys.append(out_y)
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
