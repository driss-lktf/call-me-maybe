"""Constrained decoder built on top of the public ``llm_sdk`` API.

The decoder owns three primitives used by the pipeline:

* :py:meth:`force_emit` — push a fixed string of tokens onto the
  context (used for JSON syntactic markers like ``{"name": "``);
* :py:meth:`select_from_set` — let the model pick one of a finite set
  of strings, token by token, masking every continuation that would
  not extend any candidate;
* :py:meth:`generate_number` and :py:meth:`generate_string_content` —
  generate a JSON value under a small FSM-based constraint.

We never use private SDK attributes; every interaction goes through
``encode``, ``decode`` or ``get_logits_from_input_ids``.
"""
import sys
from typing import List, Optional

import numpy as np

from llm_sdk import Small_LLM_Model

from . import grammar


class ConstrainedDecoder:
    """Generate JSON values token-by-token under structural constraints."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self._vocab: Optional[List[str]] = None
        self._vocab_size: int = 0

    # --- vocabulary -----------------------------------------------------

    @property
    def vocab(self) -> List[str]:
        """Return the cached list of decoded token strings."""
        if self._vocab is None:
            self._build_vocab()
        assert self._vocab is not None
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Return the number of token IDs the model knows about."""
        if self._vocab is None:
            self._build_vocab()
        return self._vocab_size

    def _build_vocab(self) -> None:
        probe_ids = self.encode("a")
        if not probe_ids:
            raise RuntimeError("Tokenizer encoded an empty token sequence")
        logits = self.model.get_logits_from_input_ids(probe_ids)
        self._vocab_size = len(logits)
        print(
            f"[decoder] decoding vocabulary ({self._vocab_size} tokens)...",
            file=sys.stderr,
            flush=True,
        )
        vocab: List[str] = []
        for tid in range(self._vocab_size):
            try:
                s = self.model.decode([tid])
            except Exception:
                s = ""
            vocab.append(s)
        self._vocab = vocab
        print("[decoder] vocabulary ready", file=sys.stderr, flush=True)

    # --- token plumbing -------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode ``text`` to a flat list of token IDs."""
        tensor = self.model.encode(text)
        return [int(x) for x in tensor[0].tolist()]

    def force_emit(self, input_ids: List[int], text: str) -> None:
        """Append the token IDs of ``text`` to ``input_ids`` unconditionally."""
        if not text:
            return
        input_ids.extend(self.encode(text))

    def _logits(self, input_ids: List[int]) -> np.ndarray:
        raw = self.model.get_logits_from_input_ids(input_ids)
        return np.asarray(raw, dtype=np.float64)

    # --- constrained primitives -----------------------------------------

    def select_from_set(
        self,
        input_ids: List[int],
        candidates: List[str],
        max_steps: int = 64,
    ) -> str:
        """Generate tokens until the produced string equals one candidate.

        At every step we compute the set of token IDs that, when
        appended, keep the produced string a prefix of at least one
        candidate. We pick the highest-logit token in that set.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        vocab = self.vocab
        vocab_size = self.vocab_size
        current = ""
        for _ in range(max_steps):
            extending = [
                c for c in candidates
                if c.startswith(current) and c != current
            ]
            exact = [c for c in candidates if c == current]
            if exact and not extending:
                return exact[0]
            np_logits = self._logits(input_ids)
            best_tid = -1
            best_score = -np.inf
            for tid in range(vocab_size):
                tok = vocab[tid]
                if not tok:
                    continue
                new_current = current + tok
                ok = False
                for cand in candidates:
                    if cand.startswith(new_current):
                        ok = True
                        break
                if ok and np_logits[tid] > best_score:
                    best_tid = tid
                    best_score = float(np_logits[tid])
            if best_tid == -1:
                if exact:
                    return exact[0]
                raise RuntimeError(
                    f"select_from_set: no valid continuation at {current!r}"
                )
            current += vocab[best_tid]
            input_ids.append(best_tid)
        raise RuntimeError(
            f"select_from_set: exceeded max_steps at {current!r}"
        )

    def generate_number(
        self,
        input_ids: List[int],
        max_chars: int = 32,
    ) -> str:
        """Generate a JSON number value, stopping just before its terminator.

        The chosen tokens are appended to ``input_ids``. The terminator
        token (``,``, ``}``, whitespace) is *not* consumed: the caller
        is in charge of emitting the appropriate JSON syntax.
        """
        vocab = self.vocab
        vocab_size = self.vocab_size
        terminator_chars = {",", "}", "]", " ", "\n", "\t"}
        state = grammar.number_initial_state()
        output = ""
        while len(output) < max_chars:
            np_logits = self._logits(input_ids)
            best_ext_tid = -1
            best_ext_score = -np.inf
            best_term_score = -np.inf
            for tid in range(vocab_size):
                tok = vocab[tid]
                if not tok:
                    continue
                if tok[0] in terminator_chars:
                    if np_logits[tid] > best_term_score:
                        best_term_score = float(np_logits[tid])
                _, ok = grammar.number_consume(state, tok)
                if ok and np_logits[tid] > best_ext_score:
                    best_ext_tid = tid
                    best_ext_score = float(np_logits[tid])
            if grammar.number_can_terminate(state):
                if best_term_score >= best_ext_score:
                    return output
            if best_ext_tid == -1:
                if grammar.number_can_terminate(state):
                    return output
                raise RuntimeError(
                    f"generate_number: stuck in state={state} output={output!r}"
                )
            tok = vocab[best_ext_tid]
            state, _ = grammar.number_consume(state, tok)
            output += tok
            input_ids.append(best_ext_tid)
        if grammar.number_can_terminate(state):
            return output
        raise RuntimeError(
            f"generate_number: exceeded max_chars output={output!r}"
        )

    def generate_string_content(
        self,
        input_ids: List[int],
        max_steps: int = 200,
    ) -> str:
        """Generate JSON-string content + closing quote, returning the content.

        The opening ``"`` must already have been emitted by the caller.
        On return, ``input_ids`` always contains the closing ``"``: either
        the model produced it on its own, or we force-emitted it as a
        safety net when the step limit was reached.
        """
        vocab = self.vocab
        vocab_size = self.vocab_size
        state = grammar.string_initial_state()
        output = ""
        for _ in range(max_steps):
            np_logits = self._logits(input_ids)
            best_tid = -1
            best_score = -np.inf
            for tid in range(vocab_size):
                tok = vocab[tid]
                if not tok:
                    continue
                test_state = state
                ok = True
                for c in tok:
                    test_state, valid = grammar.string_step(test_state, c)
                    if not valid:
                        ok = False
                        break
                    if grammar.string_done(test_state):
                        break
                if ok and np_logits[tid] > best_score:
                    best_tid = tid
                    best_score = float(np_logits[tid])
            if best_tid == -1:
                self.force_emit(input_ids, '"')
                return output
            tok = vocab[best_tid]
            for c in tok:
                state, valid = grammar.string_step(state, c)
                if not valid:
                    break
                if grammar.string_done(state):
                    input_ids.append(best_tid)
                    return output
                output += c
            input_ids.append(best_tid)
        self.force_emit(input_ids, '"')
        return output
