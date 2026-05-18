"""Constrained decoder built on top of the public ``llm_sdk`` API.

Bonus features implemented here
--------------------------------
* **Custom tokenizer** — ``encode()`` and the vocabulary table are built from
  the JSON file returned by ``get_path_to_tokenizer_file()`` (via
  :class:`~src.tokenizer.BPETokenizer`).  This replaces the slow decode loop
  (one SDK call per token ID) with a single JSON read, cutting vocabulary
  initialisation from ~60 s to under a second.
* **Disk cache** — the decoded vocabulary is persisted to
  ``~/.cache/call_me_maybe/`` so that subsequent runs skip even the JSON
  processing step.
* **Visualiser** — when a :class:`~src.visualizer.GenerationVisualizer` is
  attached, each constrained-decoding step prints a table of candidate tokens
  and their scores to *stderr*.

The decoder owns three primitives used by the pipeline:

* :py:meth:`force_emit` — push a fixed string of tokens onto the context
  (used for JSON syntactic markers like ``{"name": "``).
* :py:meth:`select_from_set` — let the model pick one of a finite set of
  strings, token by token, masking every continuation that would not extend
  any candidate.
* :py:meth:`generate_number` and :py:meth:`generate_string_content` —
  generate a JSON value under a small FSM-based constraint.

No private SDK attributes are used; every interaction goes through the public
``encode``, ``decode``, ``get_logits_from_input_ids``, and
``get_path_to_tokenizer_file`` methods.
"""
import sys
from typing import List, Optional

import numpy as np

from llm_sdk import Small_LLM_Model

from . import grammar
from .cache import load_vocab_cache, save_vocab_cache
from .tokenizer import BPETokenizer
from .visualizer import GenerationVisualizer


class ConstrainedDecoder:
    """Generate JSON values token-by-token under structural constraints.

    Args:
        model: A loaded :class:`~llm_sdk.Small_LLM_Model` instance.
        model_name: The Hugging Face model name, used as the cache key.
        use_cache: When *True* (default) the decoded vocabulary is persisted
            to and read from a local JSON file.
        visualizer: An optional :class:`~src.visualizer.GenerationVisualizer`
            that prints a live trace of each generation step.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        model_name: str = "Qwen/Qwen3-0.6B",
        use_cache: bool = True,
        visualizer: Optional[GenerationVisualizer] = None,
    ) -> None:
        self.model = model
        self._model_name = model_name
        self._use_cache = use_cache
        self._viz = visualizer or GenerationVisualizer(enabled=False)
        self._vocab: Optional[List[str]] = None
        self._vocab_size: int = 0
        # Attempt to load the custom BPE tokenizer from the vocabulary JSON.
        self._tokenizer: Optional[BPETokenizer] = self._load_bpe_tokenizer()

    # ------------------------------------------------------------------ init

    def _load_bpe_tokenizer(self) -> Optional[BPETokenizer]:
        """Try to initialise the custom tokenizer from tokenizer.json.

        Returns:
            A :class:`BPETokenizer` instance on success, or *None* if the
            tokenizer file is unavailable or in an unsupported format.
        """
        try:
            path = self.model.get_path_to_tokenizer_file()
            return BPETokenizer(path)
        except Exception as exc:
            print(
                f"[decoder] custom BPE tokenizer unavailable: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    # --------------------------------------------------------------- vocab

    @property
    def vocab(self) -> List[str]:
        """Decoded vocabulary list (lazily initialised)."""
        if self._vocab is None:
            self._build_vocab()
        assert self._vocab is not None
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Number of tokens the model knows about."""
        if self._vocab is None:
            self._build_vocab()
        return self._vocab_size

    def _build_vocab(self) -> None:
        """Build the decoded vocabulary table.

        Strategy (fast → slow):
        1. Disk cache (milliseconds — subsequent runs).
        2. Custom BPETokenizer from tokenizer.json (< 1 s — first run with
           the tokenizer JSON already downloaded).
        3. SDK decode loop (60–90 s — full fallback when the JSON is missing
           or in an unsupported format).
        """
        # Probe the model to get the true logit-space vocabulary size.
        probe_ids = [int(x) for x in self.model.encode("a")[0].tolist()]
        logits = self.model.get_logits_from_input_ids(probe_ids)
        self._vocab_size = len(logits)

        # 1. Disk cache
        if self._use_cache:
            cached = load_vocab_cache(self._model_name)
            if cached is not None:
                n = self._vocab_size
                if len(cached) >= n:
                    self._vocab = cached[:n]
                else:
                    self._vocab = cached + [""] * (n - len(cached))
                return

        print(
            f"[decoder] building vocabulary ({self._vocab_size} tokens)…",
            file=sys.stderr,
            flush=True,
        )

        # 2. Custom BPE tokenizer (fast vocab build from tokenizer.json)
        if self._tokenizer is not None:
            vocab: List[str] = [""] * self._vocab_size
            for tid in range(self._vocab_size):
                vocab[tid] = self._tokenizer.decoded_str(tid)
            self._vocab = vocab
            print(
                "[decoder] vocabulary ready (from tokenizer.json)",
                file=sys.stderr,
                flush=True,
            )
        else:
            # 3. Slow fallback: one SDK decode call per token ID
            vocab_list: List[str] = []
            for tid in range(self._vocab_size):
                try:
                    s = self.model.decode([tid])
                except Exception:
                    s = ""
                vocab_list.append(s)
            self._vocab = vocab_list
            print("[decoder] vocabulary ready", file=sys.stderr, flush=True)

        # Persist for future runs.
        if self._use_cache:
            save_vocab_cache(self._model_name, self._vocab)

    # --------------------------------------------------------- token helpers

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a flat list of token IDs.

        Uses the custom BPETokenizer when available; falls back to the SDK
        tokenizer otherwise.

        Args:
            text: A string that may include special tokens such as
                ``<|im_start|>``.

        Returns:
            A list of integer token IDs.
        """
        if self._tokenizer is not None:
            try:
                return self._tokenizer.encode(text)
            except Exception as exc:
                print(
                    f"[decoder] custom encode failed, using SDK: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        tensor = self.model.encode(text)
        return [int(x) for x in tensor[0].tolist()]

    def force_emit(self, input_ids: List[int], text: str) -> None:
        """Append the token IDs of *text* to *input_ids* unconditionally.

        Args:
            input_ids: The running context, modified in-place.
            text: The literal string to tokenise and append.
        """
        if not text:
            return
        input_ids.extend(self.encode(text))

    def _logits(self, input_ids: List[int]) -> np.ndarray:
        raw = self.model.get_logits_from_input_ids(input_ids)
        return np.asarray(raw, dtype=np.float64)

    # -------------------------------------------------- constrained sampling

    def select_from_set(
        self,
        input_ids: List[int],
        candidates: List[str],
        max_steps: int = 64,
        stage: str = "select",
    ) -> str:
        """Generate tokens until the produced string equals one candidate.

        At every step the set of token IDs that would keep the produced string
        a prefix of at least one candidate is computed; the highest-logit
        token in that set is selected.

        Args:
            input_ids: Running context, extended in-place.
            candidates: The allowed target strings (e.g. function names).
            max_steps: Safety limit on the number of token steps.
            stage: Label used by the visualiser.

        Returns:
            The matched candidate string.

        Raises:
            RuntimeError: If no valid continuation is found or the step
                limit is reached.
            ValueError: If *candidates* is empty.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")

        vocab = self.vocab
        vocab_size = self.vocab_size
        current = ""

        self._viz.new_prompt(
            f"{stage}: choose from {candidates}"
        ) if stage == "select" else None

        for _ in range(max_steps):
            extending = [
                c for c in candidates
                if c.startswith(current) and c != current
            ]
            exact = [c for c in candidates if c == current]
            if exact and not extending:
                self._viz.result(exact[0])
                return exact[0]

            np_logits = self._logits(input_ids)

            # Collect valid (token_id, score) pairs and track top-k for viz.
            valid_tids: List[int] = []
            top_for_viz: List[tuple] = []

            for tid in range(vocab_size):
                tok = vocab[tid]
                if not tok:
                    continue
                new_current = current + tok
                ok = any(cand.startswith(new_current) for cand in candidates)
                if ok:
                    valid_tids.append(tid)
                    top_for_viz.append((tok, float(np_logits[tid])))

            # Pick the best valid token.
            best_tid = -1
            best_score = -np.inf
            for tid in valid_tids:
                if np_logits[tid] > best_score:
                    best_tid = tid
                    best_score = float(np_logits[tid])

            # Visualise this step.
            if self._viz.enabled:
                top_for_viz.sort(key=lambda x: x[1], reverse=True)
                chosen_tok = vocab[best_tid] if best_tid != -1 else ""
                self._viz.step(
                    stage=stage,
                    built=current,
                    top_candidates=top_for_viz[:5],
                    chosen=chosen_tok,
                    n_valid=len(valid_tids),
                    n_total=vocab_size,
                )

            if best_tid == -1:
                if exact:
                    self._viz.result(exact[0])
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
        """Generate a JSON number value, stopping before its terminator.

        The terminator token (``,``, ``}``, whitespace) is *not* consumed.

        Args:
            input_ids: Running context, extended in-place.
            max_chars: Safety limit on generated length.

        Returns:
            The generated number string (e.g. ``"3.14"``).

        Raises:
            RuntimeError: If the FSM gets stuck or the length limit is hit.
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
            top_for_viz: List[tuple] = []

            for tid in range(vocab_size):
                tok = vocab[tid]
                if not tok:
                    continue
                if tok[0] in terminator_chars:
                    if np_logits[tid] > best_term_score:
                        best_term_score = float(np_logits[tid])
                _, ok = grammar.number_consume(state, tok)
                if ok:
                    top_for_viz.append((tok, float(np_logits[tid])))
                    if np_logits[tid] > best_ext_score:
                        best_ext_tid = tid
                        best_ext_score = float(np_logits[tid])

            if self._viz.enabled:
                top_for_viz.sort(key=lambda x: x[1], reverse=True)
                chosen = vocab[best_ext_tid] if best_ext_tid != -1 else ""
                self._viz.step(
                    stage="number",
                    built=output,
                    top_candidates=top_for_viz[:5],
                    chosen=chosen,
                )

            if grammar.number_can_terminate(state):
                if best_term_score >= best_ext_score:
                    self._viz.result(output)
                    return output
            if best_ext_tid == -1:
                if grammar.number_can_terminate(state):
                    self._viz.result(output)
                    return output
                raise RuntimeError(
                    f"generate_number: stuck state={state} output={output!r}"
                )
            tok = vocab[best_ext_tid]
            state, _ = grammar.number_consume(state, tok)
            output += tok
            input_ids.append(best_ext_tid)

        if grammar.number_can_terminate(state):
            self._viz.result(output)
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

        Args:
            input_ids: Running context, extended in-place.
            max_steps: Safety limit on token steps.

        Returns:
            The string content (without surrounding quotes).
        """
        vocab = self.vocab
        vocab_size = self.vocab_size
        state = grammar.string_initial_state()
        output = ""

        for _ in range(max_steps):
            np_logits = self._logits(input_ids)
            best_tid = -1
            best_score = -np.inf
            top_for_viz: List[tuple] = []

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
                if ok:
                    top_for_viz.append((tok, float(np_logits[tid])))
                    if np_logits[tid] > best_score:
                        best_tid = tid
                        best_score = float(np_logits[tid])

            if self._viz.enabled:
                top_for_viz.sort(key=lambda x: x[1], reverse=True)
                chosen = vocab[best_tid] if best_tid != -1 else ""
                self._viz.step(
                    stage="string",
                    built=output,
                    top_candidates=top_for_viz[:5],
                    chosen=chosen,
                )

            if best_tid == -1:
                self.force_emit(input_ids, '"')
                self._viz.result(output)
                return output

            tok = vocab[best_tid]
            for c in tok:
                state, valid = grammar.string_step(state, c)
                if not valid:
                    break
                if grammar.string_done(state):
                    input_ids.append(best_tid)
                    self._viz.result(output)
                    return output
                output += c
            input_ids.append(best_tid)

        self.force_emit(input_ids, '"')
        self._viz.result(output)
        return output
