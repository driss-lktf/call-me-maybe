"""Optional live visualisation of the constrained token-generation process.

Enable with ``--verbose`` on the CLI.  All output goes to *stderr* so that
stdout (which carries the JSON results) remains unaffected.

Each generation step is printed as a compact table:

    step   3 | select          | built="fn_add"       top-5 tokens:
               score     token (masked=1204/1205)
               2.418     _numbers  <--
              -0.122     _subtract
              -1.004     _multiply
              ...
"""
import sys
from typing import List, Optional, Tuple


class GenerationVisualizer:
    """Print a live trace of constrained decoding to *stderr*.

    Args:
        enabled: When *False* all methods are no-ops.
        top_k: How many candidate tokens to display per step.
    """

    def __init__(self, enabled: bool = False, top_k: int = 5) -> None:
        self._enabled = enabled
        self._top_k = top_k
        self._step = 0
        self._prompt_label = ""

    @property
    def enabled(self) -> bool:
        """Whether visualisation is active."""
        return self._enabled

    def new_prompt(self, label: str) -> None:
        """Signal that a new prompt is being processed.

        Args:
            label: Short description shown at the top of each section.
        """
        if not self._enabled:
            return
        self._step = 0
        self._prompt_label = label
        print(
            f"\n{'─' * 70}\n[viz] {label}\n{'─' * 70}",
            file=sys.stderr,
            flush=True,
        )

    def step(
        self,
        stage: str,
        built: str,
        top_candidates: List[Tuple[str, float]],
        chosen: Optional[str] = None,
        n_valid: int = 0,
        n_total: int = 0,
    ) -> None:
        """Display one generation step.

        Args:
            stage: Label for the current sub-task
                (e.g. ``"select"`` or ``"number"``).
            built: The string assembled so far in this sub-task.
            top_candidates: The top-k ``(token_str, logit_score)`` pairs
                *after* masking.
            chosen: The token string that was actually selected; highlighted
                with ``<--`` in the output.
            n_valid: Number of tokens kept after masking.
            n_total: Total vocabulary size before masking.
        """
        if not self._enabled:
            return
        self._step += 1
        mask_info = f"  valid={n_valid}/{n_total}" if n_total else ""
        print(
            f"  step {self._step:3d} | {stage:<14s} | built={built!r:<26s}{mask_info}",
            file=sys.stderr,
            flush=True,
        )
        for tok, score in top_candidates[: self._top_k]:
            marker = "  <--" if tok == chosen else ""
            print(
                f"    {score:8.3f}  {tok!r}{marker}",
                file=sys.stderr,
                flush=True,
            )

    def result(self, value: str) -> None:
        """Display the final result of a generation sub-task.

        Args:
            value: The string that was produced.
        """
        if not self._enabled:
            return
        print(
            f"  => {value!r}",
            file=sys.stderr,
            flush=True,
        )
