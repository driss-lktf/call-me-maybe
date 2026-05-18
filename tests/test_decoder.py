"""Unit tests for the constrained decoder using the MockModel fixture.

We verify the decoder's core primitives (select_from_set, generate_number,
generate_string_content) without loading a real LLM.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from src.decoder import ConstrainedDecoder
from tests.conftest import MockModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decoder(mock: MockModel) -> ConstrainedDecoder:
    """Create a ConstrainedDecoder backed by *mock* with caching disabled."""
    return ConstrainedDecoder(mock, model_name="mock", use_cache=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# select_from_set
# ---------------------------------------------------------------------------

class TestSelectFromSet:
    """select_from_set() picks the correct candidate token-by-token."""

    def test_selects_single_candidate(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        # Give the highest score to token 13 ("fn_add").
        mock_model.logit_override = {13: 5.0}
        input_ids: List[int] = []
        result = dec.select_from_set(input_ids, ["fn_add", "fn_greet"])
        assert result == "fn_add"

    def test_selects_second_candidate(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        mock_model.logit_override = {14: 5.0}
        input_ids: List[int] = []
        result = dec.select_from_set(input_ids, ["fn_add", "fn_greet"])
        assert result == "fn_greet"

    def test_raises_on_empty_candidates(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        with pytest.raises(ValueError, match="non-empty"):
            dec.select_from_set([], [])

    def test_appends_to_input_ids(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        mock_model.logit_override = {13: 5.0}
        input_ids: List[int] = [1, 2]
        dec.select_from_set(input_ids, ["fn_add"])
        # input_ids must have grown.
        assert len(input_ids) > 2

    def test_single_candidate(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        mock_model.logit_override = {13: 5.0}
        result = dec.select_from_set([], ["fn_add"])
        assert result == "fn_add"


# ---------------------------------------------------------------------------
# generate_number
# ---------------------------------------------------------------------------

class TestGenerateNumber:
    """generate_number() produces valid number strings."""

    def test_generates_digit(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        # Token 5 is "0"; token 12 is "}" (terminator, lower score).
        mock_model.logit_override = {5: 3.0, 12: 1.0}
        result = dec.generate_number([])
        # Must be parseable as float.
        float(result)
        assert "0" in result

    def test_stops_at_terminator(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        # One digit then immediately prefer terminator.
        calls: List[int] = []

        original = mock_model.get_logits_from_input_ids

        def _logits_seq(ids: List[int]) -> List[float]:
            calls.append(len(ids))
            logits = [-10.0] * len(MockModel.VOCAB)
            if len(calls) == 1:
                # First call: prefer "1" (token 6).
                logits[6] = 5.0
                logits[12] = -5.0   # "}" low
            else:
                # Subsequent calls: prefer "}" (token 12) as terminator.
                logits[12] = 5.0    # "}" highest
                logits[6] = -5.0
            return logits

        mock_model.get_logits_from_input_ids = _logits_seq  # type: ignore[method-assign]
        result = dec.generate_number([])
        assert result == "1"


# ---------------------------------------------------------------------------
# generate_string_content
# ---------------------------------------------------------------------------

class TestGenerateStringContent:
    """generate_string_content() terminates at a closing quote."""

    def test_immediate_close(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        # Token 10 is '"' → closes the string immediately.
        mock_model.logit_override = {10: 5.0}
        result = dec.generate_string_content([])
        assert result == ""

    def test_generates_one_char_then_closes(
        self, mock_model: MockModel
    ) -> None:
        dec = _make_decoder(mock_model)
        # Pre-build vocab so the custom _seq only sees generation calls.
        _ = dec.vocab
        calls: List[int] = []

        def _seq(ids: List[int]) -> List[float]:
            calls.append(len(ids))
            logits = [-10.0] * len(MockModel.VOCAB)
            if len(calls) == 1:
                logits[1] = 5.0   # "a" first
            else:
                logits[10] = 5.0  # '"' closes
            return logits

        mock_model.get_logits_from_input_ids = _seq  # type: ignore[method-assign]
        result = dec.generate_string_content([])
        assert result == "a"


# ---------------------------------------------------------------------------
# force_emit / encode
# ---------------------------------------------------------------------------

class TestForceEmit:
    """force_emit() extends input_ids with the tokens for a literal string."""

    def test_force_emit_empty_no_change(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        ids: List[int] = [1, 2, 3]
        dec.force_emit(ids, "")
        assert ids == [1, 2, 3]

    def test_force_emit_adds_tokens(self, mock_model: MockModel) -> None:
        dec = _make_decoder(mock_model)
        ids: List[int] = []
        dec.force_emit(ids, "x")
        # Some tokens should have been added.
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocab:
    """The decoder builds the vocabulary from the mock model."""

    def test_vocab_size_matches_logit_count(
        self, mock_model: MockModel
    ) -> None:
        dec = _make_decoder(mock_model)
        # Access vocab to trigger _build_vocab.
        _ = dec.vocab
        assert dec.vocab_size == len(MockModel.VOCAB)

    def test_vocab_entries_match_decode(
        self, mock_model: MockModel
    ) -> None:
        dec = _make_decoder(mock_model)
        vocab = dec.vocab
        for tid, expected in enumerate(MockModel.VOCAB):
            assert vocab[tid] == expected
