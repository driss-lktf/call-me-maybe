"""Shared fixtures for the test suite.

The LLM model is never loaded during tests.  Instead, a ``MockModel`` class
mimics the public Small_LLM_Model API with deterministic, configurable
behaviour.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.schemas import FunctionDefinition, ParameterSchema


# ---------------------------------------------------------------------------
# Mock LLM model
# ---------------------------------------------------------------------------

class MockModel:
    """Minimal stand-in for Small_LLM_Model with a tiny deterministic vocab.

    The mock vocabulary is:
        0  -> ""   (padding / unknown)
        1  -> "a"
        2  -> "b"
        3  -> "c"
        4  -> " "
        5  -> "0"
        6  -> "1"
        7  -> "2"
        8  -> "."
        9  -> "-"
        10 -> '"'  (double quote)
        11 -> ","
        12 -> "}"
        13 -> "fn_add"
        14 -> "fn_greet"
        15 -> "true"
        16 -> "false"
    """

    VOCAB: List[str] = [
        "",       # 0
        "a",      # 1
        "b",      # 2
        "c",      # 3
        " ",      # 4
        "0",      # 5
        "1",      # 6
        "2",      # 7
        ".",      # 8
        "-",      # 9
        '"',      # 10
        ",",      # 11
        "}",      # 12
        "fn_add", # 13
        "fn_greet",  # 14
        "true",   # 15
        "false",  # 16
    ]

    def __init__(self) -> None:
        # logit_override: token_id -> score (used to steer generation)
        self.logit_override: Dict[int, float] = {}

    def encode(self, text: str) -> Any:
        """Return a fake 2-D tensor-like object with token IDs."""
        ids = [ord(c) % len(self.VOCAB) for c in text[:4]]
        # The SDK returns a 2-D tensor; the decoder calls result[0].tolist().
        inner = MagicMock()
        inner.tolist.return_value = ids
        outer = MagicMock()
        outer.__getitem__ = MagicMock(return_value=inner)
        return outer

    def decode(self, ids: List[int]) -> str:
        return "".join(self.VOCAB[i] if i < len(self.VOCAB) else "" for i in ids)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        logits = [-10.0] * len(self.VOCAB)
        for tid, score in self.logit_override.items():
            if tid < len(logits):
                logits[tid] = score
        return logits

    def get_path_to_tokenizer_file(self) -> str:
        raise FileNotFoundError("mock model has no tokenizer file")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model() -> MockModel:
    """Return a fresh MockModel with no logit overrides."""
    return MockModel()


@pytest.fixture
def simple_functions() -> List[FunctionDefinition]:
    """Two simple function definitions for basic tests."""
    return [
        FunctionDefinition(
            name="fn_add",
            description="Add two numbers.",
            parameters={
                "a": ParameterSchema(type="number"),
                "b": ParameterSchema(type="number"),
            },
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet someone.",
            parameters={"name": ParameterSchema(type="string")},
        ),
    ]
