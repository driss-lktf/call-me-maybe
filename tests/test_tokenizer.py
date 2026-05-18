"""Unit tests for the custom BPE tokenizer in src/tokenizer.py.

These tests do NOT require a real tokenizer.json file.  They build a minimal
in-memory tokenizer using a tiny hand-crafted vocabulary and merge table, then
verify the BPE algorithm and byte-encoding logic.
"""
import json
import os
import tempfile
from typing import Any, Dict

from src.tokenizer import BPETokenizer, _BYTE_ENC, _BYTE_DEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tokenizer(data: Dict[str, Any]) -> str:
    """Write a tokenizer.json to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _minimal_tokenizer(
    extra_vocab: Dict[str, int] | None = None,
    merges: list | None = None,
    added_tokens: list | None = None,
) -> str:
    """Return path to a minimal tokenizer.json with configurable vocab."""
    # Build a tiny byte-level vocabulary (a-z, A-Z, 0-9, some punctuation).
    vocab: Dict[str, int] = {}
    # Single printable-ASCII tokens (their GPT-2 byte-encoder repr).
    for b in range(256):
        ch = _BYTE_ENC[b]
        vocab[ch] = b

    if extra_vocab:
        vocab.update(extra_vocab)

    return _write_tokenizer({
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges or [],
        },
        "added_tokens": added_tokens or [],
    })


# ---------------------------------------------------------------------------
# Byte encoder round-trip
# ---------------------------------------------------------------------------

class TestByteEncoder:
    """Verify the GPT-2 byte↔unicode mapping."""

    def test_encoder_has_256_entries(self) -> None:
        assert len(_BYTE_ENC) == 256

    def test_decoder_is_inverse(self) -> None:
        for b in range(256):
            ch = _BYTE_ENC[b]
            assert _BYTE_DEC[ch] == b

    def test_printable_ascii_maps_to_itself(self) -> None:
        # Bytes 33-126 (printable ASCII excl. space) map to the same char.
        for b in range(33, 127):
            assert _BYTE_ENC[b] == chr(b)

    def test_space_maps_to_special_char(self) -> None:
        # Space (byte 32) is NOT printable → mapped above 256.
        assert _BYTE_ENC[32] != " "
        assert ord(_BYTE_ENC[32]) > 255


# ---------------------------------------------------------------------------
# BPETokenizer basic construction
# ---------------------------------------------------------------------------

class TestBPETokenizerConstruct:
    """Tokenizer loads vocab and merge rules correctly."""

    def test_loads_vocab(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.vocab_size > 0

    def test_loads_merges(self) -> None:
        # Add a single merge rule.
        a_ch = _BYTE_ENC[ord("a")]
        b_ch = _BYTE_ENC[ord("b")]
        merge_token = a_ch + b_ch
        path = _minimal_tokenizer(
            extra_vocab={merge_token: 300},
            merges=[f"{a_ch} {b_ch}"],
        )
        tok = BPETokenizer(path)
        assert (a_ch, b_ch) in tok._merges

    def test_loads_added_tokens(self) -> None:
        path = _minimal_tokenizer(
            added_tokens=[{"content": "<|im_start|>", "id": 500}],
        )
        tok = BPETokenizer(path)
        assert "<|im_start|>" in tok._added
        assert tok._added["<|im_start|>"] == 500


# ---------------------------------------------------------------------------
# BPETokenizer.decoded_str
# ---------------------------------------------------------------------------

class TestDecodedStr:
    """decoded_str() maps token IDs to human-readable strings."""

    def test_known_byte_token(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        # Token ID for 'a' (byte 97) should decode to 'a'.
        assert tok.decoded_str(ord("a")) == "a"

    def test_unknown_id_returns_empty(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.decoded_str(999999) == ""

    def test_special_token_returned_verbatim(self) -> None:
        path = _minimal_tokenizer(
            added_tokens=[{"content": "<|im_end|>", "id": 501}],
        )
        tok = BPETokenizer(path)
        assert tok.decoded_str(501) == "<|im_end|>"


# ---------------------------------------------------------------------------
# BPETokenizer.encode
# ---------------------------------------------------------------------------

class TestEncode:
    """encode() produces the expected token ID sequences."""

    def test_single_byte_char(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        ids = tok.encode("a")
        # "a" -> byte 97 -> GPT-2 char ord("a") -> token id ord("a") = 97
        assert ids == [97]

    def test_multiple_chars(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        ids = tok.encode("ab")
        assert ids == [97, 98]

    def test_special_token_encoded(self) -> None:
        path = _minimal_tokenizer(
            added_tokens=[{"content": "<|im_start|>", "id": 500}],
        )
        tok = BPETokenizer(path)
        ids = tok.encode("<|im_start|>")
        assert ids == [500]

    def test_special_token_in_context(self) -> None:
        path = _minimal_tokenizer(
            added_tokens=[{"content": "<|im_start|>", "id": 500}],
        )
        tok = BPETokenizer(path)
        ids = tok.encode("<|im_start|>a")
        assert ids[0] == 500
        assert 97 in ids

    def test_bpe_merge_applied(self) -> None:
        a_ch = _BYTE_ENC[ord("a")]
        b_ch = _BYTE_ENC[ord("b")]
        merge_token = a_ch + b_ch
        path = _minimal_tokenizer(
            extra_vocab={merge_token: 300},
            merges=[f"{a_ch} {b_ch}"],
        )
        tok = BPETokenizer(path)
        ids = tok.encode("ab")
        # With the merge rule, "ab" should be a single token (ID 300).
        assert ids == [300]

    def test_empty_string(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.encode("") == []


# ---------------------------------------------------------------------------
# BPETokenizer.decode
# ---------------------------------------------------------------------------

class TestDecode:
    """decode() reconstructs text from token IDs."""

    def test_single_char(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.decode([97]) == "a"

    def test_multiple_chars(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.decode([97, 98, 99]) == "abc"

    def test_empty_list(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.decode([]) == ""

    def test_unknown_id_ignored(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        # Unknown IDs map to empty string → result is the remaining chars.
        result = tok.decode([999999, 97])
        assert "a" in result

    def test_roundtrip_ascii(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        text = "hello"
        ids = tok.encode(text)
        assert tok.decode(ids) == text


# ---------------------------------------------------------------------------
# BPETokenizer.token_info
# ---------------------------------------------------------------------------

class TestTokenInfo:
    """token_info() returns a debug string or None."""

    def test_known_token(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        info = tok.token_info(97)
        assert info is not None
        assert "97" in info

    def test_unknown_token(self) -> None:
        path = _minimal_tokenizer()
        tok = BPETokenizer(path)
        assert tok.token_info(999999) is None
