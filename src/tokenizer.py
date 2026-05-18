"""Custom BPE tokenizer loaded from tokenizer.json.

This module provides a public implementation of encode() and decode() built
directly from the vocabulary JSON returned by the SDK's
get_path_to_tokenizer_file(), without calling the SDK's own encode/decode.

How it integrates with constrained decoding
-------------------------------------------
The constrained decoder needs two operations:

1. ``encode(text)``  — turn a prompt string into a list of token IDs that can
   be fed to ``get_logits_from_input_ids``.
2. ``decoded_str(token_id)`` — given a token ID, return the human-readable
   string that token represents.  This is used to build the vocabulary table
   that the constraint engine walks when it decides which IDs are valid.

Both operations are implemented here using only the tokenizer.json file; the
slow SDK decode loop (one call per token ID) is therefore unnecessary.
"""
import json
import re
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Byte ↔ unicode mapping (identical to GPT-2 / tiktoken convention)
# ---------------------------------------------------------------------------

def _make_byte_encoder() -> Dict[int, str]:
    """Build the GPT-2-style byte-to-unicode character table.

    Bytes that correspond to printable ASCII or Latin-1 supplement characters
    are mapped to themselves; the remaining 66 bytes are mapped to Unicode
    code points above U+0100.  This lets every byte be represented as a
    printable character and round-tripped losslessly.
    """
    printable: List[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(printable)
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


_BYTE_ENC: Dict[int, str] = _make_byte_encoder()
_BYTE_DEC: Dict[str, int] = {v: k for k, v in _BYTE_ENC.items()}

# ---------------------------------------------------------------------------
# Pre-tokenisation pattern
# ---------------------------------------------------------------------------

# Simplified ASCII-safe version of the GPT-4 / Qwen3 pre-tokenisation regex.
# The full pattern uses \p{L}/\p{N} (Unicode categories) which requires the
# third-party ``regex`` library.  For ASCII-heavy prompts (function names,
# JSON markers, English text) the two patterns produce identical results.
_PRETOK = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"    # common English contractions
    r"|[^\r\na-zA-Z0-9]?[a-zA-Z]+"  # optional non-alpha prefix + letters
    r"|[0-9]{1,3}"               # up to three digits (matches tiktoken)
    r"| ?[^\sa-zA-Z0-9]+[\r\n]*"  # punctuation / symbols
    r"|\s*[\r\n]"                # newlines
    r"|\s+(?!\S)"                # trailing whitespace
    r"|\s+",                     # general whitespace
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json file.

    Provides encode() and decode() without using the SDK's own tokenizer,
    relying only on the JSON vocabulary and merge-rules files.
    """

    def __init__(self, tokenizer_path: str) -> None:
        """Load the tokenizer from *tokenizer_path* (a tokenizer.json file).

        Args:
            tokenizer_path: Absolute path to a HuggingFace tokenizer.json.
        """
        with open(tokenizer_path, encoding="utf-8") as f:
            data = json.load(f)

        model_block = data.get("model", {})

        # str -> id (in the raw BPE representation, e.g. "Ġhello" -> 12345)
        self._str_to_id: Dict[str, int] = dict(model_block.get("vocab", {}))
        # id -> raw BPE string (built as the inverse of the above)
        self._id_to_raw: Dict[int, str] = {
            v: k for k, v in self._str_to_id.items()
        }

        # BPE merge priority table: (left_piece, right_piece) -> rank
        self._merges: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(model_block.get("merges", [])):
            if isinstance(merge, str):
                parts = merge.split(" ", 1)
            else:
                parts = list(merge)
            if len(parts) == 2:
                self._merges[(parts[0], parts[1])] = rank

        # Special / added tokens (e.g. <|im_start|>, <|endoftext|>)
        self._added: Dict[str, int] = {}
        for entry in data.get("added_tokens", []):
            content: str = entry.get("content", "")
            tok_id: int = entry.get("id", -1)
            if content and tok_id >= 0:
                self._added[content] = tok_id
                self._str_to_id[content] = tok_id
                self._id_to_raw[tok_id] = content

        # BPE result cache: byte-encoded word -> list[token_id]
        self._cache: Dict[str, List[int]] = {}

        print(
            f"[tokenizer] {len(self._str_to_id)} tokens, "
            f"{len(self._merges)} merges, "
            f"{len(self._added)} special tokens",
            file=sys.stderr,
            flush=True,
        )

    # ------------------------------------------------------------------ vocab

    @property
    def vocab_size(self) -> int:
        """Number of token types (max token ID + 1)."""
        if not self._id_to_raw:
            return 0
        return max(self._id_to_raw) + 1

    def decoded_str(self, token_id: int) -> str:
        """Return the human-readable string for a single token ID.

        Special / added tokens are returned verbatim (e.g. ``<|im_start|>``).
        Regular BPE tokens are decoded via the byte-level mapping so that
        leading-space tokens (``Ġhello`` → ``" hello"``) are correctly
        represented.

        Args:
            token_id: The integer token ID.

        Returns:
            The decoded string, or an empty string if the ID is unknown.
        """
        raw = self._id_to_raw.get(token_id, "")
        if not raw:
            return ""
        # Special tokens are stored verbatim — return as-is.
        if raw in self._added:
            return raw
        # Byte-decode the GPT-2 Unicode characters back to UTF-8.
        try:
            byte_vals = bytearray(
                _BYTE_DEC[ch] for ch in raw if ch in _BYTE_DEC
            )
            return byte_vals.decode("utf-8", errors="replace")
        except Exception:
            return raw

    # ------------------------------------------------------------------ BPE

    def _bpe(self, word: str) -> List[int]:
        """Apply BPE merges to a single byte-encoded word.

        Args:
            word: A word whose characters are already in the GPT-2
                byte-level representation (e.g. ``"ĠHello"``).

        Returns:
            A list of token IDs after applying all applicable merges.
        """
        if word in self._cache:
            return self._cache[word]

        # Start with one piece per character.
        chars = list(word)

        # Iteratively merge the highest-priority adjacent pair.
        while len(chars) > 1:
            best_rank: float = float("inf")
            best_i = -1
            for i in range(len(chars) - 1):
                r = self._merges.get((chars[i], chars[i + 1]), float("inf"))
                if r < best_rank:
                    best_rank = r
                    best_i = i
            if best_i == -1 or best_rank == float("inf"):
                break
            merged = chars[best_i] + chars[best_i + 1]
            chars = chars[:best_i] + [merged] + chars[best_i + 2:]

        ids = [
            self._str_to_id[piece]
            for piece in chars
            if piece in self._str_to_id
        ]
        self._cache[word] = ids
        return ids

    # --------------------------------------------------------------- encode

    def _split_on_special(self, text: str) -> List[Tuple[str, bool]]:
        """Split *text* at special-token boundaries.

        Returns a list of ``(segment, is_special)`` pairs so that special
        tokens (e.g. ``<|im_start|>``) can be mapped to their IDs directly
        without going through the BPE algorithm.
        """
        if not self._added:
            return [(text, False)]
        # Sort longest first to avoid partial matches.
        pattern = "(" + "|".join(
            re.escape(s)
            for s in sorted(self._added, key=len, reverse=True)
        ) + ")"
        result: List[Tuple[str, bool]] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            if part in self._added:
                result.append((part, True))
            else:
                result.append((part, False))
        return result

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of token IDs using byte-level BPE.

        Special tokens (registered in ``added_tokens``) are matched first and
        mapped directly.  The remaining text is pre-tokenised with the
        simplified regex and then encoded via BPE.

        Args:
            text: The input string (may include special tokens like
                ``<|im_start|>system\\n``).

        Returns:
            A list of integer token IDs suitable for
            ``get_logits_from_input_ids``.
        """
        ids: List[int] = []
        for segment, is_special in self._split_on_special(text):
            if is_special:
                ids.append(self._added[segment])
            else:
                for token in re.findall(_PRETOK, segment):
                    # Convert each UTF-8 byte to its GPT-2 Unicode character.
                    byte_str = "".join(
                        _BYTE_ENC[b] for b in token.encode("utf-8")
                    )
                    ids.extend(self._bpe(byte_str))
        return ids

    # --------------------------------------------------------------- decode

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a human-readable string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded UTF-8 string.
        """
        raw_parts: List[str] = [self._id_to_raw.get(i, "") for i in ids]
        combined = "".join(raw_parts)
        try:
            byte_vals = bytearray(
                _BYTE_DEC[ch] for ch in combined if ch in _BYTE_DEC
            )
            return byte_vals.decode("utf-8", errors="replace")
        except Exception:
            return combined

    # --------------------------------------------------------------- info

    def token_info(self, token_id: int) -> Optional[str]:
        """Return a debug string for a token (raw BPE repr + decoded form).

        Args:
            token_id: The integer token ID.

        Returns:
            A short string like ``id=42 raw='Ġthe' decoded=' the'``,
            or ``None`` if the ID is unknown.
        """
        raw = self._id_to_raw.get(token_id)
        if raw is None:
            return None
        decoded = self.decoded_str(token_id)
        return f"id={token_id} raw={raw!r} decoded={decoded!r}"
