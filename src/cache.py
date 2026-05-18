"""Persistent on-disk cache for the decoded vocabulary list.

Building the vocabulary table requires mapping every token ID to a
human-readable string.  With ~152 k tokens in Qwen3, doing this via the
SDK's decode() method takes 60–90 s.  Using the custom BPETokenizer from
tokenizer.json is much faster, but the first run still reads and processes
the file.  This module persists the result to
``~/.cache/call_me_maybe/vocab_<model>.json`` so that subsequent runs start
in milliseconds.
"""
import json
import sys
from pathlib import Path
from typing import List, Optional


_CACHE_DIR = Path.home() / ".cache" / "call_me_maybe"


def _cache_path(model_name: str) -> Path:
    """Return the path for the cached vocabulary file of *model_name*."""
    safe = model_name.replace("/", "__").replace("\\", "__")
    return _CACHE_DIR / f"vocab_{safe}.json"


def load_vocab_cache(model_name: str) -> Optional[List[str]]:
    """Load the vocabulary list from disk, or return None if not available.

    Args:
        model_name: The Hugging Face model name (used to identify the file).

    Returns:
        A list of decoded token strings indexed by token ID, or None.
    """
    path = _cache_path(model_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            print(
                f"[cache] vocab loaded from {path} ({len(data)} tokens)",
                file=sys.stderr,
                flush=True,
            )
            return data
    except Exception as exc:
        print(f"[cache] failed to load vocab: {exc}", file=sys.stderr, flush=True)
    return None


def save_vocab_cache(model_name: str, vocab: List[str]) -> None:
    """Persist the vocabulary list to disk.

    Args:
        model_name: The Hugging Face model name.
        vocab: The list of decoded token strings.
    """
    path = _cache_path(model_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(vocab, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[cache] vocab cached to {path} ({len(vocab)} tokens)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(f"[cache] failed to save vocab: {exc}", file=sys.stderr, flush=True)


def clear_vocab_cache(model_name: str) -> bool:
    """Delete the cached vocabulary for *model_name*.

    Returns:
        True if the file was deleted, False if it did not exist.
    """
    path = _cache_path(model_name)
    if path.exists():
        path.unlink()
        return True
    return False
