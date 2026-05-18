"""Integration tests for the FunctionCallingPipeline using a mock model.

We do not load a real LLM; instead the mock model's logits are configured to
steer the decoder toward known answers.
"""
from __future__ import annotations

import json
from typing import List
import pytest

from src.decoder import ConstrainedDecoder
from src.schemas import FunctionCall, FunctionDefinition, ParameterSchema
from tests.conftest import MockModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decoder(mock: MockModel) -> ConstrainedDecoder:
    return ConstrainedDecoder(mock, model_name="mock", use_cache=False)


# ---------------------------------------------------------------------------
# Schemas validation
# ---------------------------------------------------------------------------

class TestSchemas:
    """Pydantic schema models validate correctly."""

    def test_function_definition_round_trip(
        self, simple_functions: List[FunctionDefinition]
    ) -> None:
        for fn in simple_functions:
            raw = fn.model_dump()
            reloaded = FunctionDefinition.model_validate(raw)
            assert reloaded.name == fn.name

    def test_function_call_forbids_extra_keys(self) -> None:
        with pytest.raises(Exception):
            FunctionCall.model_validate(
                {"prompt": "x", "name": "fn", "parameters": {}, "extra": 1}
            )

    def test_function_call_valid(self) -> None:
        fc = FunctionCall(
            prompt="test",
            name="fn_add",
            parameters={"a": 1.0, "b": 2.0},
        )
        assert fc.name == "fn_add"

    def test_nested_parameter_schema(self) -> None:
        schema = ParameterSchema(
            type="object",
            properties={
                "x": ParameterSchema(type="number"),
                "y": ParameterSchema(type="number"),
            },
        )
        assert schema.properties is not None
        assert "x" in schema.properties

    def test_array_parameter_schema(self) -> None:
        schema = ParameterSchema(
            type="array",
            items=ParameterSchema(type="string"),
        )
        assert schema.items is not None
        assert schema.items.type == "string"


# ---------------------------------------------------------------------------
# CLI loading utilities
# ---------------------------------------------------------------------------

class TestCLILoaders:
    """load_functions and load_prompts handle edge cases."""

    def test_load_functions_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_functions

        fdef = [
            {
                "name": "fn_add",
                "description": "Add.",
                "parameters": {"a": {"type": "number"}},
            }
        ]
        p = tmp_path / "fns.json"
        p.write_text(json.dumps(fdef))
        fns = load_functions(p)
        assert len(fns) == 1
        assert fns[0].name == "fn_add"

    def test_load_functions_missing_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_functions

        with pytest.raises(SystemExit):
            load_functions(tmp_path / "nonexistent.json")

    def test_load_functions_invalid_json(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_functions

        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(SystemExit):
            load_functions(p)

    def test_load_functions_empty_array(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_functions

        p = tmp_path / "empty.json"
        p.write_text("[]")
        with pytest.raises(SystemExit, match="no functions"):
            load_functions(p)

    def test_load_prompts_valid(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_prompts

        prompts = [{"prompt": "hello"}, {"prompt": "world"}]
        p = tmp_path / "prompts.json"
        p.write_text(json.dumps(prompts))
        result = load_prompts(p)
        assert result == ["hello", "world"]

    def test_load_prompts_missing_key(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from src.cli import load_prompts

        p = tmp_path / "prompts.json"
        p.write_text('[{"not_prompt": "x"}]')
        with pytest.raises(SystemExit):
            load_prompts(p)


# ---------------------------------------------------------------------------
# Cache module
# ---------------------------------------------------------------------------

class TestCache:
    """Vocabulary cache round-trips correctly."""

    def test_save_and_load(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import src.cache as cache_mod
        from unittest.mock import patch as _patch

        vocab = ["tok0", "tok1", "tok2"]
        with _patch.object(cache_mod, "_CACHE_DIR", tmp_path):
            cache_mod.save_vocab_cache("test-model", vocab)
            loaded = cache_mod.load_vocab_cache("test-model")
        assert loaded == vocab

    def test_load_missing_returns_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import src.cache as cache_mod
        from unittest.mock import patch as _patch

        with _patch.object(cache_mod, "_CACHE_DIR", tmp_path):
            result = cache_mod.load_vocab_cache("no-such-model")
        assert result is None

    def test_clear_removes_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import src.cache as cache_mod
        from unittest.mock import patch as _patch

        vocab = ["a", "b"]
        with _patch.object(cache_mod, "_CACHE_DIR", tmp_path):
            cache_mod.save_vocab_cache("my-model", vocab)
            deleted = cache_mod.clear_vocab_cache("my-model")
            assert deleted
            assert cache_mod.load_vocab_cache("my-model") is None


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class TestVisualizer:
    """GenerationVisualizer is a no-op when disabled."""

    def test_disabled_no_output(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from src.visualizer import GenerationVisualizer

        viz = GenerationVisualizer(enabled=False)
        viz.new_prompt("test")
        viz.step("sel", "x", [("a", 1.0)], chosen="a")
        viz.result("done")
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_enabled_produces_output(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from src.visualizer import GenerationVisualizer

        viz = GenerationVisualizer(enabled=True)
        viz.new_prompt("my prompt")
        viz.step("select", "fn_", [("fn_add", 2.5)], chosen="fn_add", n_valid=1, n_total=10)
        viz.result("fn_add")
        captured = capsys.readouterr()
        assert "fn_add" in captured.err
        assert "select" in captured.err

    def test_top_k_limits_output(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from src.visualizer import GenerationVisualizer

        viz = GenerationVisualizer(enabled=True, top_k=2)
        candidates = [(f"tok{i}", float(i)) for i in range(10)]
        viz.step("test", "", candidates)
        captured = capsys.readouterr()
        # Only top_k=2 lines should appear.
        count = captured.err.count("tok")
        assert count <= 2
