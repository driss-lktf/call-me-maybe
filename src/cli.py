"""Command-line entry point for ``python -m src``."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pydantic import ValidationError

from .pipeline import FunctionCallingPipeline
from .schemas import FunctionDefinition, TestPrompt


DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description=(
            "Translate natural-language prompts into JSON function calls "
            "using constrained decoding on a small LLM."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name.",
    )
    return parser.parse_args(argv)


def _load_json_array(path: Path, kind: str) -> List[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {kind} file not found: {path}") from exc
    except OSError as exc:
        raise SystemExit(
            f"error: cannot read {kind} file {path}: {exc}"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: invalid JSON in {kind} file {path}: {exc}"
        ) from exc
    if not isinstance(raw, list):
        raise SystemExit(
            f"error: {kind} file {path} must contain a JSON array"
        )
    return raw


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Load and validate the function definitions file."""
    raw = _load_json_array(path, "function definitions")
    fns: List[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            fns.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            raise SystemExit(
                f"error: invalid function definition at index {i}: {exc}"
            ) from exc
    if not fns:
        raise SystemExit("error: no functions defined")
    return fns


def load_prompts(path: Path) -> List[str]:
    """Load and validate the test prompts file."""
    raw = _load_json_array(path, "test prompts")
    prompts: List[str] = []
    for i, item in enumerate(raw):
        try:
            tp = TestPrompt.model_validate(item)
        except ValidationError as exc:
            raise SystemExit(
                f"error: invalid prompt at index {i}: {exc}"
            ) from exc
        prompts.append(tp.prompt)
    return prompts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    functions = load_functions(Path(args.functions_definition))
    prompts = load_prompts(Path(args.input))

    print(f"[cli] loading model {args.model}", file=sys.stderr, flush=True)
    pipeline = FunctionCallingPipeline(model_name=args.model)

    results: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts, start=1):
        print(
            f"[{i}/{len(prompts)}] {prompt}",
            file=sys.stderr,
            flush=True,
        )
        try:
            call = pipeline.process(prompt, functions)
            results.append(call.model_dump())
        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr, flush=True)
            results.append({"prompt": prompt, "error": str(exc)})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[cli] wrote {len(results)} results to {out_path}",
        file=sys.stderr,
        flush=True,
    )
    return 0
