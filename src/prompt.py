"""Build prompts for the LLM using a Qwen-style chat template.

The empty ``<think></think>`` block disables Qwen3's default reasoning
mode so the assistant jumps straight to producing the JSON answer.
"""
from typing import List

from .schemas import FunctionDefinition


SYSTEM_INSTRUCTIONS = (
    "You are a function calling assistant. Given a user request, "
    "you must choose ONE function from the list below and produce its "
    "arguments as a JSON object. Reply with ONLY a JSON object of the "
    'shape {"name": "<function_name>", "parameters": {<args>}}. '
    "Do not include any explanation."
)


def describe_function(fn: FunctionDefinition) -> str:
    """Return a one-line human description of a function."""
    if fn.parameters:
        params = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in fn.parameters.items()
        )
    else:
        params = ""
    desc = fn.description.strip() or "(no description)"
    return f"- {fn.name}({params}): {desc}"


def build_prompt(
    functions: List[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build the full chat-formatted prompt to feed to the model."""
    fn_lines = "\n".join(describe_function(fn) for fn in functions)
    system = f"{SYSTEM_INSTRUCTIONS}\n\nAvailable functions:\n{fn_lines}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
