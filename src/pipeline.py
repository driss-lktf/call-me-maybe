"""High-level pipeline: prompt -> constrained decoding -> validated FunctionCall."""
import json
from typing import Dict, List

from llm_sdk import Small_LLM_Model

from .decoder import ConstrainedDecoder
from .prompt import build_prompt
from .schemas import FunctionCall, FunctionDefinition, JsonValue


class FunctionCallingPipeline:
    """Drive the LLM through one prompt at a time and produce structured output."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = Small_LLM_Model(model_name=model_name)
        self.decoder = ConstrainedDecoder(self.model)

    def process(
        self,
        prompt_text: str,
        functions: List[FunctionDefinition],
    ) -> FunctionCall:
        """Translate ``prompt_text`` into a single :class:`FunctionCall`."""
        if not functions:
            raise ValueError("functions list must be non-empty")

        prompt = build_prompt(functions, prompt_text)
        input_ids = self.decoder.encode(prompt)

        self.decoder.force_emit(input_ids, '{"name": "')
        fn_name = self.decoder.select_from_set(
            input_ids, [fn.name for fn in functions]
        )
        self.decoder.force_emit(input_ids, '", "parameters": {')

        chosen = next(fn for fn in functions if fn.name == fn_name)
        params: Dict[str, JsonValue] = {}
        items = list(chosen.parameters.items())
        for i, (param_name, schema) in enumerate(items):
            self.decoder.force_emit(input_ids, f'"{param_name}": ')
            params[param_name] = self._generate_value(input_ids, schema.type)
            if i < len(items) - 1:
                self.decoder.force_emit(input_ids, ", ")
        self.decoder.force_emit(input_ids, "}}")

        return FunctionCall(
            prompt=prompt_text, name=fn_name, parameters=params
        )

    def _generate_value(self, input_ids: List[int], type_: str) -> JsonValue:
        if type_ == "number":
            text = self.decoder.generate_number(input_ids)
            try:
                return float(text)
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid number generated: {text!r}"
                ) from exc
        if type_ == "integer":
            text = self.decoder.generate_number(input_ids)
            try:
                return int(float(text))
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid integer generated: {text!r}"
                ) from exc
        if type_ == "boolean":
            text = self.decoder.select_from_set(
                input_ids, ["true", "false"]
            )
            return text == "true"
        # string or unknown -> treat as JSON string
        self.decoder.force_emit(input_ids, '"')
        content = self.decoder.generate_string_content(input_ids)
        try:
            value = json.loads(f'"{content}"')
        except json.JSONDecodeError:
            return content
        if isinstance(value, str):
            return value
        return content
