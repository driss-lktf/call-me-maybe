"""High-level pipeline: prompt -> constrained decoding -> validated FunctionCall.

Extended for the bonus part:
- Passes model_name and use_cache to the decoder so caching is respected.
- Passes an optional visualizer to the decoder for live tracing.
- Supports nested ``object`` and ``array`` parameter types.
"""
import json
from typing import Any, Dict, List, Optional

from llm_sdk import Small_LLM_Model

from .decoder import ConstrainedDecoder
from .prompt import build_prompt
from .schemas import FunctionCall, FunctionDefinition, JsonValue, ParameterSchema
from .visualizer import GenerationVisualizer


class FunctionCallingPipeline:
    """Drive the LLM through one prompt at a time and produce structured output.

    Args:
        model_name: Hugging Face model identifier (default: Qwen/Qwen3-0.6B).
        use_cache: Persist the decoded vocabulary between runs.
        visualizer: Optional live-trace helper.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        use_cache: bool = True,
        visualizer: Optional[GenerationVisualizer] = None,
    ) -> None:
        self._model_name = model_name
        self._viz = visualizer or GenerationVisualizer(enabled=False)
        self.model = Small_LLM_Model(model_name=model_name)
        self.decoder = ConstrainedDecoder(
            self.model,
            model_name=model_name,
            use_cache=use_cache,
            visualizer=self._viz,
        )

    def process(
        self,
        prompt_text: str,
        functions: List[FunctionDefinition],
    ) -> FunctionCall:
        """Translate *prompt_text* into a single :class:`FunctionCall`.

        Args:
            prompt_text: The natural-language user request.
            functions: Available function definitions.

        Returns:
            A validated :class:`FunctionCall` with the chosen function name
            and all parameters filled in.

        Raises:
            ValueError: If *functions* is empty.
            RuntimeError: If constrained decoding cannot produce a valid
                token at some step.
        """
        if not functions:
            raise ValueError("functions list must be non-empty")

        if self._viz.enabled:
            self._viz.new_prompt(f'"{prompt_text}"')

        prompt = build_prompt(functions, prompt_text)
        input_ids = self.decoder.encode(prompt)

        self.decoder.force_emit(input_ids, '{"name": "')
        fn_name = self.decoder.select_from_set(
            input_ids,
            [fn.name for fn in functions],
            stage="fn_name",
        )
        self.decoder.force_emit(input_ids, '", "parameters": {')

        chosen = next(fn for fn in functions if fn.name == fn_name)
        params: Dict[str, JsonValue] = {}
        items = list(chosen.parameters.items())
        for i, (param_name, schema) in enumerate(items):
            self.decoder.force_emit(input_ids, f'"{param_name}": ')
            params[param_name] = self._generate_value(input_ids, schema)
            if i < len(items) - 1:
                self.decoder.force_emit(input_ids, ", ")
        self.decoder.force_emit(input_ids, "}}")

        return FunctionCall(
            prompt=prompt_text, name=fn_name, parameters=params
        )

    # ---------------------------------------------------------- value helpers

    def _generate_value(
        self,
        input_ids: List[int],
        schema: ParameterSchema,
    ) -> JsonValue:
        """Generate a single JSON value conforming to *schema*.

        Dispatches to a type-specific generator.  Supports ``string``,
        ``number``, ``integer``, ``boolean``, ``object``, and ``array``.

        Args:
            input_ids: Running context, modified in-place.
            schema: The expected type/structure.

        Returns:
            A Python value (str, float, int, bool, dict, or list).
        """
        t = schema.type
        if t == "number":
            return self._gen_number(input_ids)
        if t == "integer":
            return self._gen_integer(input_ids)
        if t == "boolean":
            text = self.decoder.select_from_set(
                input_ids, ["true", "false"], stage="boolean"
            )
            return text == "true"
        if t == "object":
            return self._gen_object(input_ids, schema)
        if t == "array":
            return self._gen_array(input_ids, schema)
        # Default: string
        self.decoder.force_emit(input_ids, '"')
        content = self.decoder.generate_string_content(input_ids)
        return self._clean_string(content)

    def _gen_number(self, input_ids: List[int]) -> float:
        """Generate a JSON number and return it as a float."""
        text = self.decoder.generate_number(input_ids)
        try:
            return float(text)
        except ValueError as exc:
            raise RuntimeError(f"invalid number generated: {text!r}") from exc

    def _gen_integer(self, input_ids: List[int]) -> int:
        """Generate a JSON number and return it as an int."""
        text = self.decoder.generate_number(input_ids)
        try:
            return int(float(text))
        except ValueError as exc:
            raise RuntimeError(
                f"invalid integer generated: {text!r}"
            ) from exc

    @staticmethod
    def _clean_string(content: str) -> str:
        """Unescape a JSON string and strip leading whitespace artefacts."""
        try:
            value = json.loads(f'"{content}"')
        except json.JSONDecodeError:
            return content.lstrip()
        if isinstance(value, str):
            return value.lstrip()
        return content.lstrip()

    def _gen_object(
        self,
        input_ids: List[int],
        schema: ParameterSchema,
    ) -> Dict[str, Any]:
        """Generate a nested JSON object.

        If *schema.properties* is defined, each property is generated in
        declaration order.  Otherwise an empty object ``{}`` is emitted.

        Args:
            input_ids: Running context, modified in-place.
            schema: The object schema (may carry ``properties``).

        Returns:
            A Python dict.
        """
        self.decoder.force_emit(input_ids, "{")
        result: Dict[str, Any] = {}

        if schema.properties:
            prop_items = list(schema.properties.items())
            for i, (key, sub_schema) in enumerate(prop_items):
                self.decoder.force_emit(input_ids, f'"{key}": ')
                result[key] = self._generate_value(input_ids, sub_schema)
                if i < len(prop_items) - 1:
                    self.decoder.force_emit(input_ids, ", ")

        self.decoder.force_emit(input_ids, "}")
        return result

    def _gen_array(
        self,
        input_ids: List[int],
        schema: ParameterSchema,
        max_items: int = 8,
    ) -> List[Any]:
        """Generate a JSON array with a model-chosen number of elements.

        The model is asked (via :py:meth:`select_from_set`) whether to add
        another element after each one, up to *max_items*.

        Args:
            input_ids: Running context, modified in-place.
            schema: The array schema (may carry ``items``).
            max_items: Maximum number of elements to generate.

        Returns:
            A Python list.
        """
        self.decoder.force_emit(input_ids, "[")
        result: List[Any] = []
        item_schema = schema.items or ParameterSchema(type="string")

        for i in range(max_items):
            # Generate one element.
            result.append(self._generate_value(input_ids, item_schema))

            # Let the model decide: continue or close the array.
            decision = self.decoder.select_from_set(
                input_ids,
                [", ", "]"],
                stage="array_sep",
            )
            if decision == "]":
                break
            # decision is ", " — already emitted, loop for next element.

        else:
            # Hit max_items — close array unconditionally.
            self.decoder.force_emit(input_ids, "]")

        return result
