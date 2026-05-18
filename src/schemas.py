"""Pydantic models used to validate input and output JSON files.

Extended for the bonus part to support nested / complex parameter types:
- ``object`` parameters carry a ``properties`` dict of sub-schemas.
- ``array`` parameters carry an ``items`` sub-schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# A JSON leaf value (string, number, bool, or null-equivalent integer).
JsonLeaf = Union[str, float, int, bool]
# A JSON value may also be a nested dict or list.
JsonValue = Union[JsonLeaf, Dict[str, Any], List[Any]]


class ParameterSchema(BaseModel):
    """Type schema attached to a single function parameter.

    Supports primitive types (``string``, ``number``, ``integer``,
    ``boolean``) as well as:
    - ``object`` — a nested JSON object described by ``properties``.
    - ``array``  — a JSON array whose elements follow ``items``.
    """

    model_config = ConfigDict(extra="ignore")

    type: str
    # For type == "object": maps child parameter names to their schemas.
    properties: Optional[Dict[str, ParameterSchema]] = None
    # For type == "array": the schema of each element.
    items: Optional[ParameterSchema] = None


# Allow forward references (ParameterSchema references itself).
ParameterSchema.model_rebuild()


class FunctionDefinition(BaseModel):
    """Description of a callable function the LLM may select."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: Dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: Optional[ParameterSchema] = None


class TestPrompt(BaseModel):
    """One entry in the test prompts file."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One function-call result, written to the output JSON array."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, JsonValue]
