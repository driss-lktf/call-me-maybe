"""Pydantic models used to validate input and output JSON files."""
from typing import Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


JsonValue = Union[str, float, int, bool]


class ParameterSchema(BaseModel):
    """Type schema attached to a single function parameter."""

    model_config = ConfigDict(extra="ignore")

    type: str


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
