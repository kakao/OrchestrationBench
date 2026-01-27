# This code is borrowed from proprietary repository
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal, Self

from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, Field


__all__ = [
    "ErrorResponse",
    "Model",
    "ModelList",
    "UsageInfo",
    "StreamOptions",
    "ResponseFormat",
    "JSONSchemaResponseFormat",
]


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")  # OpenAI API does not allow extra fields


class ErrorResponse(OpenAIBaseModel):
    object: Literal["error"] = "error"
    """The type name of error response object, which is always "error"."""

    code: int
    """The error code"""

    message: str
    """The detail error message"""

    type: str
    """The type of error"""

    param: Optional[str] = None
    """The error parameter"""


class Model(OpenAIBaseModel):
    object: Literal["model"] = "model"
    """The type name of model object, which is always "model"."""

    id: str
    """The model identifier, which can be referenced in the API endpoints."""

    created: int
    """The Unix timestamp (in seconds) when the model was created."""

    owned_by: str = "ovis"
    """The organization that owns the model."""


class ModelList(OpenAIBaseModel):
    object: Literal["list"] = "list"
    """The type name of model list, which is always "list"."""

    data: list[Model]
    """The models which are served by this endpoint."""


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    """The number of tokens in the given prompt."""

    total_tokens: int = 0
    """The number of tokens used in the request (prompt + completion)."""

    completion_tokens: int = 0
    """The number of tokens in the generated completion."""


class JSONSchemaResponseFormat(OpenAIBaseModel):
    name: str

    description: Optional[str] = None

    json_schema: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    """The JSON Schema object to define the response."""

    strict: Optional[bool] = None


class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json_schema", "json_object", "ebnf", "regex", "lark"]
    """The format type of response, which must be one of "text", "json_object", "json_schema",
    "ebnf", "regex", and "lark".
    """

    json_schema: Optional[JSONSchemaResponseFormat] = None
    """The JSON Schema object to define the response."""

    ebnf: Optional[str] = None
    """The EBNF string to define the response."""

    regex: Optional[str] = None
    """The regular expression string to define the response."""

    lark: Optional[str] = None
    """The Lark grammar string to define the response."""

    @model_validator(mode="after")
    def check_response_format(self) -> Self:
        has_json_schema = self.json_schema is not None
        has_ebnf = self.ebnf is not None or self.ebnf == ""
        has_regex = self.regex is not None or self.regex == ""
        has_lark = self.lark is not None or self.lark == ""
        if (self.type == "text" or self.type == "json_object") and (
            has_json_schema or has_ebnf or has_regex or has_lark
        ):
            raise ValueError('"text" or "json_object" format must not have any additional fields.')
        elif self.type == "json_schema" and (
            not has_json_schema or has_ebnf or has_regex or has_lark
        ):
            raise ValueError('"json_schema" format must have JSON Schema only.')
        elif self.type == "ebnf" and (not has_ebnf or has_json_schema or has_regex or has_lark):
            raise ValueError('"ebnf" format must have EBNF only.')
        elif self.type == "regex" and (not has_regex or has_json_schema or has_ebnf or has_lark):
            raise ValueError('"regex" format must have regular expression only.')
        elif self.type == "lark" and (not has_lark or has_json_schema or has_ebnf or has_regex):
            raise ValueError('"lark" format must have Lark grammar only.')
        return self


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    """If set to true, an additional chunk will be streamed before the finish
    message (`data: [DONE]`). The `usage` field on this chunk shows the token usage statistics
    for the entier request, and the `choices` field will always be an empty array. All other
    chunks will also include a `usage` field, but with a null value.
    """

    continuous_usage_stats: Optional[bool] = False
    """Whether the usage statistics will be sent in every chunk or not.

    Note that even this parameter is set to true, the finish message (`data: [DONE]`) does not
    include the usage statistics.

    This parameter is not OpenAI-compatible but vLLM supports this.
    """

class CompletionRequest(BaseModel):
    agent_messages: dict[str, list[ChatCompletionMessageParam]] = Field(default_factory=dict)

class CompletionResponse(BaseModel):
    success: bool = Field(default=True)
    messages: list[ChatCompletionMessageParam] = Field(default_factory=list)
    agent_messages: dict[str, list[ChatCompletionMessageParam]] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
