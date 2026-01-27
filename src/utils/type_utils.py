# This code is borrowed from proprietary repository
from copy import deepcopy

import random
import string

from typing import Any, Callable, Mapping, Sequence, TypeVar, Literal, Dict
from typing_extensions import TypeIs

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)


_CHARS = string.ascii_letters + string.digits
T = TypeVar("T")

def _new_id(prefix: str, length: int) -> str:
    return f"{prefix}{''.join(random.choices(_CHARS, k=length))}"


def new_chatcmpl_id() -> str:
    return _new_id("chatcmpl-", 29)


def new_cmpl_id() -> str:
    return _new_id("cmpl-", 29)


def new_call_id() -> str:
    return _new_id("call_", 24)


def new_thinking_id() -> str:
    return _new_id("thinking_", 24)


def is_message(msg: Mapping[str, Any]) -> TypeIs[ChatCompletionMessageParam]:
    return "role" in msg


def is_system_message(
    msg: ChatCompletionMessageParam,
) -> TypeIs[ChatCompletionSystemMessageParam]:
    return msg["role"] == "system"


def is_user_message(msg: ChatCompletionMessageParam) -> TypeIs[ChatCompletionUserMessageParam]:
    return msg["role"] == "user"


def is_tool_message(msg: ChatCompletionMessageParam) -> TypeIs[ChatCompletionToolMessageParam]:
    return msg["role"] == "tool"


def is_assistant_message(
    msg: ChatCompletionMessageParam,
) -> TypeIs[ChatCompletionAssistantMessageParam]:
    return msg["role"] == "assistant"


def get_latest_message(
    msgs: Sequence[ChatCompletionMessageParam],
    checker: Callable[[ChatCompletionMessageParam], TypeIs[T]],
) -> T:
    for msg in msgs[::-1]:
        if checker(msg):
            return msg
    raise ValueError("no assistant message found")


def get_latest_user_message(
    msgs: Sequence[ChatCompletionMessageParam],
) -> ChatCompletionUserMessageParam:
    return get_latest_message(msgs, is_user_message)


def get_latest_assistant_message(
    msgs: Sequence[ChatCompletionMessageParam],
) -> ChatCompletionAssistantMessageParam:
    return get_latest_message(msgs, is_assistant_message)


def get_latest_tool_message(
    msgs: Sequence[ChatCompletionMessageParam],
) -> ChatCompletionToolMessageParam:
    return get_latest_message(msgs, is_tool_message)


def get_called_function(msg: ChatCompletionMessageParam, i: int = 0) -> Mapping[str, Any]:
    if not is_assistant_message(msg) or not msg["tool_calls"]:
        raise ValueError("the given message does not have any tool call")
    return msg["tool_calls"][i]["function"]


def ensure_messages(msgs: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    ret: list[Mapping[str, Any]] = deepcopy(list(msgs))
    for i in range(len(ret)):
        msg = {k: v for k, v in ret[i].items() if v is not None}
        if not is_message(msg):
            raise ValueError(f"{msg} is invalid for OpenAI message! (index = {i})")
        if is_assistant_message(msg):
            tool_calls = ret[i].get("tool_calls")
            if not tool_calls:
                continue
            for j, tc in enumerate(tool_calls):
                tc_id = tc.get("id")
                if not tc_id:
                    tc["id"] = new_call_id()
                    if i + j + 1 < len(ret):
                        tool_msg = ret[i + j + 1]
                        assert is_message(tool_msg) and is_tool_message(tool_msg)
                        tool_msg["tool_call_id"] = tc["id"]
        ret[i] = msg
    return ret


def get_content(msg: ChatCompletionMessageParam) -> str:
    content = msg["content"]
    if isinstance(content, str):
        return content
    acc = ""
    if content:
        for part in content:
            if part["type"] == "text":
                acc += part["text"]
    return acc


def is_user_input(msg: ChatCompletionMessageParam) -> bool:
    """Check if a message is user input."""
    return msg.get("role") == "user"


def is_tool_message(msg: ChatCompletionMessageParam) -> bool:
    """Check if a message is a tool message."""
    return msg.get("role") == "tool"


def normalize_chat_response(raw_response) -> Dict[str, Any]:
    """
    Normalize OpenAI SDK response object OR OpenAI-compatible JSON
    into a unified format with tool calls included.
    
    Returns:
        {
            "content": str or None,
            "role": str,
            "tool_calls": List[Dict],  # Unified format
        }
    """
    def convert_function_call_to_tool_call(fc: dict or Any, index: int = 0) -> Dict:
        # Converts a single function_call to tool_call-style dict
        return {
            "id": f"call_generated_{index}",
            "type": "function",
            "function": {
                "name": getattr(fc, "name", None) if not isinstance(fc, dict) else fc.get("name"),
                "arguments": getattr(fc, "arguments", None) if not isinstance(fc, dict) else fc.get("arguments"),
            }
        }

    if hasattr(raw_response, "content") and hasattr(raw_response, "tool_calls"):
        # OpenAI SDK (ChatCompletionMessage)
        tool_calls = []
        if raw_response.tool_calls:
            tool_calls = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                }
                for call in raw_response.tool_calls
            ]
        elif raw_response.function_call:
            # fallback for legacy OpenAI SDK
            tool_calls = [convert_function_call_to_tool_call(raw_response.function_call)]
        
        return {
            "content": raw_response.content,
            "role": raw_response.role,
            "tool_calls": tool_calls,
        }

    # OpenAI-compatible API
    elif isinstance(raw_response, dict) and "role" in raw_response:
        tool_calls = []
        if "tool_calls" in raw_response and raw_response["tool_calls"]:
            tool_calls = raw_response["tool_calls"]
        elif "function_call" in raw_response and raw_response["function_call"]:
            tool_calls = [convert_function_call_to_tool_call(raw_response["function_call"])]
        
        return {
            "content": raw_response.get("content") if raw_response.get("content") is not None else "\n\n",
            "role": raw_response.get("role"),
            "tool_calls": tool_calls,
        }

    raise ValueError("Unsupported chat response format")
