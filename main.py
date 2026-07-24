from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8500
ROUTE_BASE_URL_ENV = "MODELROUTE_ROUTE_BASE_URL"
UPSTREAM_BASE_URL_ENV = "MODELROUTE_UPSTREAM_BASE_URL"

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

RESPONSE_HOP_BY_HOP_HEADERS = HOP_BY_HOP_HEADERS | {
    "content-encoding",
}

ROUTING_IMAGE_PLACEHOLDER_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class APIProtocol(StrEnum):
    OPENAI_CHAT = "openai-chat"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"


UPSTREAM_ENDPOINTS = {
    APIProtocol.OPENAI_CHAT: "/v1/chat/completions",
    APIProtocol.OPENAI_RESPONSES: "/v1/responses",
    APIProtocol.ANTHROPIC_MESSAGES: "/v1/messages",
}


def endpoint_url(base_url: str, endpoint: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("base URL is required")
    if base.endswith(endpoint):
        return base
    if endpoint.startswith("/v1/") and base.endswith("/v1"):
        return f"{base}{endpoint[3:]}"
    return f"{base}{endpoint}"


def upstream_endpoint_for_protocol(base_url: str, protocol: APIProtocol) -> str:
    return endpoint_url(base_url, UPSTREAM_ENDPOINTS[protocol])


def forwarded_request_headers(request: Request) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    has_authorization = any(name.lower() == "authorization" for name in headers)
    if not has_authorization:
        x_api_key = next((value for name, value in headers.items() if name.lower() == "x-api-key"), None)
        if x_api_key:
            headers["authorization"] = f"Bearer {x_api_key}"
    return headers


def forwarded_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in RESPONSE_HOP_BY_HOP_HEADERS
    }


def add_decision_headers(headers: dict[str, str], decision: dict[str, Any]) -> dict[str, str]:
    out = dict(headers)
    model = str(decision.get("model") or "").strip()
    if model:
        out["x-modelroute-selected-model"] = model

    metadata = decision.get("metadata")
    if isinstance(metadata, dict):
        tier = metadata.get("tier")
        confidence = metadata.get("confidence")
        if tier is not None:
            out["x-modelroute-decision-tier"] = str(tier)
        if confidence is not None:
            out["x-modelroute-decision-confidence"] = str(confidence)
    return out


def media_type_from_headers(headers: dict[str, str], default: str) -> str:
    content_type = headers.pop("content-type", None) or headers.pop("Content-Type", None)
    return content_type or default


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text") or ""))
            elif block_type == "tool_result":
                parts.append(str(block.get("content") or ""))
            elif block_type == "tool_use":
                name = block.get("name") or "tool"
                parts.append(f"[tool_use:{name}] {json.dumps(block.get('input') or {}, ensure_ascii=False)}")
        return "\n".join(part for part in parts if part)
    return str(content)


def text_blocks_only(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def cache_control_of(block: Any) -> dict[str, Any] | None:
    if isinstance(block, dict):
        cc = block.get("cache_control")
        if isinstance(cc, dict):
            return cc
    return None


def anthropic_system_to_openai(system: Any) -> Any:
    if not isinstance(system, list):
        return text_from_content(system)

    out: list[dict[str, Any]] = []
    has_cache_control = False
    for block in system:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        part: dict[str, Any] = {"type": "text", "text": str(block.get("text") or "")}
        cc = cache_control_of(block)
        if cc:
            part["cache_control"] = cc
            has_cache_control = True
        out.append(part)

    if not out:
        return ""
    if not has_cache_control:
        return "\n".join(str(item.get("text") or "") for item in out)
    return out


def anthropic_user_content_to_openai(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return text_from_content(content)

    out: list[dict[str, Any]] = []
    has_cache_control = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        cc = cache_control_of(block)
        if block_type == "text":
            part: dict[str, Any] = {"type": "text", "text": str(block.get("text") or "")}
        elif block_type == "image":
            source = block.get("source")
            if not (isinstance(source, dict) and source.get("type") == "base64"):
                continue
            media_type = source.get("media_type") or "image/png"
            data = source.get("data") or ""
            part = {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        else:
            continue
        if cc:
            part["cache_control"] = cc
            has_cache_control = True
        out.append(part)

    if not out:
        return ""
    # Collapse to a plain string only when there is no cache_control to preserve —
    # cache_control can only ride on a structured content part, not a bare string.
    if not has_cache_control and all(item.get("type") == "text" for item in out):
        return "\n".join(str(item.get("text") or "") for item in out)
    return out


def anthropic_assistant_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")

    text_parts: list[dict[str, Any]] = []
    has_cache_control = False
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            part: dict[str, Any] = {"type": "text", "text": str(block.get("text") or "")}
            cc = cache_control_of(block)
            if cc:
                part["cache_control"] = cc
                has_cache_control = True
            text_parts.append(part)

    out: dict[str, Any] = {"role": "assistant"}
    if has_cache_control:
        # Keep the structured array so cache_control survives.
        out["content"] = text_parts
    else:
        out["content"] = text_blocks_only(content)

    tool_calls: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_calls.append({
                "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
    if tool_calls:
        out["tool_calls"] = tool_calls
        # OpenAI expects content to be null (not synthetic text) when only tool calls are present.
        if not has_cache_control and not out["content"]:
            out["content"] = None
    return out


def anthropic_user_message_to_openai(content: Any) -> list[dict[str, Any]]:
    """Convert an Anthropic user turn into one or more OpenAI messages.

    tool_result blocks become standalone `role:"tool"` messages (carrying the
    matching tool_call_id) so the tool_use_id link survives; remaining content
    becomes a single user message placed after the tool messages, as OpenAI
    requires tool results to immediately follow the assistant tool_calls turn.
    """
    if not isinstance(content, list):
        return [{"role": "user", "content": anthropic_user_content_to_openai(content)}]

    tool_messages: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": str(block.get("tool_use_id") or ""),
                "content": text_from_content(block.get("content")),
            }
            cc = cache_control_of(block)
            if cc:
                # Carry cache_control on a structured content part so the breakpoint survives.
                tool_message["content"] = [
                    {"type": "text", "text": text_from_content(block.get("content")), "cache_control": cc}
                ]
            tool_messages.append(tool_message)
        else:
            remaining.append(block)

    messages = list(tool_messages)
    if remaining or not tool_messages:
        messages.append({"role": "user", "content": anthropic_user_content_to_openai(remaining)})
    return messages


def anthropic_to_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": [],
    }

    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "stream", "allowed_models"):
        if key in body:
            out[key] = body[key]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]

    system = body.get("system")
    if system:
        out["messages"].append({"role": "system", "content": anthropic_system_to_openai(system)})

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            out["messages"].extend(anthropic_user_message_to_openai(message.get("content")))
        elif role == "assistant":
            out["messages"].append(anthropic_assistant_to_openai(message))

    tools = body.get("tools")
    if isinstance(tools, list):
        converted_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            converted: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
            cc = cache_control_of(tool)
            if cc:
                converted["cache_control"] = cc
            converted_tools.append(converted)
        out["tools"] = converted_tools

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            out["tool_choice"] = "auto"
        elif choice_type == "any":
            out["tool_choice"] = "required"
        elif choice_type == "tool":
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": str(tool_choice.get("name") or "")},
            }

    return out


def responses_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [
            str(item.get("text") or "")
            for item in value
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}
        ]
        if parts:
            return "\n".join(part for part in parts if part)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def responses_placeholder_text(kind: str, value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        encoded = str(value)
    return f"[Responses {kind}] {encoded}"


def responses_synthetic_tool_name(item_type: Any, name: Any = None) -> str:
    raw = "_".join(part for part in (str(item_type or "tool"), str(name or "")) if part)
    safe = "".join(
        char if char.isascii() and (char.isalnum() or char in {"_", "-"}) else "_"
        for char in raw
    )
    safe = safe.strip("_-") or "tool"
    return f"responses_{safe}"[:64]


def responses_content_part_for_decision(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"type": "text", "text": responses_placeholder_text("content", block)}

    block_type = str(block.get("type") or "")
    if block_type in {"text", "input_text", "output_text"}:
        return {"type": "text", "text": str(block.get("text") or "")}
    if block_type == "refusal":
        return {"type": "text", "text": str(block.get("refusal") or "")}
    if block_type == "input_image":
        image_url = block.get("image_url")
        part: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": str(image_url or ROUTING_IMAGE_PLACEHOLDER_DATA_URL)},
        }
        detail = block.get("detail")
        if detail in {"auto", "low", "high"}:
            part["image_url"]["detail"] = detail
        elif detail == "original":
            part["image_url"]["detail"] = "high"
        return part
    if block_type == "input_file":
        file_value = {
            key: block[key]
            for key in ("file_data", "file_id", "filename")
            if block.get(key) is not None
        }
        if file_value and ("file_id" in file_value or "file_data" in file_value):
            return {"type": "file", "file": file_value}
        return {"type": "text", "text": responses_placeholder_text("input_file", block)}
    return {"type": "text", "text": responses_placeholder_text(block_type or "content", block)}


def responses_message_for_decision(item: dict[str, Any]) -> dict[str, Any]:
    source_role = str(item.get("role") or "user")
    role = source_role if source_role in {"user", "system", "developer", "assistant"} else "user"
    content = item.get("content")
    if isinstance(content, str):
        converted_content: Any = content
    elif isinstance(content, list):
        converted_parts = [responses_content_part_for_decision(block) for block in content]
        if role != "user":
            converted_parts = [
                part
                if part.get("type") == "text"
                else {
                    "type": "text",
                    "text": responses_placeholder_text("non-user multimodal content", part),
                }
                for part in converted_parts
            ]
        converted_content = converted_parts
    else:
        converted_content = responses_value_to_text(content)

    message: dict[str, Any] = {"role": role, "content": converted_content}
    if role != source_role:
        message["content"] = f"[Responses role={source_role}] {responses_value_to_text(converted_content)}"
    return message


def responses_tool_call_arguments(item: dict[str, Any], item_type: str) -> str:
    if item_type == "function_call":
        return str(item.get("arguments") or "{}")
    if item_type == "custom_tool_call":
        return json.dumps({"input": item.get("input") or ""}, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)


def responses_input_item_for_decision(
    item: Any,
    *,
    index: int,
    known_call_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    if not isinstance(item, dict):
        return ([{"role": "user", "content": responses_placeholder_text("input item", item)}], set())

    item_type = str(item.get("type") or "message")
    if item_type == "message" or ("type" not in item and isinstance(item.get("role"), str)):
        return [responses_message_for_decision(item)], set()

    if item_type == "function_call":
        name = str(item.get("name") or responses_synthetic_tool_name(item_type))
        call_id = str(item.get("call_id") or item.get("id") or f"call_responses_{index}")
        known_call_ids.add(call_id)
        return [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": responses_tool_call_arguments(item, item_type),
                },
            }],
        }], {name}

    is_tool_output = item_type == "function_call_output" or item_type.endswith("_call_output")
    if is_tool_output:
        call_id = str(item.get("call_id") or item.get("id") or f"call_responses_{index}")
        messages: list[dict[str, Any]] = []
        synthetic_tools: set[str] = set()
        if call_id not in known_call_ids:
            synthetic_name = responses_synthetic_tool_name(item_type.removesuffix("_output"))
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": synthetic_name, "arguments": "{}"},
                }],
            })
            known_call_ids.add(call_id)
            synthetic_tools.add(synthetic_name)
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": responses_value_to_text(
                item.get("output", item.get("result", item.get("content")))
            ),
        })
        return messages, synthetic_tools

    if item_type.endswith("_call") and item_type != "mcp_approval_request":
        name = str(item.get("name") or responses_synthetic_tool_name(item_type))
        if item_type != "custom_tool_call":
            name = responses_synthetic_tool_name(item_type, item.get("name"))
        call_id = str(item.get("call_id") or item.get("id") or f"call_responses_{index}")
        known_call_ids.add(call_id)
        return [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": responses_tool_call_arguments(item, item_type),
                },
            }],
        }], {name}

    if "tool" in item_type:
        name = responses_synthetic_tool_name(item_type)
        return ([{
            "role": "user",
            "content": responses_placeholder_text(item_type, item),
        }], {name})

    return ([{
        "role": "user",
        "content": responses_placeholder_text(item_type or "input item", item),
    }], set())


def responses_tool_for_decision(tool: Any, *, index: int) -> tuple[dict[str, Any], str]:
    if isinstance(tool, dict) and tool.get("type") == "function":
        name = str(tool.get("name") or f"responses_function_{index}")
        function: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {},
        }
        for key in ("description", "strict"):
            if key in tool:
                function[key] = tool[key]
        return {"type": "function", "function": function}, name

    item_type = tool.get("type") if isinstance(tool, dict) else "tool"
    source_name = tool.get("name") if isinstance(tool, dict) else None
    name = responses_synthetic_tool_name(item_type, source_name if source_name else str(index))
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Routing placeholder for the Responses {item_type or 'tool'} tool.",
            "parameters": {"type": "object", "additionalProperties": True},
            "strict": False,
        },
    }, name


def responses_response_format_for_decision(text_config: Any) -> dict[str, Any] | None:
    if not isinstance(text_config, dict) or not isinstance(text_config.get("format"), dict):
        return None
    format_config = text_config["format"]
    format_type = format_config.get("type")
    if format_type == "json_schema":
        json_schema = {
            key: format_config[key]
            for key in ("name", "description", "schema", "strict")
            if key in format_config
        }
        return {"type": "json_schema", "json_schema": json_schema}
    if format_type == "json_object":
        return {"type": "json_object"}
    return None


def responses_tool_choice_for_decision(
    tool_choice: Any,
    *,
    source_tool_names: dict[str, str],
) -> Any:
    if isinstance(tool_choice, str) and tool_choice in {"auto", "none", "required"}:
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = str(tool_choice.get("type") or "")
    if choice_type == "function":
        name = str(tool_choice.get("name") or "")
    else:
        source_name = str(tool_choice.get("name") or "")
        name = source_tool_names.get(
            f"{choice_type}:{source_name}",
            source_tool_names.get(choice_type, ""),
        )
    if not name:
        return None
    return {"type": "function", "function": {"name": name}}


def responses_to_route_decision_request(body: dict[str, Any]) -> dict[str, Any]:
    """Project a Responses request into a valid Chat Completions request shape."""
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": responses_value_to_text(instructions)})
    if body.get("prompt") is not None:
        messages.append({
            "role": "system",
            "content": responses_placeholder_text("prompt template", body["prompt"]),
        })

    known_call_ids: set[str] = set()
    synthetic_tool_names: set[str] = set()
    input_value = body.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, dict):
        converted, synthetic = responses_input_item_for_decision(
            input_value,
            index=0,
            known_call_ids=known_call_ids,
        )
        messages.extend(converted)
        synthetic_tool_names.update(synthetic)
    elif isinstance(input_value, list):
        for index, item in enumerate(input_value):
            converted, synthetic = responses_input_item_for_decision(
                item,
                index=index,
                known_call_ids=known_call_ids,
            )
            messages.extend(converted)
            synthetic_tool_names.update(synthetic)

    if not any(message.get("role") == "user" for message in messages):
        state_kind = "continuation" if body.get("previous_response_id") or body.get("conversation") else "request"
        messages.append({"role": "user", "content": f"[Responses {state_kind} without inline user text]"})

    out: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
    }
    for key in (
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "store",
        "metadata",
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "user",
        "allowed_models",
    ):
        if key in body:
            out[key] = body[key]
    if "max_output_tokens" in body:
        out["max_completion_tokens"] = body["max_output_tokens"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        out["reasoning_effort"] = reasoning["effort"]
    text_config = body.get("text")
    if isinstance(text_config, dict) and text_config.get("verbosity") is not None:
        out["verbosity"] = text_config["verbosity"]
    response_format = responses_response_format_for_decision(text_config)
    if response_format is not None:
        out["response_format"] = response_format

    converted_tools: list[dict[str, Any]] = []
    source_tool_names: dict[str, str] = {}
    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            converted, converted_name = responses_tool_for_decision(tool, index=index)
            converted_tools.append(converted)
            if isinstance(tool, dict):
                tool_type = str(tool.get("type") or "")
                source_name = str(tool.get("name") or "")
                source_tool_names[tool_type] = converted_name
                source_tool_names[f"{tool_type}:{source_name}"] = converted_name

    existing_names = {
        str(tool.get("function", {}).get("name") or "")
        for tool in converted_tools
        if isinstance(tool.get("function"), dict)
    }
    for name in sorted(synthetic_tool_names - existing_names):
        converted_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": "Routing placeholder for a Responses input tool call.",
                "parameters": {"type": "object", "additionalProperties": True},
                "strict": False,
            },
        })
    if converted_tools:
        out["tools"] = converted_tools

    converted_tool_choice = responses_tool_choice_for_decision(
        body.get("tool_choice"),
        source_tool_names=source_tool_names,
    )
    if converted_tool_choice is not None:
        out["tool_choice"] = converted_tool_choice
    return out


def upstream_body_from_source(body: dict[str, Any], selected_model: str) -> dict[str, Any]:
    out = dict(body)
    out["model"] = selected_model
    out.pop("allowed_models", None)
    return out


async def parse_json_body(request: Request) -> dict[str, Any] | Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "request body must be an object"}, status_code=400)
    return body


async def call_route_decision(
    *,
    body: dict[str, Any],
    request: Request,
    route_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any] | Response:
    headers = forwarded_request_headers(request)
    async with httpx.AsyncClient(transport=outbound_transport, timeout=60.0) as client:
        response = await client.post(endpoint_url(route_base_url, "/v1/route-decision"), json=body, headers=headers)
    if response.status_code < 200 or response.status_code >= 300:
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=forwarded_response_headers(response),
            media_type=response.headers.get("content-type") or "application/json",
        )
    try:
        decision = response.json()
    except ValueError:
        return JSONResponse({"error": "route-decision returned non-JSON response"}, status_code=502)
    if not isinstance(decision, dict) or not str(decision.get("model") or "").strip():
        return JSONResponse({"error": "route-decision response is missing model"}, status_code=502)
    return decision


async def forward_non_stream(
    *,
    source_body: dict[str, Any],
    request: Request,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    decision: dict[str, Any],
    protocol: APIProtocol,
) -> Response:
    headers = forwarded_request_headers(request)
    selected_model = str(decision["model"])
    upstream_endpoint = upstream_endpoint_for_protocol(upstream_base_url, protocol)
    upstream_body = upstream_body_from_source(source_body, selected_model)

    async with httpx.AsyncClient(transport=outbound_transport, timeout=60.0) as client:
        response = await client.post(upstream_endpoint, json=upstream_body, headers=headers)

    response_headers = add_decision_headers(forwarded_response_headers(response), decision)
    media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "application/json")

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=response_headers,
        media_type=media_type,
    )


async def forward_stream(
    *,
    source_body: dict[str, Any],
    request: Request,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    decision: dict[str, Any],
    protocol: APIProtocol,
) -> Response:
    headers = forwarded_request_headers(request)
    selected_model = str(decision["model"])
    upstream_endpoint = upstream_endpoint_for_protocol(upstream_base_url, protocol)
    upstream_body = upstream_body_from_source(source_body, selected_model)

    client = httpx.AsyncClient(transport=outbound_transport, timeout=None)
    request_to_send = client.build_request("POST", upstream_endpoint, json=upstream_body, headers=headers)
    response = await client.send(request_to_send, stream=True)
    response_headers = add_decision_headers(forwarded_response_headers(response), decision)

    if response.status_code < 200 or response.status_code >= 300:
        content = await response.aread()
        await response.aclose()
        await client.aclose()
        media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "application/json")
        return Response(content=content, status_code=response.status_code, headers=response_headers, media_type=media_type)

    async def stream_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "text/event-stream")
    return StreamingResponse(stream_iterator(), status_code=response.status_code, headers=response_headers, media_type=media_type)


async def route_and_forward(
    *,
    request: Request,
    route_base_url: str,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    protocol: APIProtocol,
) -> Response:
    body_or_response = await parse_json_body(request)
    if isinstance(body_or_response, Response):
        return body_or_response

    source_body = body_or_response
    if protocol is APIProtocol.ANTHROPIC_MESSAGES:
        decision_body = anthropic_to_openai_request(source_body)
    elif protocol is APIProtocol.OPENAI_RESPONSES:
        decision_body = responses_to_route_decision_request(source_body)
    else:
        decision_body = dict(source_body)
    decision = await call_route_decision(
        body=decision_body,
        request=request,
        route_base_url=route_base_url,
        outbound_transport=outbound_transport,
    )
    if isinstance(decision, Response):
        return decision

    stream = bool(decision_body.get("stream"))
    if stream:
        return await forward_stream(
            source_body=source_body,
            request=request,
            upstream_base_url=upstream_base_url,
            outbound_transport=outbound_transport,
            decision=decision,
            protocol=protocol,
        )
    return await forward_non_stream(
        source_body=source_body,
        request=request,
        upstream_base_url=upstream_base_url,
        outbound_transport=outbound_transport,
        decision=decision,
        protocol=protocol,
    )


def create_app(
    *,
    route_base_url: str | None = None,
    upstream_base_url: str | None = None,
    outbound_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    route_url = route_base_url or os.environ.get(ROUTE_BASE_URL_ENV, "")
    upstream_url = upstream_base_url or os.environ.get(UPSTREAM_BASE_URL_ENV, "")
    app = FastAPI(title="modelroute", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "route_base_url_configured": bool(route_url),
            "upstream_base_url_configured": bool(upstream_url),
            "time": int(time.time()),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        if not route_url or not upstream_url:
            return JSONResponse({"error": "route and upstream base URLs must be configured"}, status_code=500)
        return await route_and_forward(
            request=request,
            route_base_url=route_url,
            upstream_base_url=upstream_url,
            outbound_transport=outbound_transport,
            protocol=APIProtocol.OPENAI_CHAT,
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        if not route_url or not upstream_url:
            return JSONResponse({"error": "route and upstream base URLs must be configured"}, status_code=500)
        return await route_and_forward(
            request=request,
            route_base_url=route_url,
            upstream_base_url=upstream_url,
            outbound_transport=outbound_transport,
            protocol=APIProtocol.OPENAI_RESPONSES,
        )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        if not route_url or not upstream_url:
            return JSONResponse({"error": "route and upstream base URLs must be configured"}, status_code=500)
        return await route_and_forward(
            request=request,
            route_base_url=route_url,
            upstream_base_url=upstream_url,
            outbound_transport=outbound_transport,
            protocol=APIProtocol.ANTHROPIC_MESSAGES,
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI/Anthropic route-decision forwarding test tool")
    parser.add_argument("--host", default=os.environ.get("MODELROUTE_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MODELROUTE_PORT", str(DEFAULT_PORT))))
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
