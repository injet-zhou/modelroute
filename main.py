from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
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

FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
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


def upstream_body_from_openai(body: dict[str, Any], selected_model: str) -> dict[str, Any]:
    out = dict(body)
    out["model"] = selected_model
    out.pop("allowed_models", None)
    return out


def openai_message_to_anthropic_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        text = text_from_content(content)
        if text:
            blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        try:
            input_value = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            input_value = {}
        blocks.append({
            "type": "tool_use",
            "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": str(function.get("name") or ""),
            "input": input_value,
        })
    return blocks or [{"type": "text", "text": ""}]


def openai_response_to_anthropic(data: dict[str, Any], selected_model: str) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    finish_reason = str(choice.get("finish_reason") or "stop")
    return {
        "id": str(data.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": selected_model,
        "content": openai_message_to_anthropic_blocks(message),
        "stop_reason": FINISH_TO_STOP_REASON.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def openai_stream_to_anthropic(response: httpx.Response, selected_model: str) -> AsyncIterator[bytes]:
    message_id = f"msg_{uuid.uuid4().hex}"
    yield sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": selected_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    buffer = ""
    finish_reason = "stop"
    output_tokens = 0
    next_index = 0
    open_index: int | None = None  # the currently-open Anthropic content block, if any
    open_kind: str | None = None   # "text" or "tool"
    tool_index_map: dict[int, int] = {}  # OpenAI tool_calls[].index -> Anthropic block index

    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            usage = data.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                output_tokens = int(usage.get("completion_tokens") or 0)

            choices = data.get("choices") if isinstance(data.get("choices"), list) else []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            if choice.get("finish_reason"):
                finish_reason = str(choice.get("finish_reason"))
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

            text = delta.get("content")
            if text:
                if open_kind == "tool":
                    yield sse_event("content_block_stop", {"type": "content_block_stop", "index": open_index})
                    open_index, open_kind = None, None
                if open_kind != "text":
                    open_index, open_kind = next_index, "text"
                    next_index += 1
                    yield sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": open_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": open_index,
                    "delta": {"type": "text_delta", "text": str(text)},
                })

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tc_index = tool_call.get("index")
                    if tc_index is None:
                        tc_index = 0
                    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    if tc_index not in tool_index_map:
                        if open_index is not None:
                            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": open_index})
                        block_index = next_index
                        next_index += 1
                        tool_index_map[tc_index] = block_index
                        open_index, open_kind = block_index, "tool"
                        yield sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": str(function.get("name") or ""),
                                "input": {},
                            },
                        })
                    block_index = tool_index_map[tc_index]
                    arguments = function.get("arguments")
                    if arguments:
                        yield sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": str(arguments)},
                        })

    if open_index is not None:
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": open_index})
    yield sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": FINISH_TO_STOP_REASON.get(finish_reason, "end_turn"), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse_event("message_stop", {"type": "message_stop"})


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
    body: dict[str, Any],
    request: Request,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    decision: dict[str, Any],
    anthropic_response: bool,
) -> Response:
    headers = forwarded_request_headers(request)
    selected_model = str(decision["model"])
    async with httpx.AsyncClient(transport=outbound_transport, timeout=60.0) as client:
        response = await client.post(
            endpoint_url(upstream_base_url, "/v1/chat/completions"),
            json=upstream_body_from_openai(body, selected_model),
            headers=headers,
        )

    response_headers = add_decision_headers(forwarded_response_headers(response), decision)
    if response.status_code < 200 or response.status_code >= 300 or not anthropic_response:
        media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "application/json")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=media_type,
        )

    try:
        data = response.json()
    except ValueError:
        return JSONResponse({"error": "upstream returned non-JSON response"}, status_code=502, headers=response_headers)
    if not isinstance(data, dict):
        return JSONResponse({"error": "upstream response body must be an object"}, status_code=502, headers=response_headers)
    return JSONResponse(openai_response_to_anthropic(data, selected_model), status_code=response.status_code, headers=response_headers)


async def forward_stream(
    *,
    body: dict[str, Any],
    request: Request,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    decision: dict[str, Any],
    anthropic_response: bool,
) -> Response:
    headers = forwarded_request_headers(request)
    selected_model = str(decision["model"])
    client = httpx.AsyncClient(transport=outbound_transport, timeout=None)
    request_to_send = client.build_request(
        "POST",
        endpoint_url(upstream_base_url, "/v1/chat/completions"),
        json=upstream_body_from_openai(body, selected_model),
        headers=headers,
    )
    response = await client.send(request_to_send, stream=True)
    response_headers = add_decision_headers(forwarded_response_headers(response), decision)

    if response.status_code < 200 or response.status_code >= 300:
        content = await response.aread()
        await response.aclose()
        await client.aclose()
        media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "application/json")
        return Response(content=content, status_code=response.status_code, headers=response_headers, media_type=media_type)

    async def raw_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    async def anthropic_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in openai_stream_to_anthropic(response, selected_model):
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    if anthropic_response:
        response_headers.pop("content-type", None)
        return StreamingResponse(
            anthropic_iterator(),
            status_code=response.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )
    media_type = media_type_from_headers(response_headers, response.headers.get("content-type") or "text/event-stream")
    return StreamingResponse(raw_iterator(), status_code=response.status_code, headers=response_headers, media_type=media_type)


async def route_and_forward(
    *,
    request: Request,
    route_base_url: str,
    upstream_base_url: str,
    outbound_transport: httpx.AsyncBaseTransport | None,
    anthropic_request: bool,
) -> Response:
    body_or_response = await parse_json_body(request)
    if isinstance(body_or_response, Response):
        return body_or_response

    decision_body = anthropic_to_openai_request(body_or_response) if anthropic_request else dict(body_or_response)
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
            body=decision_body,
            request=request,
            upstream_base_url=upstream_base_url,
            outbound_transport=outbound_transport,
            decision=decision,
            anthropic_response=anthropic_request,
        )
    return await forward_non_stream(
        body=decision_body,
        request=request,
        upstream_base_url=upstream_base_url,
        outbound_transport=outbound_transport,
        decision=decision,
        anthropic_response=anthropic_request,
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
            anthropic_request=False,
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
            anthropic_request=True,
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
