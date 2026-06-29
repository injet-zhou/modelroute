from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from main import create_app


def run(coro):
    return asyncio.run(coro)


async def request_app(
    transport: httpx.MockTransport,
    path: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    app = create_app(
        route_base_url="http://route.test",
        upstream_base_url="http://upstream.test",
        outbound_transport=transport,
    )
    asgi_transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=asgi_transport, base_url="http://modelroute.test") as client:
        return await client.post(path, json=body, headers=headers)


def json_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data, headers={"content-type": "application/json"})


def openai_completion_response(model: str, content: str = "ok") -> dict[str, Any]:
    return {
        "id": "chatcmpl_123",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_openai_chat_decides_then_rewrites_model_and_strips_allowed_models() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            calls["decision_auth"] = request.headers.get("authorization")
            return json_response({
                "model": "provider/selected",
                "metadata": {"tier": "MEDIUM", "confidence": 0.72},
            })
        calls["upstream_body"] = body
        calls["upstream_auth"] = request.headers.get("authorization")
        return json_response(openai_completion_response(body["model"]))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/chat/completions",
        {
            "model": "uncommon-route/auto",
            "messages": [{"role": "user", "content": "hello"}],
            "allowed_models": ["provider/selected"],
        },
        headers={"authorization": "Bearer test-key"},
    ))

    assert response.status_code == 200
    assert calls["decision_body"]["allowed_models"] == ["provider/selected"]
    assert calls["upstream_body"]["model"] == "provider/selected"
    assert "allowed_models" not in calls["upstream_body"]
    assert calls["decision_auth"] == "Bearer test-key"
    assert calls["upstream_auth"] == "Bearer test-key"
    assert response.headers["x-modelroute-selected-model"] == "provider/selected"
    assert response.headers["x-modelroute-decision-tier"] == "MEDIUM"
    assert response.headers["x-modelroute-decision-confidence"] == "0.72"


def test_anthropic_messages_converts_for_decision_and_returns_anthropic_response() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({
                "model": "anthropic-compatible/selected",
                "metadata": {"tier": "COMPLEX", "confidence": 0.91},
            })
        calls["upstream_body"] = body
        return json_response(openai_completion_response(body["model"], "anthropic ok"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {
            "model": "claude-sonnet",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 128,
            "allowed_models": ["anthropic-compatible/selected"],
        },
    ))

    assert response.status_code == 200
    assert calls["decision_body"]["model"] == "claude-sonnet"
    assert calls["decision_body"]["messages"][0] == {"role": "system", "content": "Be concise."}
    assert calls["decision_body"]["messages"][1] == {"role": "user", "content": "hello"}
    assert calls["decision_body"]["allowed_models"] == ["anthropic-compatible/selected"]
    assert calls["upstream_body"]["model"] == "anthropic-compatible/selected"
    assert "allowed_models" not in calls["upstream_body"]

    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "anthropic-compatible/selected"
    assert data["content"] == [{"type": "text", "text": "anthropic ok"}]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_anthropic_x_api_key_is_forwarded_as_bearer_auth() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            calls["decision_auth"] = request.headers.get("authorization")
            calls["decision_x_api_key"] = request.headers.get("x-api-key")
            return json_response({"model": "anthropic-compatible/selected"})
        calls["upstream_auth"] = request.headers.get("authorization")
        calls["upstream_x_api_key"] = request.headers.get("x-api-key")
        body = json.loads(request.content)
        return json_response(openai_completion_response(body["model"], "anthropic ok"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
        },
        headers={"x-api-key": "anthropic-style-key", "anthropic-version": "2023-06-01"},
    ))

    assert response.status_code == 200
    assert calls["decision_auth"] == "Bearer anthropic-style-key"
    assert calls["upstream_auth"] == "Bearer anthropic-style-key"
    assert calls["decision_x_api_key"] == "anthropic-style-key"
    assert calls["upstream_x_api_key"] == "anthropic-style-key"


def test_route_decision_failure_is_returned_without_upstream_call() -> None:
    calls = {"upstream": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"error": {"code": "allowlist_exhausted"}}, status_code=400)
        calls["upstream"] += 1
        return json_response(openai_completion_response("should-not-run"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/chat/completions",
        {"model": "uncommon-route/auto", "messages": [{"role": "user", "content": "hello"}]},
    ))

    assert response.status_code == 400
    assert response.json() == {"error": {"code": "allowlist_exhausted"}}
    assert calls["upstream"] == 0


def test_missing_decision_model_returns_502() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"metadata": {"tier": "SIMPLE"}})

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/chat/completions",
        {"model": "uncommon-route/auto", "messages": [{"role": "user", "content": "hello"}]},
    ))

    assert response.status_code == 502
    assert response.json() == {"error": "route-decision response is missing model"}


def test_openai_stream_is_forwarded_after_decision() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected", "metadata": {"tier": "SIMPLE", "confidence": 0.5}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
        )

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/chat/completions",
        {"model": "uncommon-route/auto", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
    ))

    assert response.status_code == 200
    assert response.headers["x-modelroute-selected-model"] == "provider/selected"
    assert b'data: {"choices":[{"delta":{"content":"hi"}}]}' in response.content


def test_anthropic_stream_is_converted_from_openai_sse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected", "metadata": {"tier": "SIMPLE", "confidence": 0.5}})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
        )

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {"model": "claude", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
    ))

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert '"text": "hi"' in response.text
    assert "event: message_stop" in response.text


def test_anthropic_stream_converts_openai_tool_calls() -> None:
    sse = (
        b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"get_weather","arguments":"{\\"city\\""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":":\\"Paris\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected"})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {"model": "claude", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
    ))

    assert response.status_code == 200
    text = response.text

    # A tool_use content block must be opened with the tool id and name.
    tool_starts = [
        json.loads(line[len("data:"):].strip())
        for line in text.splitlines()
        if line.startswith("data:") and '"content_block_start"' in line
    ]
    tool_block = next(e for e in tool_starts if e["content_block"]["type"] == "tool_use")
    assert tool_block["content_block"]["name"] == "get_weather"
    assert tool_block["content_block"]["id"] == "call_1"

    # input_json_delta fragments must reassemble to the full tool arguments.
    partials = [
        json.loads(line[len("data:"):].strip())["delta"]["partial_json"]
        for line in text.splitlines()
        if line.startswith("data:") and '"input_json_delta"' in line
    ]
    assert json.loads("".join(partials)) == {"city": "Paris"}

    # finish_reason tool_calls maps to stop_reason tool_use.
    assert '"stop_reason": "tool_use"' in text


def test_anthropic_tool_result_becomes_openai_tool_message() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        return json_response(openai_completion_response(body["model"], "done"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {
            "model": "claude",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "18C sunny"}]},
                ]},
            ],
        },
    ))

    assert response.status_code == 200
    messages = calls["upstream_body"]["messages"]

    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    # Assistant content is not polluted with synthetic [tool_use:...] text.
    assert assistant["content"] in (None, "")

    tool_message = next(m for m in messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "toolu_1"
    assert tool_message["content"] == "18C sunny"
    # The tool message immediately follows the assistant tool_calls turn.
    assert messages.index(tool_message) == messages.index(assistant) + 1
    # No Python repr or synthetic tool markers leaked anywhere.
    assert "[tool_use:" not in json.dumps(messages)
    assert "'type':" not in json.dumps(messages)


def test_cache_control_is_preserved_on_all_surfaces() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        return json_response(openai_completion_response(body["model"], "done"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {
            "model": "claude",
            "max_tokens": 64,
            "system": [
                {"type": "text", "text": "stable preamble"},
                {"type": "text", "text": "<big doc>", "cache_control": {"type": "ephemeral"}},
            ],
            "tools": [
                {"name": "get_weather", "description": "w", "input_schema": {"type": "object"},
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "long context", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "ack", "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "res",
                     "cache_control": {"type": "ephemeral"}},
                ]},
            ],
        },
    ))

    assert response.status_code == 200
    body = calls["upstream_body"]

    # System: cache_control rides on the structured text part.
    system = next(m for m in body["messages"] if m["role"] == "system")
    assert system["content"][1]["cache_control"] == {"type": "ephemeral"}

    # User text part: TTL is preserved verbatim.
    user = next(m for m in body["messages"] if m["role"] == "user")
    assert user["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    # Assistant text part.
    assistant = next(m for m in body["messages"] if m["role"] == "assistant")
    assert assistant["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Tool result content part.
    tool_message = next(m for m in body["messages"] if m["role"] == "tool")
    assert tool_message["content"][0]["cache_control"] == {"type": "ephemeral"}

    # Tool definition.
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_control_keeps_plain_string_content() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        return json_response(openai_completion_response(body["model"], "done"))

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {
            "model": "claude",
            "max_tokens": 64,
            "system": "be concise",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        },
    ))

    assert response.status_code == 200
    # Without cache_control, content collapses to compact plain strings (no bloat).
    for message in calls["upstream_body"]["messages"]:
        assert isinstance(message["content"], str)


