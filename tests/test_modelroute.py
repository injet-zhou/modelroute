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
