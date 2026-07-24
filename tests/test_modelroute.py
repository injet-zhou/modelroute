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


def openai_responses_response(model: str, content: str = "ok") -> dict[str, Any]:
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


def anthropic_response(model: str, content: str = "ok") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
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
        calls["upstream_path"] = request.url.path
        assert request.url.path == "/v1/messages"
        return json_response(anthropic_response(body["model"], "anthropic ok"))

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
    # Decision still gets OpenAI-converted format
    assert calls["decision_body"]["model"] == "claude-sonnet"
    assert calls["decision_body"]["messages"][0] == {"role": "system", "content": "Be concise."}
    assert calls["decision_body"]["messages"][1] == {"role": "user", "content": "hello"}
    assert calls["decision_body"]["allowed_models"] == ["anthropic-compatible/selected"]

    # Upstream now receives ORIGINAL Anthropic format
    assert calls["upstream_path"] == "/v1/messages"
    assert calls["upstream_body"]["model"] == "anthropic-compatible/selected"
    assert calls["upstream_body"]["system"] == "Be concise."
    assert calls["upstream_body"]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert "allowed_models" not in calls["upstream_body"]

    # Response is Anthropic format (native, not converted)
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
        assert request.url.path == "/v1/messages"
        return json_response(anthropic_response(body["model"], "anthropic ok"))

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


def test_openai_responses_projects_for_decision_and_forwards_original_protocol() -> None:
    calls: dict[str, Any] = {}
    upstream_response = openai_responses_response("provider/selected", "responses ok")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({
                "model": "provider/selected",
                "metadata": {"tier": "COMPLEX", "confidence": 0.88},
            })
        calls["upstream_path"] = request.url.path
        calls["upstream_body"] = body
        return json_response(upstream_response)

    source_body = {
        "model": "auto-router/auto",
        "instructions": "Be concise.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,aGVsbG8=",
                        "detail": "original",
                    },
                    {"type": "input_file", "file_id": "file_123", "filename": "brief.pdf"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"query":"image"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
        "max_output_tokens": 321,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                "strict": True,
            },
            "verbosity": "low",
        },
        "tools": [
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
            {"type": "web_search_preview"},
        ],
        "allowed_models": ["provider/selected"],
        "previous_response_id": "resp_previous",
        "store": False,
        "reasoning": {"effort": "high"},
        "tool_choice": {"type": "web_search_preview"},
    }
    response = run(request_app(httpx.MockTransport(handler), "/v1/responses", source_body))

    assert response.status_code == 200
    decision_body = calls["decision_body"]
    assert decision_body["model"] == "auto-router/auto"
    assert decision_body["messages"][0] == {"role": "system", "content": "Be concise."}
    user_message = decision_body["messages"][1]
    assert user_message["content"][0] == {"type": "text", "text": "What is in this image?"}
    assert user_message["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,aGVsbG8=",
            "detail": "high",
        },
    }
    assert user_message["content"][2] == {
        "type": "file",
        "file": {"file_id": "file_123", "filename": "brief.pdf"},
    }
    assert decision_body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"query":"image"}',
    }
    assert decision_body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "result",
    }
    assert decision_body["max_completion_tokens"] == 321
    assert decision_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "strict": True,
        },
    }
    assert decision_body["reasoning_effort"] == "high"
    assert decision_body["verbosity"] == "low"
    assert decision_body["store"] is False
    assert decision_body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object"},
        },
    }
    built_in_tool = decision_body["tools"][1]
    assert built_in_tool["type"] == "function"
    assert built_in_tool["function"]["name"] == "responses_web_search_preview_1"
    assert built_in_tool["function"]["strict"] is False
    assert decision_body["tool_choice"] == {
        "type": "function",
        "function": {"name": "responses_web_search_preview_1"},
    }
    assert decision_body["allowed_models"] == ["provider/selected"]

    assert calls["upstream_path"] == "/v1/responses"
    assert calls["upstream_body"]["model"] == "provider/selected"
    assert "allowed_models" not in calls["upstream_body"]
    for key in (
        "input",
        "instructions",
        "previous_response_id",
        "store",
        "reasoning",
        "text",
        "tools",
        "tool_choice",
    ):
        assert calls["upstream_body"][key] == source_body[key]
    assert response.json() == upstream_response
    assert response.headers["x-modelroute-selected-model"] == "provider/selected"
    assert response.headers["x-modelroute-decision-tier"] == "COMPLEX"
    assert response.headers["x-modelroute-decision-confidence"] == "0.88"


def test_openai_responses_projects_unique_items_as_valid_chat_placeholders() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        return json_response(openai_responses_response(body["model"]))

    source_body = {
        "model": "auto-router/auto",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": "file_image", "detail": "low"},
                    {"type": "input_file", "file_url": "https://example.test/brief.pdf"},
                ],
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "routing"},
            },
            {
                "type": "computer_call_output",
                "call_id": "computer_1",
                "output": {"type": "computer_screenshot", "image_url": "https://example.test/screen.png"},
            },
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{"type": "function", "name": "deferred_lookup"}],
            },
        ],
        "tools": [{"type": "computer_use_preview", "display_width": 1280, "display_height": 720}],
        "allowed_models": ["provider/selected"],
    }
    response = run(request_app(httpx.MockTransport(handler), "/v1/responses", source_body))

    assert response.status_code == 200
    decision_body = calls["decision_body"]
    image_part, file_url_part = decision_body["messages"][0]["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["image_url"]["detail"] == "low"
    assert file_url_part["type"] == "text"
    assert "input_file" in file_url_part["text"]

    web_call = decision_body["messages"][1]["tool_calls"][0]
    assert web_call["id"] == "ws_1"
    assert web_call["function"]["name"] == "responses_web_search_call"
    assert json.loads(web_call["function"]["arguments"])["action"]["query"] == "routing"

    reconstructed_call = decision_body["messages"][2]["tool_calls"][0]
    assert reconstructed_call == {
        "id": "computer_1",
        "type": "function",
        "function": {"name": "responses_computer_call", "arguments": "{}"},
    }
    assert decision_body["messages"][3]["role"] == "tool"
    assert decision_body["messages"][3]["tool_call_id"] == "computer_1"
    assert decision_body["messages"][4]["role"] == "user"
    assert "additional_tools" in decision_body["messages"][4]["content"]

    projected_tools = {
        tool["function"]["name"]: tool
        for tool in decision_body["tools"]
    }
    assert {
        "responses_computer_use_preview_0",
        "responses_web_search_call",
        "responses_computer_call",
        "responses_additional_tools",
    } <= projected_tools.keys()
    assert all(tool["type"] == "function" for tool in projected_tools.values())

    assert calls["upstream_body"]["model"] == "provider/selected"
    assert calls["upstream_body"]["input"] == source_body["input"]
    assert calls["upstream_body"]["tools"] == source_body["tools"]
    assert "allowed_models" not in calls["upstream_body"]


def test_openai_responses_state_only_continuation_gets_routing_prompt() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        return json_response(openai_responses_response(body["model"]))

    source_body = {
        "model": "auto-router/auto",
        "previous_response_id": "resp_previous",
        "prompt": {"id": "pmpt_router", "version": "2"},
        "input": [],
    }
    response = run(request_app(httpx.MockTransport(handler), "/v1/responses", source_body))

    assert response.status_code == 200
    assert calls["decision_body"]["messages"][0]["role"] == "system"
    assert "pmpt_router" in calls["decision_body"]["messages"][0]["content"]
    assert calls["decision_body"]["messages"][1] == {
        "role": "user",
        "content": "[Responses continuation without inline user text]",
    }
    assert calls["upstream_body"] == {
        "model": "provider/selected",
        "previous_response_id": "resp_previous",
        "prompt": {"id": "pmpt_router", "version": "2"},
        "input": [],
    }


def test_openai_responses_stream_is_forwarded_without_event_conversion() -> None:
    calls: dict[str, Any] = {}
    sse = (
        'event: response.created\n'
        'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}\n\n'
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_path"] = request.url.path
        calls["upstream_body"] = body
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/responses",
        {
            "model": "auto-router/auto",
            "input": "hello",
            "stream": True,
            "allowed_models": ["provider/selected"],
        },
    ))

    assert response.status_code == 200
    assert calls["decision_body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert calls["upstream_path"] == "/v1/responses"
    assert calls["upstream_body"] == {
        "model": "provider/selected",
        "input": "hello",
        "stream": True,
    }
    assert response.text == sse
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-modelroute-selected-model"] == "provider/selected"


def test_openai_responses_stream_error_is_returned_without_sse_conversion() -> None:
    upstream_error = {"error": {"message": "rate limited", "type": "rate_limit_error"}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected"})
        assert request.url.path == "/v1/responses"
        return json_response(upstream_error, status_code=429)

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/responses",
        {"model": "auto-router/auto", "input": "hello", "stream": True},
    ))

    assert response.status_code == 429
    assert response.json() == upstream_error
    assert response.headers["x-modelroute-selected-model"] == "provider/selected"


def test_anthropic_stream_is_forwarded_from_native_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected", "metadata": {"tier": "SIMPLE", "confidence": 0.5}})

        assert request.url.path == "/v1/messages"
        lines = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"' + body["model"] + '","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]
        return httpx.Response(200, content="".join(lines), headers={"content-type": "text/event-stream"})

    response = run(request_app(
        httpx.MockTransport(handler),
        "/v1/messages",
        {"model": "claude", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
    ))

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert '"text":"hi"' in response.text
    assert "event: message_stop" in response.text


def test_anthropic_stream_forwards_native_tool_calls() -> None:
    sse = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",'
        '"model":"provider/selected","content":[],"stop_reason":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":'
        '{"type":"tool_use","id":"call_1","name":"get_weather","input":{}}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"Paris\\"}"}}\n\n'
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason": "tool_use"},"usage":{"output_tokens":15}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "route.test":
            return json_response({"model": "provider/selected"})
        assert request.url.path == "/v1/messages"
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


def test_anthropic_tool_result_becomes_openai_tool_message_for_decision() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        calls["upstream_path"] = request.url.path

        assert request.url.path == "/v1/messages"
        return json_response(anthropic_response(body["model"], "done"))

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

    # Decision body still gets OpenAI format with tool messages
    decision_messages = calls["decision_body"]["messages"]
    assistant = next(m for m in decision_messages if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    # Assistant content is not polluted with synthetic [tool_use:...] text.
    assert assistant["content"] in (None, "")

    tool_message = next(m for m in decision_messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "toolu_1"
    assert tool_message["content"] == "18C sunny"
    # The tool message immediately follows the assistant tool_calls turn.
    assert decision_messages.index(tool_message) == decision_messages.index(assistant) + 1

    # Upstream now receives ORIGINAL Anthropic format
    assert calls["upstream_path"] == "/v1/messages"
    upstream_messages = calls["upstream_body"]["messages"]
    assert len(upstream_messages) == 3
    user_turn = upstream_messages[2]
    assert user_turn["role"] == "user"
    tool_result_block = next(b for b in user_turn["content"] if b.get("type") == "tool_result")
    assert tool_result_block["tool_use_id"] == "toolu_1"
    # Content can be either a list or already extracted as text depending on anthropic_user_content_to_openai conversion
    content = tool_result_block["content"]
    if isinstance(content, list):
        assert content == [{"type": "text", "text": "18C sunny"}]
    else:
        assert content == "18C sunny"
    # No Python repr or synthetic tool markers leaked anywhere.
    assert "[tool_use:" not in json.dumps(upstream_messages)
    assert "'type':" not in json.dumps(upstream_messages)


def test_cache_control_is_preserved_on_all_surfaces() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        assert request.url.path == "/v1/messages"
        return json_response(anthropic_response(body["model"], "done"))

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

    # Decision body preserves cache_control in OpenAI format
    decision_body = calls["decision_body"]
    system = next(m for m in decision_body["messages"] if m["role"] == "system")
    assert system["content"][1]["cache_control"] == {"type": "ephemeral"}

    # Upstream receives ORIGINAL Anthropic format with cache_control intact
    upstream_body = calls["upstream_body"]
    assert upstream_body["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert upstream_body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert upstream_body["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Tool result content part
    assert upstream_body["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Tool definition
    tools = upstream_body["tools"]
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_control_keeps_plain_string_content() -> None:
    calls: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "route.test":
            calls["decision_body"] = body
            return json_response({"model": "provider/selected"})
        calls["upstream_body"] = body
        assert request.url.path == "/v1/messages"
        return json_response(anthropic_response(body["model"], "done"))

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

    # Decision body collapses to plain strings in OpenAI format
    decision_body = calls["decision_body"]
    system = next(m for m in decision_body["messages"] if m["role"] == "system")
    assert system["content"] == "be concise"
    user = next(m for m in decision_body["messages"] if m["role"] == "user")
    assert user["content"] == "hi"

    # Upstream receives Anthropic format with plain string
    upstream_body = calls["upstream_body"]
    assert upstream_body["system"] == "be concise"
    assert upstream_body["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
