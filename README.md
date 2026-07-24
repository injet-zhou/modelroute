# modelroute

Route-decision forwarding test tool for OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages requests.

The service accepts client requests, calls `/v1/route-decision`, replaces the request model with the selected model, and forwards the original protocol to a compatible upstream. It is intended for end-to-end testing of the route-decision API.

## Run

```bash
MODELROUTE_ROUTE_BASE_URL=http://127.0.0.1:8403 \
MODELROUTE_UPSTREAM_BASE_URL=http://127.0.0.1:11434 \
uv run python main.py
```

Defaults:

- `MODELROUTE_HOST=127.0.0.1`
- `MODELROUTE_PORT=8500`

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages`
- `GET /health`

`allowed_models` may be supplied as a top-level JSON field. It is sent to `/v1/route-decision` and removed before the final upstream request.

Responses API requests are projected into a valid Chat Completions request only for route analysis. Text, image, file, function-call, tool, token-budget, reasoning, and structured-output fields are converted to their Chat equivalents. Responses-only items and built-in tools use valid synthetic Chat placeholders so their routing capability signals are retained without blocking the request.

The original request fields—including `previous_response_id`, conversation state, reasoning options, tools, and structured-output configuration—are preserved when forwarding to the upstream `/v1/responses` endpoint. Streaming Responses SSE events are forwarded without conversion. Chat Completions and Anthropic Messages requests are likewise forwarded to their protocol-native upstream endpoints after route analysis.

The final response body remains protocol-compatible. Decision details are exposed through response headers:

- `x-modelroute-selected-model`
- `x-modelroute-decision-tier`
- `x-modelroute-decision-confidence`

## Test

```bash
uv run pytest -v
```
