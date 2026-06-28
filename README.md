# modelroute

Route-decision forwarding test tool for OpenAI-compatible chat completions and Anthropic Messages requests.

The service accepts client requests, calls `/v1/route-decision`, replaces the request model with the selected model, and forwards the request to an OpenAI-compatible upstream. It is intended for end-to-end testing of the route-decision API.

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
- `POST /v1/messages`
- `GET /health`

`allowed_models` may be supplied as a top-level JSON field. It is sent to `/v1/route-decision` and removed before the final upstream request.

The final response body remains protocol-compatible. Decision details are exposed through response headers:

- `x-modelroute-selected-model`
- `x-modelroute-decision-tier`
- `x-modelroute-decision-confidence`

## Test

```bash
uv run pytest -v
```
