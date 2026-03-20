# Kingsrow AI — AIBase

Shared MLX inference engine for the Kingsrow home lab. Loads the model once into memory and exposes it to multiple projects through registrable child routers — no model reload required when adding new projects.

---

## Architecture

```
main.py                  ← single entry point and LaunchDaemon target
kingsrow_ai_base.py      ← shared engine, base endpoints, BaseRouter class
luka_ai_router.py        ← child router: LUKA-specific endpoints
otro_proyecto_router.py  ← (future) same pattern, different prefix
```

One process → one model in RAM → all projects share the inference engine.

---

## Requirements

- Apple Silicon (M1/M2/M3) with macOS
- Python 3.12+
- Virtual environment with MLX (`~/mlx-env` recommended)

```bash
pip install mlx-vlm pydantic-settings fastapi uvicorn httpx pillow beautifulsoup4 duckduckgo-search --break-system-packages
```

---

## Configuration

All variables are optional. Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `KR_MODEL_PATH` | `mlx-community/Qwen3.5-35B-A3B-4bit` | MLX model to load |
| `KR_HOST` | `0.0.0.0` | Server bind address |
| `KR_PORT` | `8181` | Server port |
| `KR_IMG_MAX` | `1024` | Maximum image size in pixels |
| `KR_API_KEY` | *(empty)* | Required API key in `X-API-Key` header. Empty = no auth |
| `KR_MAX_TOKENS_CHAT` | `8192` | Max tokens for `/v1/messages` endpoint |
| `KR_MAX_TOKENS_OPENAI` | `4096` | Max tokens for `/v1/chat/completions` endpoint |
| `KR_MAX_TOKENS_LUKA` | `600` | Max tokens for LUKA endpoints |
| `KR_MAX_CTX_TOKENS` | `20000` | Context limit before truncating history |
| `KR_CTX_COLA_MSGS` | `6` | Recent messages to preserve when truncating |
| `KR_WEB_SEARCH_MAX_RESULTS` | `5` | Maximum results per web search |
| `KR_WEB_FETCH_MAX_CHARS` | `3000` | Maximum characters extracted per URL |

---

## Starting the server

### Manual

```bash
source ~/mlx-env/bin/activate
python ~/projects/AIBase/main.py
```

### As a service (LaunchDaemon)

The server is managed by a macOS LaunchDaemon that starts it automatically at boot.

```bash
# Stop
sudo launchctl stop com.kingsrow.ai

# Start
sudo launchctl start com.kingsrow.ai

# View logs
tail -f /tmp/kingsrow-ai.log
```

---

## Base endpoints

### `GET /health`
Server status, loaded model, and registered routers.

### `GET /v1/models`
Available models list (OpenAI-compatible).

### `POST /v1/messages`
Anthropic-compatible format (Claude Code). Supports `tools`, `tool_choice`, `system`, `stream`.

```json
{
  "model": "optional-name",
  "messages": [{"role": "user", "content": "Hello"}],
  "system": "You are an assistant...",
  "max_tokens": 1024
}
```

### `POST /v1/chat/completions`
OpenAI-compatible format (AnythingLLM, LiteLLM, etc.). Supports `stream`.

```json
{
  "model": "optional-name",
  "messages": [
    {"role": "system", "content": "You are an assistant..."},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 1024
}
```

---

## Automatic web search

The engine includes a lightweight classifier that decides whether a question requires up-to-date data. If so, it searches DuckDuckGo, fetches real page content, and injects the results into the context before final inference.

The classifier **does not** search for:
- Definitions and concepts
- Math and code
- History

The classifier **does** search for:
- Asset prices and exchange rates
- Recent news and events
- Current weather
- Recent statistics and official figures

---

## Adding a new project

1. Create `another_project_router.py` extending `BaseRouter`:

```python
from kingsrow_ai_base import BaseRouter, MotorInferencia

class AnotherProjectRouter(BaseRouter):
    prefix = "/another-project"

    def _registrar_rutas(self) -> None:

        @self.router.post("/my-endpoint")
        def my_endpoint(req: MyRequest):
            result = self.motor.texto("my prompt")
            return {"response": result}
```

2. Register it in `main.py`:

```python
from another_project_router import AnotherProjectRouter

server = KingsrowAI()
server.registrar(LukaRouter)
server.registrar(AnotherProjectRouter)
server.build()
server.run()
```

3. Restart the daemon — the model is already in RAM and does not reload.

---

## LUKA endpoints

Available under the `/luka/` prefix:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/luka/categorizar-factura-texto` | Categorizes a plain-text invoice |
| `POST` | `/luka/categorizar-factura-imagen` | Categorizes an invoice from a base64 image |
| `POST` | `/luka/categorizar-gasto-manual` | Categorizes a natural language expense description |

---

## Important notes

- The model uses ~7 GB of unified RAM (Qwen3.5-35B-A3B-4bit MoE).
- Context is automatically truncated to `KR_MAX_CTX_TOKENS` to prevent Metal GPU crashes.
- The server binds to `0.0.0.0` by default — adjust firewall rules accordingly.
- Docker containers on the same machine must reach the server via the local network IP, not `127.0.0.1`.