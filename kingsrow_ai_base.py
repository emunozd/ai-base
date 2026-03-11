"""
kingsrow_ai_base.py — Motor de inferencia IA compartido (Kingsrow Home Lab)
============================================================================
Clase base que:
  - Carga y mantiene el modelo MLX en memoria (singleton).
  - Expone inferencia de texto e imagen con control fino sobre el chat template.
  - Registra routers hijos (uno por proyecto) en la app FastAPI.
  - Expone POST /v1/chat/completions compatible con OpenAI/AnythingLLM (con streaming SSE).
  - Expone GET  /health con info de todos los routers registrados.

Para agregar un proyecto nuevo:
  1. Crea una clase hija de BaseRouter en su propio archivo.
  2. En main.py instanciala y pásala a KingsrowAI.registrar().

Arranque:
    source ~/mlx-env/bin/activate
    python ~/projects/AIBase/main.py
"""

import base64
import io
import json
import logging
import os
import re
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Generator, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config — sobreescribible vía variables de entorno
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("KR_MODEL_PATH", "mlx-community/Qwen3.5-35B-A3B-4bit")
HOST       = os.getenv("KR_HOST",       "192.168.0.90")
PORT       = int(os.getenv("KR_PORT",   "8181"))
IMG_MAX_PX = int(os.getenv("KR_IMG_MAX","1024"))
API_KEY    = os.getenv("KR_API_KEY",    "")   # vacío = sin autenticación


# ─────────────────────────────────────────────────────────────────────────────
# Singleton del modelo
# ─────────────────────────────────────────────────────────────────────────────
class _ModeloMLX:
    """
    Carga el modelo MLX una sola vez y lo mantiene en memoria.
    Todos los routers comparten esta misma instancia — los 19 GB se pagan una vez.
    """
    _model:     Any = None
    _processor: Any = None
    _config:    Any = None

    @classmethod
    def cargar(cls) -> None:
        if cls._model is None:
            logger.info("Cargando modelo MLX: %s", MODEL_PATH)
            cls._model, cls._processor = load(MODEL_PATH)
            cls._config = load_config(MODEL_PATH)
            logger.info("Modelo listo en memoria.")

    @classmethod
    def get(cls) -> tuple[Any, Any, Any]:
        if cls._model is None:
            raise RuntimeError("El modelo no ha sido cargado. Llama a _ModeloMLX.cargar() primero.")
        return cls._model, cls._processor, cls._config


# ─────────────────────────────────────────────────────────────────────────────
# Motor de inferencia compartido
# ─────────────────────────────────────────────────────────────────────────────
class MotorInferencia:
    """
    Wrappers de bajo nivel sobre mlx_vlm.
    Los routers hijos llaman a estos métodos — nunca tocan mlx_vlm directamente.
    """

    @staticmethod
    def texto(prompt_usuario: str, max_tokens: int = 512) -> str:
        model, processor, config = _ModeloMLX.get()
        prompt = apply_chat_template(
            processor, config, prompt_usuario,
            num_images=0,
            enable_thinking=False,      # ← control fino: thinking desactivado
        )
        result = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
        return result.text.strip() if hasattr(result, "text") else str(result).strip()

    @staticmethod
    def imagen(prompt_usuario: str, imagen_b64: str, max_tokens: int = 600) -> str:
        model, processor, config = _ModeloMLX.get()
        img_bytes = base64.b64decode(imagen_b64)
        tmp_path  = os.path.join(tempfile.gettempdir(), f"kr_img_{uuid.uuid4().hex}.jpg")
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((IMG_MAX_PX, IMG_MAX_PX))    # ← control fino: resize antes de enviar
            img.save(tmp_path, "JPEG")
            prompt = apply_chat_template(
                processor, config, prompt_usuario,
                num_images=1,
                enable_thinking=False,
            )
            result = generate(
                model, processor, prompt,
                image=tmp_path,
                max_tokens=max_tokens,
                verbose=False,
            )
            return result.text.strip() if hasattr(result, "text") else str(result).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def extraer_json(texto: str) -> Any:
        """Extrae el primer JSON válido (objeto o lista) del texto del modelo."""
        try:
            return json.loads(texto)
        except json.JSONDecodeError:
            pass
        for patron in (r"\[.*\]", r"\{.*\}"):
            match = re.search(patron, texto, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        raise ValueError(f"No se pudo extraer JSON válido del texto: {texto!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Clase base para routers de proyecto
# ─────────────────────────────────────────────────────────────────────────────
class BaseRouter(ABC):
    """
    Clase base que deben extender los routers de proyecto.

    Cada proyecto:
      - Define su prefix (ej. "/luka", "/proyecto-x").
      - Implementa _registrar_rutas() donde declara sus @router.post / @router.get.
      - Recibe el MotorInferencia inyectado — no instancia nada de MLX por su cuenta.
    """

    prefix: str = "/proyecto"   # sobreescribir en la clase hija

    def __init__(self, motor: MotorInferencia) -> None:
        self.motor  = motor
        self.router = APIRouter(prefix=self.prefix)
        self._registrar_rutas()
        logger.info("Router registrado: %s", self.prefix)

    @abstractmethod
    def _registrar_rutas(self) -> None:
        """Declara aquí todos los endpoints del proyecto con self.router."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Modelos Pydantic para /v1/chat/completions (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
class _Mensaje(BaseModel):
    role:    str
    content: str


class _ChatRequest(BaseModel):
    model:       Optional[str]   = None
    messages:    list[_Mensaje]  = []
    max_tokens:  Optional[int]   = 512
    temperature: Optional[float] = None   # aceptado pero ignorado
    stream:      Optional[bool]  = False


# ─────────────────────────────────────────────────────────────────────────────
# Modelos Pydantic para /v1/messages (Anthropic — Claude Code)
# ─────────────────────────────────────────────────────────────────────────────
class _AnthropicContentBlock(BaseModel):
    type: str
    text: Optional[str] = None

class _AnthropicMensaje(BaseModel):
    role:    str
    content: Any   # str o lista de content blocks

class _AnthropicRequest(BaseModel):
    model:      Optional[str]              = None
    messages:   list[_AnthropicMensaje]    = []
    system:     Optional[Any]              = None
    max_tokens: Optional[int]              = 1024
    stream:     Optional[bool]             = False


# ─────────────────────────────────────────────────────────────────────────────
# Servidor principal
# ─────────────────────────────────────────────────────────────────────────────
class KingsrowAI:
    """
    Orquestador central.
    - Carga el modelo antes de que FastAPI arranque (síncrono, fuera del event loop).
    - Registra routers hijos.
    - Expone /v1/chat/completions compatible con OpenAI / AnythingLLM (streaming SSE).
    - Expone /health global.
    """

    def __init__(self) -> None:
        self._motor:   MotorInferencia   = MotorInferencia()
        self._routers: list[BaseRouter]  = []
        self._app:     Optional[FastAPI] = None

    def registrar(self, router_cls: type[BaseRouter]) -> "KingsrowAI":
        """Registra una clase hija de BaseRouter. Retorna self para encadenado."""
        instancia = router_cls(self._motor)
        self._routers.append(instancia)
        return self

    # ── Construcción de la app ────────────────────────────────────────────────
    def build(self) -> FastAPI:
        # Cargar modelo ANTES de crear la app — síncrono, fuera del event loop
        _ModeloMLX.cargar()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield   # modelo ya cargado, nada que hacer aquí

        app = FastAPI(
            title="Kingsrow AI",
            description="Motor de inferencia MLX compartido — Kingsrow Home Lab",
            version="2.0.0",
            lifespan=lifespan,
        )

        # Middleware de API Key (opcional — se activa si KR_API_KEY está definida)
        if API_KEY:
            from fastapi import Request
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import JSONResponse

            class _APIKeyMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Request, call_next):
                    if request.url.path == "/health":
                        return await call_next(request)
                    key = request.headers.get("X-API-Key", "")
                    if key != API_KEY:
                        return JSONResponse({"detail": "API key inválida o ausente."}, status_code=401)
                    return await call_next(request)

            app.add_middleware(_APIKeyMiddleware)
            logger.info("Autenticación por API Key activada.")

        # Endpoints base
        self._montar_endpoints_base(app)

        # Routers de proyectos
        for r in self._routers:
            app.include_router(r.router)

        self._app = app
        return app

    def run(self) -> None:
        app = self._app or self.build()
        logger.info("Iniciando Kingsrow AI en http://%s:%d", HOST, PORT)
        uvicorn.run(app, host=HOST, port=PORT)

    # ── Endpoints base ────────────────────────────────────────────────────────
    def _montar_endpoints_base(self, app: FastAPI) -> None:
        motor   = self._motor
        routers = self._routers

        # ── /health ──────────────────────────────────────────────────────────
        @app.get("/health")
        def health():
            return {
                "status":  "ok",
                "service": "kingsrow-ai",
                "modelo":  MODEL_PATH,
                "host":    HOST,
                "port":    PORT,
                "routers": [r.prefix for r in routers],
            }

        # ── /v1/models  (requerido por Claude Code al arrancar) ─────────────
        @app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id":         MODEL_PATH,
                        "object":     "model",
                        "created":    0,
                        "owned_by":   "kingsrow",
                    }
                ],
            }

        # ── /v1/messages  (Anthropic-compatible — Claude Code) ──────────────
        @app.post("/v1/messages")
        def anthropic_messages(req: _AnthropicRequest, request: Any = None):
            """
            Endpoint compatible con la API de Anthropic.
            Claude Code apunta aquí vía ANTHROPIC_BASE_URL.
            Soporta stream=true (SSE con eventos Anthropic) y stream=false.
            """
            if not req.messages:
                raise HTTPException(status_code=422, detail="'messages' no puede estar vacío.")

            # Construir prompt combinando system prompt + historial
            partes = []
            if req.system:
                if isinstance(req.system, list):
                    system_texto = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in req.system
                        if not isinstance(b, dict) or b.get("type") == "text"
                    )
                else:
                    system_texto = str(req.system)
                partes.append(f"[SYSTEM]: {system_texto}")
            for msg in req.messages:
                contenido = msg.content
                # content puede ser string o lista de blocks
                if isinstance(contenido, list):
                    texto = " ".join(
                        b.get("text", "") if isinstance(b, dict) else (b.text or "")
                        for b in contenido
                        if (isinstance(b, dict) and b.get("type") == "text")
                        or (hasattr(b, "type") and b.type == "text")
                    )
                else:
                    texto = str(contenido)
                if msg.role == "user":
                    partes.append(f"[USER]: {texto}")
                elif msg.role == "assistant":
                    partes.append(f"[ASSISTANT]: {texto}")

            prompt_completo = "\n".join(partes)
            max_tok         = req.max_tokens or 1024
            if max_tok < 1024: max_tok = 4096
            msg_id          = f"msg_{uuid.uuid4().hex}"
            created         = int(time.time())
            model_name      = req.model or MODEL_PATH

            try:
                respuesta = motor.texto(prompt_completo, max_tokens=max_tok)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            input_tokens  = len(prompt_completo.split())
            output_tokens = len(respuesta.split())

            # ── Streaming SSE formato Anthropic (Claude Code) ─────────────
            if req.stream:
                def _anthropic_sse() -> Generator[str, None, None]:
                    # 1. message_start
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n"
                    # 2. content_block_start
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    # 3. ping
                    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                    # 4. content_block_delta con toda la respuesta
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': respuesta}})}\n\n"
                    # 5. content_block_stop
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    # 6. message_delta
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                    # 7. message_stop
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                return StreamingResponse(_anthropic_sse(), media_type="text/event-stream")

            # ── Respuesta JSON completa formato Anthropic ─────────────────
            return {
                "id":            msg_id,
                "type":          "message",
                "role":          "assistant",
                "content":       [{"type": "text", "text": respuesta}],
                "model":         model_name,
                "stop_reason":   "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens":  input_tokens,
                    "output_tokens": output_tokens,
                },
            }

        # ── /v1/chat/completions  (OpenAI-compatible + SSE streaming) ────────
        @app.post("/v1/chat/completions")
        def chat_completions(req: _ChatRequest):
            """
            Endpoint compatible con OpenAI.
            AnythingLLM, Open WebUI y cualquier cliente OpenAI-compatible
            pueden apuntar a http://192.168.0.90:8181 como proveedor.
            Soporta stream=true (SSE) y stream=false (JSON completo).
            """
            if not req.messages:
                raise HTTPException(status_code=422, detail="'messages' no puede estar vacío.")

            # Construir prompt a partir del historial de mensajes
            partes = []
            for msg in req.messages:
                if msg.role == "system":
                    partes.append(f"[SYSTEM]: {msg.content}")
                elif msg.role == "user":
                    partes.append(f"[USER]: {msg.content}")
                elif msg.role == "assistant":
                    partes.append(f"[ASSISTANT]: {msg.content}")

            prompt_completo = "\n".join(partes)
            max_tok         = req.max_tokens or 512
            cmpl_id         = f"chatcmpl-{uuid.uuid4().hex}"
            created         = int(time.time())
            model_name      = req.model or MODEL_PATH

            try:
                respuesta = motor.texto(prompt_completo, max_tokens=max_tok)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            # ── Streaming SSE (AnythingLLM, Open WebUI, etc.) ─────────────
            if req.stream:
                def _sse_generator() -> Generator[str, None, None]:
                    # Chunk 1: rol del asistente
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                    # Chunk 2: contenido completo
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': respuesta}, 'finish_reason': None}]})}\n\n"
                    # Chunk 3: cierre
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(_sse_generator(), media_type="text/event-stream")

            # ── Respuesta JSON completa (no streaming) ────────────────────
            return {
                "id":      cmpl_id,
                "object":  "chat.completion",
                "created": created,
                "model":   model_name,
                "choices": [
                    {
                        "index":         0,
                        "message":       {"role": "assistant", "content": respuesta},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens":     len(prompt_completo.split()),
                    "completion_tokens": len(respuesta.split()),
                    "total_tokens":      len(prompt_completo.split()) + len(respuesta.split()),
                },
            }

