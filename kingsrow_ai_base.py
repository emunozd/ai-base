"""
kingsrow_ai_base.py — Motor de inferencia IA compartido (Kingsrow Home Lab)
============================================================================
  - Singleton MLX: el modelo se carga una vez, todos los routers lo comparten.
  - Routers hijos (BaseRouter): un archivo por proyecto, sin tocar este.
  - POST /v1/messages         — Anthropic-compatible (Claude Code)
  - POST /v1/chat/completions — OpenAI-compatible (AnythingLLM)
  - GET  /v1/models, GET /health
  - Búsqueda web automática via DuckDuckGo:
      Antes de la inferencia principal, un clasificador ligero decide con JSON
      {"buscar": true, "query": "..."} si la pregunta requiere datos actualizados.
      Si buscar=true, el resultado se inyecta en el contexto y se hace la
      inferencia final con información real.
      Requiere: pip install duckduckgo-search --break-system-packages

Arranque:
    source ~/mlx-env/bin/activate
    python ~/projects/AIBase/main.py

Variables de entorno (todas opcionales):
    KR_MODEL_PATH            default: mlx-community/Qwen3.5-35B-A3B-4bit
    KR_HOST                  default: 192.168.0.90
    KR_PORT                  default: 8181
    KR_IMG_MAX               default: 1024
    KR_API_KEY               default: vacío = sin auth
    KR_MAX_TOKENS_CHAT       default: 8192
    KR_MAX_TOKENS_OPENAI     default: 4096
    KR_MAX_TOKENS_LUKA       default: 600
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
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = os.getenv("KR_MODEL_PATH",           "mlx-community/Qwen3.5-35B-A3B-4bit")
HOST              = os.getenv("KR_HOST",                  "192.168.0.90")
PORT              = int(os.getenv("KR_PORT",              "8181"))
IMG_MAX_PX        = int(os.getenv("KR_IMG_MAX",           "1024"))
API_KEY           = os.getenv("KR_API_KEY",               "")
MAX_TOKENS_CHAT   = int(os.getenv("KR_MAX_TOKENS_CHAT",   "8192"))
MAX_TOKENS_OPENAI = int(os.getenv("KR_MAX_TOKENS_OPENAI", "4096"))
MAX_TOKENS_LUKA   = int(os.getenv("KR_MAX_TOKENS_LUKA",   "600"))

# El clasificador solo necesita devolver JSON corto
_MAX_TOKENS_CLASIFICADOR = 64


# ─────────────────────────────────────────────────────────────────────────────
# Singleton del modelo
# ─────────────────────────────────────────────────────────────────────────────
class _ModeloMLX:
    _model:     Any = None
    _processor: Any = None
    _config:    Any = None

    @classmethod
    def cargar(cls) -> None:
        if cls._model is None:
            logger.info("Cargando modelo MLX: %s", MODEL_PATH)
            cls._model, cls._processor = vlm_load(MODEL_PATH)
            cls._config = load_config(MODEL_PATH)
            logger.info("Modelo listo en memoria.")

    @classmethod
    def get(cls) -> tuple[Any, Any, Any]:
        if cls._model is None:
            raise RuntimeError("El modelo no ha sido cargado.")
        return cls._model, cls._processor, cls._config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extraer_texto_content(content: Any) -> str:
    """Normaliza content: str, lista de dicts o lista de objetos → str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                partes.append(b.get("text", ""))
            elif hasattr(b, "type") and b.type == "text":
                partes.append(b.text or "")
        return " ".join(partes)
    return str(content)


def _construir_prompt(mensajes: list[dict], system: Any = None) -> str:
    """
    Construye el prompt multi-turno usando processor.apply_chat_template()
    directamente — no mlx_vlm.prompt_utils que descarta kwargs adicionales.
    """
    _, processor, _ = _ModeloMLX.get()

    msgs = []
    if system:
        system_texto = _extraer_texto_content(system)
        if system_texto.strip():
            msgs.append({"role": "system", "content": system_texto})

    for m in mensajes:
        role    = m.get("role", "user")
        content = _extraer_texto_content(m.get("content", ""))
        if role in ("user", "assistant", "system") and content.strip():
            msgs.append({"role": role, "content": content})

    if not msgs:
        raise ValueError("No hay mensajes válidos.")

    return processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda web — DuckDuckGo, sin API key
# pip install duckduckgo-search --break-system-packages
# ─────────────────────────────────────────────────────────────────────────────
def _web_search(query: str, max_results: int = 5) -> Optional[str]:
    """
    Devuelve los resultados como string, o None si no hay internet o falla la búsqueda.
    Nunca lanza excepción — el caller decide qué hacer con None.
    Requiere: pip install ddgs --break-system-packages
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs no instalado. Ejecuta: pip install ddgs --break-system-packages")
        return None
    try:
        resultados = []
        with DDGS() as client:
            # region="wt-wt" = worldwide, sin filtro geográfico — mejores resultados en inglés
            for r in client.text(query, max_results=max_results, region="wt-wt"):
                resultados.append(f"- {r['title']}: {r['body']} ({r['href']})")
        return "\n".join(resultados) if resultados else None
    except Exception as e:
        logger.warning("Búsqueda web falló (sin internet?): %s", e)
        return None


def _clasificar_busqueda(pregunta: str) -> Optional[str]:
    """
    Inferencia ligera (64 tokens) sobre la última pregunta del usuario.
    El modelo responde SOLO con JSON: {"buscar": true, "query": "..."} o {"buscar": false}.
    Devuelve la query si se necesita búsqueda, None si no.

    Por qué funciona: el modelo es muy confiable siguiendo instrucciones de
    JSON simple en un solo turno — mucho más que emitir tool_calls espontáneos.
    """
    model, processor, _ = _ModeloMLX.get()

    system_prompt = (
        'Eres un clasificador. Responde ÚNICAMENTE con JSON válido, sin texto adicional, '
        'sin explicaciones, sin markdown. '
        'Formato exacto: {"buscar": true, "query": "..."} o {"buscar": false}. '
        'Responde buscar=true solo si la pregunta requiere información actualizada '
        'que no está en tu entrenamiento: precios en tiempo real, noticias recientes, '
        'eventos actuales, datos posteriores a 2024. '
        'Responde buscar=false para conocimiento general, matemáticas, código, historia, '
        'conceptos, o cualquier cosa que no dependa de datos recientes. '
        'IMPORTANTE: el campo "query" debe estar siempre en inglés para obtener mejores resultados de búsqueda.'
    )

    prompt = processor.apply_chat_template(
        [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": pregunta},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    result    = vlm_generate(model, processor, prompt, max_tokens=_MAX_TOKENS_CLASIFICADOR, verbose=False)
    respuesta = result.text.strip() if hasattr(result, "text") else str(result).strip()

    # Limpiar posibles artefactos de markdown
    respuesta = re.sub(r"```json|```", "", respuesta).strip()

    try:
        datos = json.loads(respuesta)
        if datos.get("buscar") is True:
            query = datos.get("query", "").strip()
            if query:
                logger.info("Clasificador: búsqueda necesaria → %r", query)
                return query
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Clasificador no devolvió JSON válido: %r", respuesta[:100])

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia de chat con búsqueda web automática
# ─────────────────────────────────────────────────────────────────────────────
def _inferir_chat(mensajes: list[dict], system: Any = None, max_tokens: int = MAX_TOKENS_CHAT) -> str:
    """
    Flujo:
      1. Clasificador ligero sobre la última pregunta del usuario.
      2a. buscar=true  → _web_search → inferencia final con resultados en contexto.
      2b. buscar=false → inferencia directa sin overhead.
    """
    model, processor, _ = _ModeloMLX.get()

    # Extraer la última pregunta del usuario para el clasificador
    ultima_pregunta = ""
    for m in reversed(mensajes):
        if m.get("role") == "user":
            ultima_pregunta = _extraer_texto_content(m.get("content", ""))
            break

    query_busqueda = _clasificar_busqueda(ultima_pregunta) if ultima_pregunta else None

    if query_busqueda:
        resultado_busqueda = _web_search(query_busqueda)
        if resultado_busqueda:
            logger.info("Búsqueda completada (%d chars)", len(resultado_busqueda))
            # Inyectar en system prompt — el modelo lo trata como verdad autoritativa
            # y no como conversación que puede ignorar.
            system_texto = _extraer_texto_content(system) if system else ""
            system_con_contexto = (
                (system_texto.strip() + "\n\n") if system_texto.strip() else ""
            ) + (
                f"CONTEXTO ACTUALIZADO DE INTERNET (usa esta información para responder, "
                f"es más confiable que tu conocimiento de entrenamiento):\n"
                f"Búsqueda realizada: '{query_busqueda}'\n\n"
                f"{resultado_busqueda}"
            )
            prompt = _construir_prompt(mensajes, system=system_con_contexto)
        else:
            logger.warning("Búsqueda falló o sin resultados, respondiendo con conocimiento propio.")
            prompt = _construir_prompt(mensajes, system)
    else:
        prompt = _construir_prompt(mensajes, system)

    result = vlm_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    return result.text.strip() if hasattr(result, "text") else str(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Motor de inferencia compartido — API pública para routers hijos
# ─────────────────────────────────────────────────────────────────────────────
class MotorInferencia:
    """
    - texto()        → un solo turno sin búsqueda web. Para routers de proyecto (LUKA).
    - chat()         → multi-turno con búsqueda web automática. Para chatbot y Claude Code.
    - imagen()       → inferencia con imagen. Para facturas de LUKA.
    - extraer_json() → extrae el primer JSON válido de la respuesta del modelo.
    """

    @staticmethod
    def texto(prompt_usuario: str, max_tokens: int = MAX_TOKENS_LUKA) -> str:
        model, processor, config = _ModeloMLX.get()
        prompt = vlm_apply_chat_template(
            processor, config, prompt_usuario,
            num_images=0,
            enable_thinking=False,
        )
        result = vlm_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
        return result.text.strip() if hasattr(result, "text") else str(result).strip()

    @staticmethod
    def chat(mensajes: list[dict], system: Any = None, max_tokens: int = MAX_TOKENS_CHAT) -> str:
        return _inferir_chat(mensajes, system=system, max_tokens=max_tokens)

    @staticmethod
    def imagen(prompt_usuario: str, imagen_b64: str, max_tokens: int = MAX_TOKENS_LUKA) -> str:
        model, processor, config = _ModeloMLX.get()
        img_bytes = base64.b64decode(imagen_b64)
        tmp_path  = os.path.join(tempfile.gettempdir(), f"kr_img_{uuid.uuid4().hex}.jpg")
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((IMG_MAX_PX, IMG_MAX_PX))
            img.save(tmp_path, "JPEG")
            prompt = vlm_apply_chat_template(
                processor, config, prompt_usuario,
                num_images=1,
                enable_thinking=False,
            )
            result = vlm_generate(model, processor, prompt, image=tmp_path, max_tokens=max_tokens, verbose=False)
            return result.text.strip() if hasattr(result, "text") else str(result).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def extraer_json(texto: str) -> Any:
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
        raise ValueError(f"No se pudo extraer JSON válido: {texto!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Clase base para routers de proyecto
# ─────────────────────────────────────────────────────────────────────────────
class BaseRouter(ABC):
    prefix: str = "/proyecto"

    def __init__(self, motor: MotorInferencia) -> None:
        self.motor  = motor
        self.router = APIRouter(prefix=self.prefix)
        self._registrar_rutas()
        logger.info("Router registrado: %s", self.prefix)

    @abstractmethod
    def _registrar_rutas(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Modelos Pydantic
# ─────────────────────────────────────────────────────────────────────────────
class _OAIMensaje(BaseModel):
    role:    str
    content: Any

class _OAIRequest(BaseModel):
    model:       Optional[str]     = None
    messages:    list[_OAIMensaje] = []
    max_tokens:  Optional[int]     = None
    temperature: Optional[float]   = None
    stream:      Optional[bool]    = False

class _AnthropicMensaje(BaseModel):
    role:    str
    content: Any

class _AnthropicRequest(BaseModel):
    model:      Optional[str]           = None
    messages:   list[_AnthropicMensaje] = []
    system:     Optional[Any]           = None
    max_tokens: Optional[int]           = None
    stream:     Optional[bool]          = False


# ─────────────────────────────────────────────────────────────────────────────
# Servidor principal
# ─────────────────────────────────────────────────────────────────────────────
class KingsrowAI:

    def __init__(self) -> None:
        self._motor:   MotorInferencia  = MotorInferencia()
        self._routers: list[BaseRouter] = []
        self._app:     Optional[FastAPI]= None

    def registrar(self, router_cls: type[BaseRouter]) -> "KingsrowAI":
        instancia = router_cls(self._motor)
        self._routers.append(instancia)
        return self

    def build(self) -> FastAPI:
        _ModeloMLX.cargar()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        app = FastAPI(
            title="Kingsrow AI",
            description="Motor de inferencia MLX compartido — Kingsrow Home Lab",
            version="3.2.0",
            lifespan=lifespan,
        )

        if API_KEY:
            from fastapi import Request
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import JSONResponse

            class _APIKeyMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Request, call_next):
                    if request.url.path in ("/health", "/v1/models"):
                        return await call_next(request)
                    if request.headers.get("X-API-Key", "") != API_KEY:
                        return JSONResponse({"detail": "API key inválida o ausente."}, status_code=401)
                    return await call_next(request)

            app.add_middleware(_APIKeyMiddleware)

        self._montar_endpoints_base(app)
        for r in self._routers:
            app.include_router(r.router)

        self._app = app
        return app

    def run(self) -> None:
        app = self._app or self.build()
        logger.info("Iniciando Kingsrow AI en http://%s:%d", HOST, PORT)
        uvicorn.run(app, host=HOST, port=PORT)

    def _montar_endpoints_base(self, app: FastAPI) -> None:
        motor   = self._motor
        routers = self._routers

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

        @app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [{"id": MODEL_PATH, "object": "model", "created": 0, "owned_by": "kingsrow"}],
            }

        # ── /v1/messages  (Anthropic — Claude Code) ──────────────────────────
        @app.post("/v1/messages")
        def anthropic_messages(req: _AnthropicRequest):
            if not req.messages:
                raise HTTPException(status_code=422, detail="'messages' no puede estar vacío.")

            mensajes   = [{"role": m.role, "content": m.content} for m in req.messages]
            max_tok    = req.max_tokens or MAX_TOKENS_CHAT
            msg_id     = f"msg_{uuid.uuid4().hex}"
            created    = int(time.time())
            model_name = req.model or MODEL_PATH

            try:
                respuesta = motor.chat(mensajes, system=req.system, max_tokens=max_tok)
            except Exception as e:
                logger.exception("Error en /v1/messages")
                raise HTTPException(status_code=500, detail=str(e))

            input_tokens  = sum(len(_extraer_texto_content(m.content).split()) for m in req.messages)
            output_tokens = len(respuesta.split())

            if req.stream:
                def _sse() -> Generator[str, None, None]:
                    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': respuesta}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                return StreamingResponse(_sse(), media_type="text/event-stream")

            return {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": respuesta}],
                "model": model_name, "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }

        # ── /v1/chat/completions  (OpenAI — AnythingLLM) ─────────────────────
        @app.post("/v1/chat/completions")
        def chat_completions(req: _OAIRequest):
            if not req.messages:
                raise HTTPException(status_code=422, detail="'messages' no puede estar vacío.")

            system   = None
            mensajes = []
            for m in req.messages:
                if m.role == "system":
                    system = m.content
                else:
                    mensajes.append({"role": m.role, "content": m.content})

            max_tok    = req.max_tokens or MAX_TOKENS_OPENAI
            cmpl_id    = f"chatcmpl-{uuid.uuid4().hex}"
            created    = int(time.time())
            model_name = req.model or MODEL_PATH

            try:
                respuesta = motor.chat(mensajes, system=system, max_tokens=max_tok)
            except Exception as e:
                logger.exception("Error en /v1/chat/completions")
                raise HTTPException(status_code=500, detail=str(e))

            prompt_tokens = sum(len(_extraer_texto_content(m.content).split()) for m in req.messages)
            comp_tokens   = len(respuesta.split())

            if req.stream:
                def _sse() -> Generator[str, None, None]:
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': respuesta}, 'finish_reason': None}]})}\n\n"
                    yield f"data: {json.dumps({'id': cmpl_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(_sse(), media_type="text/event-stream")

            return {
                "id": cmpl_id, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": respuesta}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens, "total_tokens": prompt_tokens + comp_tokens},
            }