"""
kingsrow_ai_base.py — Motor de inferencia IA compartido (Kingsrow Home Lab)
============================================================================
Clase base que:
  - Carga y mantiene el modelo MLX en memoria (singleton).
  - Expone inferencia de texto e imagen con control fino sobre el chat template.
  - Registra routers hijos (uno por proyecto) en la app FastAPI.
  - Expone POST /v1/messages  compatible con Anthropic / Claude Code (SSE).
  - Expone POST /v1/chat/completions compatible con OpenAI / AnythingLLM (SSE).
  - Expone GET  /v1/models y GET /health.
  - Búsqueda web via DuckDuckGo cuando el modelo la solicita (tool_use).

Cambios respecto a la versión anterior:
  - chat() usa apply_chat_template con lista de mensajes estructurados (multi-turno real).
  - max_tokens por defecto: 8192 en /v1/messages, 4096 en /v1/chat/completions.
  - Bucle tool_use: si el modelo pide web_search, ejecuta y hace segunda inferencia.
  - system como Any (acepta str o lista de bloques que manda Claude Code).

Para agregar un proyecto nuevo:
  1. Crea una clase hija de BaseRouter en su propio archivo.
  2. En main.py instanciala y pásala a KingsrowAI.registrar().

Arranque:
    source ~/mlx-env/bin/activate
    python ~/projects/AIBase/main.py

Variables de entorno (todas opcionales):
    KR_MODEL_PATH   (default: mlx-community/Qwen3.5-35B-A3B-4bit)
    KR_HOST         (default: 192.168.0.90)
    KR_PORT         (default: 8181)
    KR_IMG_MAX      (default: 1024)
    KR_API_KEY      (default: vacío = sin auth)
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
from mlx_lm import load as lm_load
from mlx_lm import generate as lm_generate
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load as vlm_load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("KR_MODEL_PATH", "mlx-community/Qwen3.5-35B-A3B-4bit")
HOST       = os.getenv("KR_HOST",       "192.168.0.90")
PORT       = int(os.getenv("KR_PORT",   "8181"))
IMG_MAX_PX = int(os.getenv("KR_IMG_MAX","1024"))
API_KEY    = os.getenv("KR_API_KEY",    "")

# Límites de tokens por defecto — ajustables via env
MAX_TOKENS_CHAT    = int(os.getenv("KR_MAX_TOKENS_CHAT",    "8192"))   # /v1/messages (Claude Code)
MAX_TOKENS_OPENAI  = int(os.getenv("KR_MAX_TOKENS_OPENAI",  "4096"))   # /v1/chat/completions (AnythingLLM)
MAX_TOKENS_LUKA    = int(os.getenv("KR_MAX_TOKENS_LUKA",    "600"))    # endpoints LUKA


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda web (DuckDuckGo — sin API key, sin límite)
# ─────────────────────────────────────────────────────────────────────────────
def _web_search(query: str, max_results: int = 5) -> str:
    """
    Busca en DuckDuckGo y devuelve un string con los resultados.
    Se usa como tool cuando el modelo lo solicita.
    Requiere: pip install duckduckgo-search --break-system-packages
    Si no está instalado devuelve un mensaje de error amigable.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Búsqueda web no disponible. Instala: pip install duckduckgo-search --break-system-packages"

    try:
        resultados = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                resultados.append(f"- {r['title']}: {r['body']} ({r['href']})")
        return "\n".join(resultados) if resultados else "Sin resultados para: " + query
    except Exception as e:
        return f"Error en búsqueda web: {e}"


# Definición de la tool web_search que se le pasa al modelo
_WEB_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Busca información actualizada en internet usando DuckDuckGo. "
            "Úsala cuando el usuario pregunte por eventos recientes, noticias, precios, "
            "o cualquier dato que pueda haber cambiado después de tu fecha de entrenamiento."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta de búsqueda en el idioma más apropiado."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de resultados (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Singleton del modelo
# ─────────────────────────────────────────────────────────────────────────────
class _ModeloMLX:
    """
    Carga el modelo MLX una sola vez y lo mantiene en memoria.
    Todos los routers comparten esta instancia — los 19 GB se pagan una vez.
    """
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
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────
def _extraer_texto_content(content: Any) -> str:
    """Normaliza content que puede ser str, lista de bloques dict o lista de objetos."""
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


def _construir_prompt_chat(mensajes: list[dict], system: Any = None) -> str:
    """
    Construye el prompt completo para inferencia de texto multi-turno.
    Usa el formato de chat template de Qwen directamente vía apply_chat_template.
    """
    _, processor, config = _ModeloMLX.get()

    # Armar lista de mensajes en formato estándar {role, content: str}
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
        raise ValueError("No hay mensajes válidos para construir el prompt.")

    # apply_chat_template acepta lista de dicts con role/content
    prompt = apply_chat_template(
        processor,
        config,
        msgs,          # ← lista estructurada, no string — multi-turno real
        num_images=0,
        enable_thinking=False,
    )
    return prompt


def _inferir_chat(mensajes: list[dict], system: Any = None, max_tokens: int = 8192) -> str:
    """
    Inferencia multi-turno con soporte de tool_use (web_search).
    Bucle:
      1. Genera respuesta.
      2. Si el modelo pide web_search, ejecuta y agrega resultado al historial.
      3. Segunda inferencia con el resultado de la búsqueda.
    Máximo 1 vuelta de tool_use para evitar loops infinitos.
    """
    model, processor, _ = _ModeloMLX.get()

    prompt = _construir_prompt_chat(mensajes, system)
    result = vlm_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    respuesta = result.text.strip() if hasattr(result, "text") else str(result).strip()

    # Detectar si el modelo quiere hacer una búsqueda web
    # Qwen puede indicarlo con un bloque JSON de tool_call o con texto explícito
    tool_call = _detectar_tool_call(respuesta)
    if tool_call and tool_call.get("name") == "web_search":
        query       = tool_call.get("arguments", {}).get("query", "")
        max_results = tool_call.get("arguments", {}).get("max_results", 5)
        logger.info("Tool call: web_search(query=%r)", query)

        resultado_busqueda = _web_search(query, max_results=max_results)

        # Agregar el resultado al historial y hacer segunda inferencia
        mensajes_con_tool = list(mensajes) + [
            {"role": "assistant", "content": respuesta},
            {
                "role": "user",
                "content": (
                    f"Resultado de la búsqueda web para '{query}':\n\n"
                    f"{resultado_busqueda}\n\n"
                    "Con esta información, responde la pregunta original de forma completa."
                )
            },
        ]
        prompt2  = _construir_prompt_chat(mensajes_con_tool, system)
        result2  = vlm_generate(model, processor, prompt2, max_tokens=max_tokens, verbose=False)
        respuesta = result2.text.strip() if hasattr(result2, "text") else str(result2).strip()

    return respuesta


def _detectar_tool_call(texto: str) -> Optional[dict]:
    """
    Intenta detectar un tool_call en la respuesta del modelo.
    Qwen puede devolver JSON con {"name": "web_search", "arguments": {...}}
    dentro de bloques <tool_call>...</tool_call> o directamente.
    """
    # Buscar bloque <tool_call>
    match = re.search(r"<tool_call>(.*?)</tool_call>", texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Buscar JSON directo con "name" y "arguments"
    match = re.search(r'\{"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}', texto, re.DOTALL)
    if match:
        try:
            return {"name": match.group(1), "arguments": json.loads(match.group(2))}
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Motor de inferencia compartido
# ─────────────────────────────────────────────────────────────────────────────
class MotorInferencia:
    """
    Wrappers de inferencia que usan los routers hijos.
    - texto()  → inferencia de un solo turno (endpoints LUKA, uso interno)
    - chat()   → inferencia multi-turno con tool_use (Claude Code, AnythingLLM)
    - imagen() → inferencia con imagen adjunta (facturas LUKA)
    - extraer_json() → extrae JSON de la respuesta del modelo
    """

    @staticmethod
    def texto(prompt_usuario: str, max_tokens: int = MAX_TOKENS_LUKA) -> str:
        """Inferencia de un solo turno. Úsalo desde los routers de proyecto (LUKA, etc.)."""
        model, processor, config = _ModeloMLX.get()
        prompt = apply_chat_template(
            processor, config, prompt_usuario,
            num_images=0,
            enable_thinking=False,
        )
        result = vlm_generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
        return result.text.strip() if hasattr(result, "text") else str(result).strip()

    @staticmethod
    def chat(mensajes: list[dict], system: Any = None, max_tokens: int = MAX_TOKENS_CHAT) -> str:
        """
        Inferencia multi-turno con soporte de tool_use (web_search).
        Úsalo desde los endpoints de chat (/v1/messages, /v1/chat/completions).
        mensajes: lista de dicts con {"role": "user"|"assistant", "content": str|list}
        """
        return _inferir_chat(mensajes, system=system, max_tokens=max_tokens)

    @staticmethod
    def imagen(prompt_usuario: str, imagen_b64: str, max_tokens: int = MAX_TOKENS_LUKA) -> str:
        """Inferencia con imagen adjunta. Para facturas de LUKA."""
        model, processor, config = _ModeloMLX.get()
        img_bytes = base64.b64decode(imagen_b64)
        tmp_path  = os.path.join(tempfile.gettempdir(), f"kr_img_{uuid.uuid4().hex}.jpg")
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((IMG_MAX_PX, IMG_MAX_PX))
            img.save(tmp_path, "JPEG")
            prompt = apply_chat_template(
                processor, config, prompt_usuario,
                num_images=1,
                enable_thinking=False,
            )
            result = vlm_generate(
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
    Cada proyecto define su prefix y registra sus rutas en _registrar_rutas().
    Recibe el MotorInferencia inyectado — nunca toca mlx_vlm directamente.
    """

    prefix: str = "/proyecto"

    def __init__(self, motor: MotorInferencia) -> None:
        self.motor  = motor
        self.router = APIRouter(prefix=self.prefix)
        self._registrar_rutas()
        logger.info("Router registrado: %s", self.prefix)

    @abstractmethod
    def _registrar_rutas(self) -> None:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Modelos Pydantic
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI (/v1/chat/completions)
class _OAIMensaje(BaseModel):
    role:    str
    content: Any   # str o lista de bloques

class _OAIRequest(BaseModel):
    model:       Optional[str]   = None
    messages:    list[_OAIMensaje] = []
    max_tokens:  Optional[int]   = None
    temperature: Optional[float] = None
    stream:      Optional[bool]  = False

# Anthropic (/v1/messages)
class _AnthropicMensaje(BaseModel):
    role:    str
    content: Any   # str o lista de bloques

class _AnthropicRequest(BaseModel):
    model:      Optional[str]               = None
    messages:   list[_AnthropicMensaje]     = []
    system:     Optional[Any]               = None   # str o lista de bloques (Claude Code manda lista)
    max_tokens: Optional[int]               = None
    stream:     Optional[bool]              = False


# ─────────────────────────────────────────────────────────────────────────────
# Servidor principal
# ─────────────────────────────────────────────────────────────────────────────
class KingsrowAI:
    """
    Orquestador central.
    - Carga el modelo de forma síncrona antes de que FastAPI arranque.
    - Registra routers hijos vía registrar().
    - Expone /v1/messages, /v1/chat/completions, /v1/models, /health.
    """

    def __init__(self) -> None:
        self._motor:   MotorInferencia   = MotorInferencia()
        self._routers: list[BaseRouter]  = []
        self._app:     Optional[FastAPI] = None

    def registrar(self, router_cls: type[BaseRouter]) -> "KingsrowAI":
        instancia = router_cls(self._motor)
        self._routers.append(instancia)
        return self

    def build(self) -> FastAPI:
        _ModeloMLX.cargar()   # síncrono, fuera del event loop

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        app = FastAPI(
            title="Kingsrow AI",
            description="Motor de inferencia MLX compartido — Kingsrow Home Lab",
            version="3.0.0",
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
                    key = request.headers.get("X-API-Key", "")
                    if key != API_KEY:
                        return JSONResponse({"detail": "API key inválida o ausente."}, status_code=401)
                    return await call_next(request)

            app.add_middleware(_APIKeyMiddleware)
            logger.info("Autenticación por API Key activada.")

        self._montar_endpoints_base(app)
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

        # ── /v1/models ───────────────────────────────────────────────────────
        @app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [{
                    "id":       MODEL_PATH,
                    "object":   "model",
                    "created":  0,
                    "owned_by": "kingsrow",
                }],
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
                "id":            msg_id,
                "type":          "message",
                "role":          "assistant",
                "content":       [{"type": "text", "text": respuesta}],
                "model":         model_name,
                "stop_reason":   "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }

        # ── /v1/chat/completions  (OpenAI — AnythingLLM) ─────────────────────
        @app.post("/v1/chat/completions")
        def chat_completions(req: _OAIRequest):
            if not req.messages:
                raise HTTPException(status_code=422, detail="'messages' no puede estar vacío.")

            # Separar system del resto
            system  = None
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
                "id":      cmpl_id,
                "object":  "chat.completion",
                "created": created,
                "model":   model_name,
                "choices": [{
                    "index":         0,
                    "message":       {"role": "assistant", "content": respuesta},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens":     prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens":      prompt_tokens + comp_tokens,
                },
            }