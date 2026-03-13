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
def _extraer_texto_content(content):
    """
    Normaliza content → str. Maneja todos los tipos de bloques de Claude Code:
    text, tool_use, tool_result. Imágenes descartadas.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for b in content:
            t = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
            if t == "text":
                val = b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                partes.append(val or "")
            elif t == "tool_use":
                name = b.get("name", "")  if isinstance(b, dict) else getattr(b, "name", "")
                inp  = b.get("input", {}) if isinstance(b, dict) else getattr(b, "input", {})
                tid  = b.get("id", "")    if isinstance(b, dict) else getattr(b, "id", "")
                partes.append("[tool_use id=" + tid + " name=" + name + " input=" + json.dumps(inp, ensure_ascii=False) + "]")
            elif t == "tool_result":
                tid  = b.get("tool_use_id", "") if isinstance(b, dict) else getattr(b, "tool_use_id", "")
                cont = b.get("content", "")     if isinstance(b, dict) else getattr(b, "content", "")
                if isinstance(cont, list):
                    cont = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in cont)
                partes.append("[tool_result id=" + tid + "]\n" + str(cont) + "\n[/tool_result]")
        return "\n".join(partes)
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
def _parsear_tool_calls(texto: str):
    """
    Qwen emite tool calls como texto plano con etiquetas XML.
    Parsea y devuelve (tool_blocks, texto_limpio).
    Soporta: <tool_call>{json}</tool_call>
    """
    import re as _re
    tool_blocks  = []
    texto_limpio = texto

    patron = _re.compile(r"<tool_call>(.*?)</tool_call>", _re.DOTALL)
    for m in patron.finditer(texto):
        try:
            datos = json.loads(m.group(1).strip())
            tool_blocks.append({
                "type":  "tool_use",
                "id":    "toolu_" + uuid.uuid4().hex[:16],
                "name":  datos.get("name", "unknown"),
                "input": datos.get("input", datos.get("parameters", datos.get("arguments", {}))),
            })
        except Exception:
            pass
    texto_limpio = patron.sub("", texto_limpio).strip()
    return tool_blocks, texto_limpio


def _url_es_antigua(url: str) -> bool:
    """Detecta URLs con años anteriores a 2025 en el path."""
    match = re.search(r'/(20\d{2})/', url)
    if match:
        return int(match.group(1)) < 2025
    return False


def _fetch_url(url: str, max_chars: int = 3000) -> Optional[str]:
    """
    Descarga el contenido real de una URL y lo limpia.
    Rechaza: respuestas 4xx/5xx, URLs con años anteriores a 2025.
    Devuelve texto plano truncado o None si falla.
    """
    if _url_es_antigua(url):
        # logger.warning("fetch_url omitida (URL antigua): %s", url)
        return None
    try:
        import httpx
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        r = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
        if r.status_code >= 400:
            # logger.warning("fetch_url rechazada HTTP %d: %s", r.status_code, url)
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        texto = " ".join(soup.get_text(separator=" ").split())
        return texto[:max_chars] if texto.strip() else None
    except Exception as e:
        logger.warning("fetch_url falló (%s): %s", url, e)
        return None


def _web_search(query: str, max_results: int = 5) -> Optional[str]:
    """
    1. Busca con ddgs para obtener URLs relevantes.
    2. Hace fetch del contenido real de la primera URL que responda.
    Si el fetch falla para todas, usa los snippets como fallback.
    Requiere: pip install ddgs httpx beautifulsoup4 --break-system-packages
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs no instalado. Ejecuta: pip install ddgs --break-system-packages")
        return None
    try:
        urls = []
        with DDGS() as client:
            for r in client.text(query, max_results=max_results, region="wt-wt"):
                urls.append((r["title"], r["href"], r["body"]))

        if not urls:
            return None

        # Fetch del contenido real — primera URL que responda
        for titulo, url, snippet in urls[:3]:
            contenido = _fetch_url(url)
            if contenido:
                # logger.info("Fetch exitoso: %s", url)
                return f"Fuente: {titulo} ({url})\n\n{contenido}"

        # Fallback a snippets si el fetch falla para todas
        logger.warning("Fetch falló para todas las URLs, usando snippets.")
        return "\n".join(f"- {t}: {s} ({u})" for t, u, s in urls)

    except Exception as e:
        logger.warning("Búsqueda web falló (sin internet?): %s", e)
        return None


def _clasificar_busqueda(pregunta: str) -> list[str]:
    """
    Inferencia ligera sobre la última pregunta del usuario.
    Devuelve lista de queries a buscar (una por tema), o [] si no hace falta.
    Nunca incluye queries para fecha/hora — eso lo provee el servidor.
    Queries siempre en inglés.
    """
    model, processor, _ = _ModeloMLX.get()

    system_prompt = (
        'Eres un clasificador de búsquedas web. Responde ÚNICAMENTE con JSON válido, '
        'sin texto adicional, sin explicaciones, sin markdown. '
        'Formato: {"queries": ["query1", "query2"]} o {"queries": []}. '
        'SOLO genera queries para datos que cambian frecuentemente y no están en tu entrenamiento: '
        'precios de activos, clima actual, noticias recientes, resultados deportivos, '
        'tasas, estadísticas o cifras oficiales recientes. '
        'NUNCA busques: definiciones, conceptos, explicaciones, historia, matemáticas, código. '
        'Todas las queries en inglés. '
        'El año actual es 2026. Inclúyelo siempre en queries de datos anuales '
        '(tasas, salarios, estadísticas, precios históricos).'
    )

    prompt = processor.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": pregunta},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    result    = vlm_generate(model, processor, prompt, max_tokens=128, verbose=False)
    respuesta = result.text.strip() if hasattr(result, "text") else str(result).strip()
    respuesta = re.sub(r"```json|```", "", respuesta).strip()

    try:
        datos   = json.loads(respuesta)
        queries = [q.strip() for q in datos.get("queries", []) if q.strip()]
        # logger.info("Clasificador: queries → %s", queries)
        return queries
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Clasificador no devolvió JSON válido: %r", respuesta[:100])
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia de chat con búsqueda web automática
# ─────────────────────────────────────────────────────────────────────────────
def _inferir_chat(mensajes: list[dict], system: Any = None, max_tokens: int = MAX_TOKENS_CHAT) -> str:
    """
    Flujo:
      1. Fecha/hora actual siempre inyectada desde el servidor — sin búsqueda.
      2. Clasificador detecta qué temas requieren búsqueda (0, 1 o N queries independientes).
      3. Una búsqueda por query.
      4. Todo el contexto va al system prompt antes de la inferencia final.
    """
    from datetime import datetime
    model, processor, _ = _ModeloMLX.get()

    # Fecha siempre disponible — el servidor la tiene, no hace falta buscarla
    fecha_actual = datetime.now().strftime("%A %d de %B de %Y, %H:%M")

    # Extraer la última pregunta del usuario — solo si es texto plano, no tool_result
    ultima_pregunta = ""
    for m in reversed(mensajes):
        if m.get("role") == "user":
            content_raw = m.get("content", "")
            # Si el último mensaje de usuario contiene tool_result, es output de herramienta
            # no una pregunta humana — no clasificar para búsqueda
            es_tool_result = (
                isinstance(content_raw, list) and
                any((b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "tool_result"
                    for b in content_raw)
            )
            if not es_tool_result:
                ultima_pregunta = _extraer_texto_content(content_raw)
            break

    # Clasificar qué temas requieren búsqueda web
    queries = _clasificar_busqueda(ultima_pregunta) if ultima_pregunta else []

    # Ejecutar búsquedas y acumular contexto
    bloques_web = []
    for query in queries:
        resultado = _web_search(query)
        if resultado:
            # logger.info("Búsqueda OK para %r (%d chars)", query, len(resultado))
            bloques_web.append(f"Búsqueda: '{query}'\n{resultado}")
        # else: sin resultados para query

    # Armar system prompt: system original + fecha + datos web
    system_texto = _extraer_texto_content(system) if system else ""
    partes = []
    if system_texto.strip():
        partes.append(system_texto.strip())
    partes.append(f"Fecha y hora actual: {fecha_actual}.")
    if bloques_web:
        partes.append(
            "INSTRUCCIÓN CRÍTICA: Los siguientes datos fueron obtenidos ahora mismo de internet. "
            "DEBES usarlos para responder. PROHIBIDO decir que no tienes acceso a internet.\n\n"
            "=== DATOS DE INTERNET ===\n" +
            "\n\n---\n\n".join(bloques_web) +
            "\n=== FIN DE DATOS ==="
        )

    prompt = _construir_prompt(mensajes, system="\n\n".join(partes))
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
    tools:      Optional[list[Any]]     = None
    tool_choice: Optional[Any]          = None


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
            msg_id     = "msg_" + uuid.uuid4().hex
            created    = int(time.time())
            model_name = req.model or MODEL_PATH

            # Inyectar definición de herramientas en el system prompt
            # para que Qwen sepa qué tools puede invocar y en qué formato
            system_con_tools = req.system
            if req.tools:
                tools_txt = json.dumps(req.tools, ensure_ascii=False, indent=2)
                tools_block = (
                    "Tienes acceso a las siguientes herramientas. "
                    "Cuando necesites usar una, emite EXACTAMENTE este formato y nada más:\n"
                    "<tool_call>{\"name\": \"<nombre>\", \"input\": {<parametros>}}</tool_call>\n\n"
                    "Herramientas disponibles:\n" + tools_txt
                )
                if isinstance(system_con_tools, str) and system_con_tools.strip():
                    system_con_tools = system_con_tools + "\n\n" + tools_block
                elif isinstance(system_con_tools, list):
                    system_con_tools = list(system_con_tools) + [{"type": "text", "text": tools_block}]
                else:
                    system_con_tools = tools_block

            try:
                respuesta = motor.chat(mensajes, system=system_con_tools, max_tokens=max_tok)
            except Exception as e:
                logger.exception("Error en /v1/messages")
                raise HTTPException(status_code=500, detail=str(e))

            input_tokens  = sum(len(_extraer_texto_content(m.content).split()) for m in req.messages)
            output_tokens = len(respuesta.split())

            # Detectar si el modelo emitió tool_use en texto plano y parsearlo
            # Claude Code espera stop_reason="tool_use" + content block tipo tool_use
            tool_blocks, texto_limpio = _parsear_tool_calls(respuesta)
            tiene_tools = len(tool_blocks) > 0
            stop_reason = "tool_use" if tiene_tools else "end_turn"

            # Armar content blocks
            content_blocks = []
            if texto_limpio.strip():
                content_blocks.append({"type": "text", "text": texto_limpio})
            content_blocks.extend(tool_blocks)

            if req.stream:
                def _sse() -> Generator[str, None, None]:
                    yield "event: message_start\ndata: " + json.dumps({
                        "type": "message_start",
                        "message": {"id": msg_id, "type": "message", "role": "assistant",
                                    "content": [], "model": model_name,
                                    "stop_reason": None, "stop_sequence": None,
                                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
                    }) + "\n\n"
                    yield "event: ping\ndata: " + json.dumps({"type": "ping"}) + "\n\n"
                    for i, block in enumerate(content_blocks):
                        yield "event: content_block_start\ndata: " + json.dumps({
                            "type": "content_block_start", "index": i, "content_block": block
                        }) + "\n\n"
                        if block["type"] == "text":
                            yield "event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta", "index": i,
                                "delta": {"type": "text_delta", "text": block["text"]}
                            }) + "\n\n"
                        elif block["type"] == "tool_use":
                            yield "event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta", "index": i,
                                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}))}
                            }) + "\n\n"
                        yield "event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop", "index": i
                        }) + "\n\n"
                    yield "event: message_delta\ndata: " + json.dumps({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens}
                    }) + "\n\n"
                    yield "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n"
                return StreamingResponse(_sse(), media_type="text/event-stream")

            return {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": content_blocks,
                "model": model_name, "stop_reason": stop_reason, "stop_sequence": None,
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