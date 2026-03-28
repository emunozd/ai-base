"""
kalo_ai_router.py — Router de inferencia IA para KALO
======================================================
Clase hija de BaseRouter que define los endpoints de KALO:
  POST /kalo/analizar-foto-comida     — estima calorías desde una imagen
  POST /kalo/interpretar-texto        — interpreta texto libre del usuario
  POST /kalo/sugerencia-nutricional   — consejo según el balance del día

Este archivo NO es el punto de entrada del servidor.
El servidor se levanta desde main.py — ese es el único LaunchDaemon.

Para registrar en main.py:
    from kalo_ai_router import KaloRouter
    KaloRouter(motor, app)
"""

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, KingsrowAI, MotorInferencia


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_BASE = (
    "Eres KALO, el asistente nutricional personal de una app de seguimiento calórico para colombianos. "
    "Responde SIEMPRE en español. NUNCA uses otro idioma. "
    "Sé directo, amigable y motivador. "
    "NUNCA inventes datos que el usuario no haya proporcionado. "
    "SOLO devuelve el JSON solicitado, sin texto adicional, sin backticks, sin markdown."
)

PROMPT_FOTO = """Analiza la imagen de este plato de comida.
Si hay un cubierto (tenedor o cuchillo) úsalo como referencia de tamaño para estimar las porciones.

Devuelve ÚNICAMENTE este JSON:
{{
  "descripcion": "descripción breve del plato y sus componentes",
  "kcal_estimadas": total_calorias_entero,
  "confianza": "ALTA",
  "detalle": "Arroz blanco ~200kcal, pollo ~180kcal, ensalada ~70kcal"
}}

Reglas:
- confianza: ALTA (alimentos claramente visibles), MEDIA (parcialmente visible o ambiguo), BAJA (muy difícil de determinar).
- kcal_estimadas: número entero, calorías totales del plato visible.
- detalle: desglose breve por componente con kcal aproximadas.
- Si no hay cubierto de referencia, asume porción estándar colombiana y baja confianza a MEDIA.
- Si la imagen no es de comida, devuelve kcal_estimadas: 0 y confianza: BAJA."""

PROMPT_TEXTO = """El usuario de una app de seguimiento calórico escribió esto libremente:
"{texto}"

Contexto del usuario hoy:
- Calorías objetivo: {objetivo} kcal
- Calorías consumidas: {consumidas} kcal
- Calorías quemadas por ejercicio: {quemadas} kcal
- Calorías disponibles: {disponibles} kcal

Tu tarea es interpretar qué quiere hacer el usuario y extraer los datos relevantes.

Devuelve ÚNICAMENTE este JSON:
{{
  "intent": "REGISTRAR_COMIDA | REGISTRAR_EJERCICIO | CONSULTAR_BALANCE | PEDIR_SUGERENCIA | OTRO",
  "descripcion": "qué identificaste que quiere hacer",
  "kcal": numero_o_null,
  "alimento": "nombre del alimento si aplica o null",
  "ejercicio": "nombre del ejercicio si aplica o null",
  "duracion_min": numero_o_null,
  "respuesta_directa": "respuesta conversacional al usuario en 1-2 oraciones"
}}"""

PROMPT_SUGERENCIA = """El usuario lleva el siguiente balance calórico hoy:
- Objetivo diario: {objetivo} kcal
- Consumidas: {consumidas} kcal
- Quemadas (ejercicio): {quemadas} kcal
- Disponibles: {disponibles} kcal
- Hora del día: {hora}
- Comidas registradas hoy: {comidas}

Genera una sugerencia nutricional personalizada, práctica y motivadora.

Devuelve ÚNICAMENTE este JSON:
{{
  "mensaje": "sugerencia principal en 2-3 oraciones",
  "opciones": ["opción de comida 1", "opción de comida 2", "opción de comida 3"],
  "advertencia": "texto de advertencia si las calorías están muy desbalanceadas, o null"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request
# ─────────────────────────────────────────────────────────────────────────────

class FotoComidaRequest(BaseModel):
    imagen_b64: str


class TextoLibreRequest(BaseModel):
    texto: str
    objetivo: float = 2000
    consumidas: float = 0
    quemadas: float = 0
    disponibles: float = 2000


class SugerenciaRequest(BaseModel):
    objetivo: float
    consumidas: float
    quemadas: float
    disponibles: float
    hora: str = "12:00"
    comidas: str = "Sin registros aún"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de validación
# ─────────────────────────────────────────────────────────────────────────────

INTENTS_VALIDOS = {
    "REGISTRAR_COMIDA", "REGISTRAR_EJERCICIO",
    "CONSULTAR_BALANCE", "PEDIR_SUGERENCIA", "OTRO",
}

def _validar_foto(data: dict) -> dict:
    try:
        kcal = int(float(data.get("kcal_estimadas", 0)))
    except (TypeError, ValueError):
        raise ValueError("kcal_estimadas debe ser un número.")
    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in {"ALTA", "MEDIA", "BAJA"}:
        confianza = "MEDIA"
    descripcion = str(data.get("descripcion", "")).strip()
    if not descripcion:
        raise ValueError("El modelo no devolvió descripción del plato.")
    return {
        "descripcion":    descripcion,
        "kcal_estimadas": kcal,
        "confianza":      confianza,
        "detalle":        str(data.get("detalle", "")).strip() or None,
    }


def _validar_texto(data: dict) -> dict:
    intent = str(data.get("intent", "OTRO")).upper().strip()
    if intent not in INTENTS_VALIDOS:
        intent = "OTRO"
    return {
        "intent":           intent,
        "descripcion":      str(data.get("descripcion", "")).strip() or None,
        "kcal":             float(data["kcal"]) if data.get("kcal") is not None else None,
        "alimento":         data.get("alimento") or None,
        "ejercicio":        data.get("ejercicio") or None,
        "duracion_min":     int(data["duracion_min"]) if data.get("duracion_min") is not None else None,
        "respuesta_directa": str(data.get("respuesta_directa", "")).strip() or None,
    }


def _validar_sugerencia(data: dict) -> dict:
    mensaje = str(data.get("mensaje", "")).strip()
    if not mensaje:
        raise ValueError("El modelo no devolvió sugerencia.")
    opciones = data.get("opciones", [])
    if not isinstance(opciones, list):
        opciones = []
    return {
        "mensaje":     mensaje,
        "opciones":    [str(o).strip() for o in opciones if o],
        "advertencia": str(data.get("advertencia", "")).strip() or None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Router KALO
# ─────────────────────────────────────────────────────────────────────────────

class KaloRouter(BaseRouter):
    """
    Router de inferencia IA para KALO.
    Todos los endpoints quedan bajo el prefijo /kalo/.

    La kalo-api (Docker) apunta a cada endpoint según la función:
        LLM_VISION_URL=http://<aibase-ip>:8181/kalo/analizar-foto-comida
        LLM_TEXT_URL=http://<aibase-ip>:8181/kalo/interpretar-texto
        LLM_SUGGEST_URL=http://<aibase-ip>:8181/kalo/sugerencia-nutricional
    """

    prefix = "/kalo"

    def _registrar_rutas(self) -> None:

        @self.router.post("/analizar-foto-comida")
        def analizar_foto_comida(req: FotoComidaRequest):
            """Estima las calorías de un plato a partir de una imagen en base64."""
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                data = self.motor.extraer_json(
                    self.motor.imagen(PROMPT_FOTO, req.imagen_b64, max_tokens=400)
                )
                return _validar_foto(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/interpretar-texto")
        def interpretar_texto(req: TextoLibreRequest):
            """
            Interpreta texto libre del usuario en el contexto de su balance calórico.
            Devuelve el intent detectado y datos extraídos para que la API actúe.
            """
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
            try:
                prompt = PROMPT_TEXTO.format(
                    texto=req.texto.strip(),
                    objetivo=req.objetivo,
                    consumidas=req.consumidas,
                    quemadas=req.quemadas,
                    disponibles=req.disponibles,
                )
                data = self.motor.extraer_json(
                    self.motor.texto(prompt, max_tokens=300)
                )
                return _validar_texto(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/sugerencia-nutricional")
        def sugerencia_nutricional(req: SugerenciaRequest):
            """
            Genera una sugerencia nutricional personalizada según el balance del día.
            """
            try:
                prompt = PROMPT_SUGERENCIA.format(
                    objetivo=req.objetivo,
                    consumidas=req.consumidas,
                    quemadas=req.quemadas,
                    disponibles=req.disponibles,
                    hora=req.hora,
                    comidas=req.comidas,
                )
                data = self.motor.extraer_json(
                    self.motor.texto(prompt, max_tokens=400)
                )
                return _validar_sugerencia(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))