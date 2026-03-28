"""
kalo_ai_router.py — Router de inferencia IA para KALO
======================================================
Clase hija de BaseRouter que define el endpoint de KALO:
  POST /kalo/analizar-foto-comida

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
# Prompt del sistema
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Eres un nutricionista experto en análisis calórico de alimentos. "
    "Tu única función es estimar las calorías de platos de comida a partir de imágenes. "
    "Responde SIEMPRE en español. NUNCA uses otro idioma. "
    "NUNCA expliques tu razonamiento. SOLO devuelve el JSON solicitado, sin texto adicional, "
    "sin backticks, sin markdown."
)

PROMPT_FOTO = """Analiza la imagen de este plato de comida.
Si hay un cubierto (tenedor o cuchillo) úsalo como referencia de tamaño para estimar las porciones.

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
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


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de request
# ─────────────────────────────────────────────────────────────────────────────
class FotoComidaRequest(BaseModel):
    imagen_b64: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper de validación
# ─────────────────────────────────────────────────────────────────────────────
def _validar_analisis(data: dict) -> dict:
    confianzas_validas = {"ALTA", "MEDIA", "BAJA"}

    try:
        kcal = int(float(data.get("kcal_estimadas", 0)))
    except (TypeError, ValueError):
        raise ValueError("kcal_estimadas debe ser un número.")

    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in confianzas_validas:
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


# ─────────────────────────────────────────────────────────────────────────────
# Router KALO
# ─────────────────────────────────────────────────────────────────────────────
class KaloRouter(BaseRouter):
    """
    Router de inferencia IA para KALO.
    Todos los endpoints quedan bajo el prefijo /kalo/.

    La kalo-api (Docker) debe apuntar a:
        LLM_VISION_URL=http://<aibase-ip>:8181/kalo/v1/chat/completions

    El endpoint es compatible con el formato OpenAI /v1/chat/completions
    que espera vision_client.py — AIBase lo expone bajo /kalo/.
    """

    prefix = "/kalo"

    def _registrar_rutas(self) -> None:

        @self.router.post("/analizar-foto-comida")
        def analizar_foto_comida(req: FotoComidaRequest):
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                data = self.motor.extraer_json(
                    self.motor.imagen(PROMPT_FOTO, req.imagen_b64, max_tokens=400)
                )
                return _validar_analisis(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
