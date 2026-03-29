"""
kalo_ai_router.py — Router de inferencia IA para KALO
======================================================
Endpoints:
  POST /kalo/clasificar-intent       — detecta si es comida, ejercicio u otro
  POST /kalo/inferir-comida          — infiere kcal de texto libre de comida
  POST /kalo/inferir-ejercicio       — infiere kcal de texto libre de ejercicio
  POST /kalo/analizar-foto-comida    — estima kcal desde foto (plato o tabla nutricional)
  POST /kalo/sugerencia-nutricional  — consejo según balance del día

Flujo de texto libre en el bot:
  texto → /clasificar-intent → COMIDA → /inferir-comida → confirmación → guardar
                              → EJERCICIO → /inferir-ejercicio → confirmación → guardar
                              → CONSULTA → mostrar resumen
                              → OTRO → respuesta genérica

Flujo de foto:
  imagen → /analizar-foto-comida → detecta PLATO o TABLA_NUTRICIONAL
         → si TABLA: pide porciones consumidas → calcula kcal → confirmación → guardar
         → si PLATO: muestra estimado → confirmación → guardar
"""

from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, MotorInferencia


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Clasificador de intent ─────────────────────────────────────────────────
PROMPT_INTENT = """El usuario de una app de seguimiento calórico escribió:
"{texto}"

Clasifica de qué se trata. Devuelve ÚNICAMENTE este JSON:
{{
  "intent": "COMIDA | EJERCICIO | CONSULTA | OTRO",
  "confianza": "ALTA | MEDIA | BAJA"
}}

Reglas:
- COMIDA: menciona algo que comió, bebió, o quiere registrar como alimento (incluso bebidas, suplementos, batidos, yogur, kéfir, proteína).
- EJERCICIO: menciona actividad física, distancia, pasos, tiempo de entrenamiento, calorías quemadas.
- CONSULTA: pregunta por su balance, progreso, cuánto le queda, sugerencias.
- OTRO: cualquier cosa fuera de contexto nutricional/fitness."""


# ── 2. Inferir calorías de comida ─────────────────────────────────────────────
PROMPT_INFERIR_COMIDA = """El usuario describió lo que comió o bebió:
"{texto}"

Eres un nutricionista experto en alimentación colombiana y latinoamericana.
Infiere las calorías aproximadas basándote en:
- Las porciones mencionadas (medio pocillo ≈ 125ml, pocillo ≈ 250ml, cucharada ≈ 15g, taza ≈ 240ml)
- Base de datos nutricional estándar
- Si es mezcla de ingredientes, suma cada componente por separado

Devuelve ÚNICAMENTE este JSON:
{{
  "descripcion": "descripción clara del alimento o preparación",
  "kcal": numero_entero_aproximado,
  "detalle": "Ingrediente 1 ~Xkcal, Ingrediente 2 ~Ykcal, ...",
  "confianza": "ALTA | MEDIA | BAJA",
  "nota": "aclaración si la porción fue ambigua, o null"
}}

Reglas CRÍTICAS:
- NUNCA devuelvas 0 kcal a menos que sea agua pura o algo sin calorías comprobado.
- Yogur kéfir entero 125ml ≈ 75-90 kcal.
- Batido proteína (1 scoop ≈ 30g) ≈ 120 kcal, crema de maní 1 cucharada ≈ 95 kcal.
- Si la porción es ambigua, usa una porción estándar colombiana y baja confianza a MEDIA.
- Sé conservador pero nunca cero en alimentos que claramente tienen calorías."""


# ── 3. Inferir calorías de ejercicio ─────────────────────────────────────────
PROMPT_INFERIR_EJERCICIO = """El usuario describió una actividad física:
"{texto}"

Datos del usuario:
- Peso: {peso_kg} kg
- Edad: {edad} años

Infiere las calorías quemadas. Si el usuario ya dio las kcal, úsalas directamente.
Si dio distancia, tiempo o pasos, calcula usando MET estándar y el peso del usuario.

Referencias MET (kcal = MET × peso_kg × horas):
- Caminar normal: MET 3.5 | 10.000 pasos ≈ 7-8 km
- Correr 8-10 km/h: MET 8.0
- Correr >10 km/h: MET 10.0
- Pesas moderado: MET 4.0
- Pesas intenso: MET 6.0
- Bicicleta moderada: MET 6.0
- Yoga/stretching: MET 2.5
- HIIT: MET 8.0

Devuelve ÚNICAMENTE este JSON:
{{
  "descripcion": "descripción clara del ejercicio",
  "kcal_quemadas": numero_entero_aproximado,
  "duracion_min": numero_entero_o_null,
  "distancia_km": numero_decimal_o_null,
  "confianza": "ALTA | MEDIA | BAJA",
  "nota": "supuestos usados para el cálculo, o null"
}}

Reglas:
- Si el usuario dio kcal explícitamente → confianza ALTA, úsalas sin modificar.
- Si calculaste por distancia/pasos → indica en nota cómo lo calculaste.
- duracion_min: null si no se mencionó ni puede inferirse."""


# ── 4. Análisis de foto (plato O tabla nutricional) ───────────────────────────
PROMPT_FOTO = """Analiza esta imagen con cuidado.

Primero determina qué tipo de imagen es:

CASO A — Es un plato, alimento o bebida:
Usa el cubierto (tenedor/cuchillo) como referencia de tamaño si está presente.
Devuelve:
{
  "tipo": "PLATO",
  "descripcion": "descripción del alimento/plato y componentes",
  "kcal_estimadas": total_entero,
  "confianza": "ALTA | MEDIA | BAJA",
  "detalle": "Componente 1 ~Xkcal, Componente 2 ~Ykcal"
}

CASO B — Es una tabla nutricional (etiqueta de producto):
Lee TODOS los valores con precisión. Presta especial atención a:
- "Número de porciones por envase" o "Porciones por envase" → va en porciones_por_envase
- "Calorías por porción" o "Por porción" → va en kcal_por_porcion
- "Tamaño de porción" → va en porcion_g (solo el número en gramos o ml)

Devuelve:
{
  "tipo": "TABLA_NUTRICIONAL",
  "producto": "nombre del producto si es legible, o descripción del empaque",
  "kcal_por_porcion": numero_entero_exacto_de_la_tabla,
  "porcion_g": numero_gramos_o_ml_de_la_porcion,
  "porciones_por_envase": numero_entero_exacto_de_la_tabla,
  "kcal_total_envase": kcal_por_porcion_multiplicado_por_porciones
}

Reglas CRÍTICAS:
- Para TABLA_NUTRICIONAL: copia los números EXACTAMENTE como aparecen en la etiqueta. NO los inventes.
- Tamaño de porcion es DIFERENTE a numero de porciones por envase, captura únicamente el NUMERO DE PORCIONES POR ENVASE o NUMERO DE PORCIONES y si el campo dice 5, pon 5 en porciones_por_envase.
- Si el campo "Por porción" dice 45, pon 45 en kcal_por_porcion.
- Si el líquido en el recipiente NO es agua pura, NUNCA pongas 0 kcal.
- Yogur, kéfir, jugos, batidos, leches vegetales SIEMPRE tienen calorías.
- Si no puedes identificar bien el alimento, da un estimado razonable y baja confianza a BAJA."""


# ── 5. Sugerencia nutricional ─────────────────────────────────────────────────
PROMPT_SUGERENCIA = """Balance calórico del usuario hoy:
- Objetivo: {objetivo} kcal
- Consumidas: {consumidas} kcal
- Quemadas ejercicio: {quemadas} kcal
- Disponibles: {disponibles} kcal
- Hora: {hora}
- Registros: {comidas}

Devuelve ÚNICAMENTE este JSON:
{{
  "mensaje": "sugerencia práctica y motivadora en 2-3 oraciones",
  "opciones": ["opción comida 1", "opción comida 2", "opción comida 3"],
  "advertencia": "advertencia si hay desbalance grave, o null"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request
# ─────────────────────────────────────────────────────────────────────────────

class IntentRequest(BaseModel):
    texto: str

class InferirComidaRequest(BaseModel):
    texto: str

class InferirEjercicioRequest(BaseModel):
    texto: str
    peso_kg: float = 70.0
    edad: int = 30

class FotoRequest(BaseModel):
    imagen_b64: str
    porciones_consumidas: Optional[float] = None

class SugerenciaRequest(BaseModel):
    objetivo: float
    consumidas: float
    quemadas: float
    disponibles: float
    hora: str = "12:00"
    comidas: str = "Sin registros aún"


# ─────────────────────────────────────────────────────────────────────────────
# Validadores
# ─────────────────────────────────────────────────────────────────────────────

INTENTS_VALIDOS = {"COMIDA", "EJERCICIO", "CONSULTA", "OTRO"}
CONFIANZAS      = {"ALTA", "MEDIA", "BAJA"}


def _v_intent(data: dict) -> dict:
    intent = str(data.get("intent", "OTRO")).upper().strip()
    if intent not in INTENTS_VALIDOS:
        intent = "OTRO"
    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in CONFIANZAS:
        confianza = "MEDIA"
    return {"intent": intent, "confianza": confianza}


def _v_comida(data: dict) -> dict:
    try:
        kcal = int(float(data.get("kcal", 0)))
    except (TypeError, ValueError):
        raise ValueError("kcal debe ser un número.")
    if kcal <= 0:
        raise ValueError("El modelo devolvió 0 kcal — revisar descripción.")
    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in CONFIANZAS:
        confianza = "MEDIA"
    return {
        "descripcion": str(data.get("descripcion", "")).strip(),
        "kcal":        kcal,
        "detalle":     str(data.get("detalle", "")).strip() or None,
        "confianza":   confianza,
        "nota":        str(data.get("nota", "")).strip() or None,
    }


def _v_ejercicio(data: dict) -> dict:
    try:
        kcal = int(float(data.get("kcal_quemadas", 0)))
    except (TypeError, ValueError):
        raise ValueError("kcal_quemadas debe ser un número.")
    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in CONFIANZAS:
        confianza = "MEDIA"
    return {
        "descripcion":   str(data.get("descripcion", "")).strip(),
        "kcal_quemadas": kcal,
        "duracion_min":  int(data["duracion_min"]) if data.get("duracion_min") is not None else None,
        "distancia_km":  float(data["distancia_km"]) if data.get("distancia_km") is not None else None,
        "confianza":     confianza,
        "nota":          str(data.get("nota", "")).strip() or None,
    }


def _v_foto(data: dict, porciones: Optional[float]) -> dict:
    tipo = str(data.get("tipo", "PLATO")).upper().strip()

    if tipo == "TABLA_NUTRICIONAL":
        kcal_porcion = float(data.get("kcal_por_porcion", 0))
        p = porciones or 1.0
        return {
            "tipo":               "TABLA_NUTRICIONAL",
            "producto":           str(data.get("producto", "")).strip(),
            "kcal_por_porcion":   int(kcal_porcion),
            "porcion_g":          data.get("porcion_g"),
            "porciones_consumidas": p,
            "kcal_estimadas":     int(kcal_porcion * p),
            "confianza":          "ALTA",
            "detalle":            f"{p} porción(es) × {int(kcal_porcion)} kcal",
        }

    # PLATO por defecto
    try:
        kcal = int(float(data.get("kcal_estimadas", 0)))
    except (TypeError, ValueError):
        kcal = 0
    confianza = str(data.get("confianza", "MEDIA")).upper().strip()
    if confianza not in CONFIANZAS:
        confianza = "MEDIA"
    return {
        "tipo":           "PLATO",
        "descripcion":    str(data.get("descripcion", "")).strip(),
        "kcal_estimadas": kcal,
        "confianza":      confianza,
        "detalle":        str(data.get("detalle", "")).strip() or None,
    }


import re as _re

def _extraer_json_robusto(texto: str) -> dict:
    """Extrae el primer objeto JSON del texto aunque venga con texto previo."""
    import json
    # Intentar directo primero
    try:
        return json.loads(texto)
    except Exception:
        pass
    # Buscar el primer { ... } en el texto
    match = _re.search(r'\{.*\}', texto, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    raise ValueError(f"No se pudo extraer JSON válido del texto: {texto[:200]}")


def _v_sugerencia(data: dict) -> dict:
    mensaje = str(data.get("mensaje", "")).strip()
    if not mensaje:
        raise ValueError("Sin sugerencia.")
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
    Router IA de KALO. Todos los endpoints bajo /kalo/
    En kalo-api: LLM_BASE_URL=http://<aibase-ip>:8181/kalo
    """

    prefix = "/kalo"

    def _registrar_rutas(self) -> None:

        @self.router.post("/clasificar-intent")
        def clasificar_intent(req: IntentRequest):
            """Clasifica el intent del texto libre: COMIDA, EJERCICIO, CONSULTA u OTRO."""
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="Texto vacío.")
            try:
                raw = self.motor.texto(PROMPT_INTENT.format(texto=req.texto.strip()), max_tokens=80)
                data = _extraer_json_robusto(raw)
                return _v_intent(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/inferir-comida")
        def inferir_comida(req: InferirComidaRequest):
            """Infiere calorías de texto libre de comida/bebida."""
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="Texto vacío.")
            try:
                raw = self.motor.texto(PROMPT_INFERIR_COMIDA.format(texto=req.texto.strip()), max_tokens=300)
                data = _extraer_json_robusto(raw)
                return _v_comida(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/inferir-ejercicio")
        def inferir_ejercicio(req: InferirEjercicioRequest):
            """Infiere calorías quemadas de texto libre de ejercicio."""
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="Texto vacío.")
            try:
                raw = self.motor.texto(
                    PROMPT_INFERIR_EJERCICIO.format(
                        texto=req.texto.strip(),
                        peso_kg=req.peso_kg,
                        edad=req.edad,
                    ),
                    max_tokens=300,
                )
                data = _extraer_json_robusto(raw)
                return _v_ejercicio(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/analizar-foto-comida")
        def analizar_foto_comida(req: FotoRequest):
            """
            Analiza imagen — detecta automáticamente PLATO o TABLA_NUTRICIONAL.
            Para tabla: usa porciones_consumidas para calcular kcal totales.
            """
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="Imagen vacía.")
            try:
                raw = self.motor.imagen(PROMPT_FOTO, req.imagen_b64, max_tokens=400)
                data = _extraer_json_robusto(raw)
                return _v_foto(data, req.porciones_consumidas)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/sugerencia-nutricional")
        def sugerencia_nutricional(req: SugerenciaRequest):
            """Genera sugerencia nutricional personalizada según el balance del día."""
            try:
                data = self.motor.extraer_json(
                    self.motor.texto(
                        PROMPT_SUGERENCIA.format(
                            objetivo=req.objetivo,
                            consumidas=req.consumidas,
                            quemadas=req.quemadas,
                            disponibles=req.disponibles,
                            hora=req.hora,
                            comidas=req.comidas,
                        ),
                        max_tokens=400,
                    )
                )
                return _v_sugerencia(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))