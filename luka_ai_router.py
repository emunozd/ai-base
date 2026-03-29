"""
luka_ai_router.py — Router de inferencia IA para LUKA
======================================================
Clase hija de BaseRouter que define los tres endpoints de LUKA:
  POST /luka/categorizar-factura-texto
  POST /luka/categorizar-factura-imagen
  POST /luka/categorizar-gasto-manual

Este archivo NO es el punto de entrada del servidor.
El servidor se levanta desde main.py — ese es el único LaunchDaemon.

Para agregar otro proyecto en el futuro:
    1. Crea otro_proyecto_router.py con una clase que extienda BaseRouter.
    2. Importa y registra esa clase en main.py.
    3. Sus endpoints quedan bajo /otro-proyecto/...

Estrategia de categorización de facturas con imagen (v3 — directo):
    Pasada única → modelo de visión lee la imagen y devuelve JSON directamente
                   [{descripcion, monto, descuento, categoria}]
    Python       → aplica descuento (monto - descuento) y agrupa totales por categoría

Estrategia anterior (v2 — dos pasadas de texto) está comentada para rollback.

El formato de salida hacia luka-api no cambia en ningún caso.
"""
import logging
import re
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, KingsrowAI, MotorInferencia

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes LUKA
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIAS_VALIDAS = {
    "HOGAR", "HOGAR_ARRIENDO", "HOGAR_SERVICIOS", "HOGAR_REPARACIONES",
    "CANASTA", "CANASTA_VERDURAS", "CANASTA_PROTEINA", "CANASTA_ASEO", "CANASTA_HIGIENE",
    "MEDICAMENTOS", "OCIO", "ANTOJO",
    "TRANSPORTE", "TECNOLOGÍA", "ROPA", "EDUCACIÓN", "MASCOTAS",
}

SYSTEM_PROMPT = (
    "Eres LUKA, el motor de análisis financiero de una app de finanzas personales para colombianos. "
    "Tu única función es clasificar gastos en categorías y devolver JSON válido. "
    "Responde SIEMPRE en español. NUNCA uses otro idioma. "
    "NUNCA expliques tu razonamiento. SOLO devuelve el JSON solicitado, sin texto adicional."
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts v3 — imagen directa a JSON
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_IMAGEN_DIRECTA = """Analiza esta imagen de factura o recibo.
Para cada artículo extrae: nombre, monto, descuento y categoría.

REGLAS DE MONTOS — MUY IMPORTANTE:
- El punto (.) es separador de miles en facturas colombianas. 42.668 = 42668. NUNCA es decimal.
- "monto" es el precio del artículo antes de cualquier descuento, como entero sin separadores.
- "descuento" es el valor rebajado si hay una línea de descuento asociada al artículo. Sin descuento usa 0.
- UN DESCUENTO SIEMPRE ES MENOR QUE EL PRECIO ORIGINAL. Si encuentras dos valores asociados a un artículo
  donde uno es mayor y otro menor, el MAYOR es el monto y el MENOR es el descuento — sin excepción.
- Cada descuento pertenece ÚNICAMENTE al artículo que lo precede inmediatamente.
  Un valor grande NO puede ser el descuento de un artículo de precio pequeño.
- Los valores con signo "-" al final son descuentos, no precios.
- La suma de (monto - descuento) de todos los artículos debe SER IGUAL al VALOR TOTAL
  que aparece al final de la factura. Si tu suma se aleja mucho del Valor Total indicado,
  DEBES REVISAR los artículos donde confundiste precio con descuento y CORREGIRLOS antes de responder.
- NO restes nada — devuelve monto y descuento por separado como enteros.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE CATEGORÍA:
CANASTA_VERDURAS  → frutas, verduras, tubérculos, legumbres frescas.
CANASTA_PROTEINA  → carnes, pollo, pescado, huevos, lácteos, mantequilla.
CANASTA_ASEO      → detergente, cloro, limpiapisos, trapero, desinfectante, vinagre limpieza, toallas manos, pañuelos.
CANASTA_HIGIENE   → shampoo, crema, maquillaje, cuidado facial, jabón baño, aceite corporal, toallas higiénicas.
CANASTA           → arroz, aceite, sal, azúcar, pasta, enlatados, especias, champiñones. Usar si no encaja arriba.
HOGAR_ARRIENDO    → arriendo, administración.
HOGAR_SERVICIOS   → agua, luz, gas, internet, teléfono fijo.
HOGAR_REPARACIONES → plomería, electricista, pintura, puertas. NUNCA electrodomésticos.
HOGAR             → hogar sin clasificación clara.
MEDICAMENTOS      → farmacia, medicamentos, consulta médica.
OCIO              → entretenimiento, streaming, videojuegos, viajes.
ANTOJO            → comida por placer, restaurante, domicilio, dulces, snacks, gaseosas.
TRANSPORTE        → bus, taxi, Uber, gasolina, peaje.
TECNOLOGÍA        → celular, computador, TV, electrodomésticos, software.
ROPA              → ropa, zapatos, accesorios de vestir.
EDUCACIÓN         → libros, cursos, útiles, matrícula.
MASCOTAS          → veterinario, concentrado, accesorios mascotas.

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
{{
  "comercio": "nombre del comercio o null",
  "fecha": "YYYY-MM-DD o null",
  "items": [
    {{"descripcion": "nombre del artículo", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "NOMBRE_CATEGORIA"}},
    ...
  ]
}}"""

PROMPT_TEXTO_DIRECTO = """Analiza el siguiente texto de factura o recibo.
Para cada artículo extrae: nombre, monto, descuento y categoría.

REGLAS DE MONTOS — MUY IMPORTANTE:
- El punto (.) es separador de miles en facturas colombianas. 42.668 = 42668. NUNCA es decimal.
- "monto" es el precio del artículo antes del descuento, como entero sin separadores.
- "descuento" es el valor de la línea "Descuento XX% VALOR-" si existe. Sin descuento usa 0.
- Un descuento SIEMPRE es menor que el monto del artículo.
- NO restes — devuelve monto y descuento por separado. Python hará la resta.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE CATEGORÍA:
CANASTA_VERDURAS  → frutas, verduras, tubérculos, legumbres frescas.
CANASTA_PROTEINA  → carnes, pollo, pescado, huevos, lácteos, mantequilla.
CANASTA_ASEO      → detergente, cloro, limpiapisos, trapero, desinfectante, vinagre limpieza, toallas manos, pañuelos.
CANASTA_HIGIENE   → shampoo, crema, maquillaje, cuidado facial, jabón baño, aceite corporal, toallas higiénicas.
CANASTA           → arroz, aceite, sal, azúcar, pasta, enlatados, especias, champiñones. Usar si no encaja arriba.
HOGAR_ARRIENDO    → arriendo, administración.
HOGAR_SERVICIOS   → agua, luz, gas, internet, teléfono fijo.
HOGAR_REPARACIONES → plomería, electricista, pintura, puertas. NUNCA electrodomésticos.
HOGAR             → hogar sin clasificación clara.
MEDICAMENTOS      → farmacia, medicamentos, consulta médica.
OCIO              → entretenimiento, streaming, videojuegos, viajes.
ANTOJO            → comida por placer, restaurante, domicilio, dulces, snacks, gaseosas.
TRANSPORTE        → bus, taxi, Uber, gasolina, peaje.
TECNOLOGÍA        → celular, computador, TV, electrodomésticos, software.
ROPA              → ropa, zapatos, accesorios de vestir.
EDUCACIÓN         → libros, cursos, útiles, matrícula.
MASCOTAS          → veterinario, concentrado, accesorios mascotas.

TEXTO DE LA FACTURA:
{{contenido}}

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
{{
  "comercio": "nombre del comercio o null",
  "fecha": "YYYY-MM-DD o null",
  "items": [
    {{"descripcion": "nombre del artículo", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "NOMBRE_CATEGORIA"}},
    ...
  ]
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompts v2 — comentados para rollback
# ─────────────────────────────────────────────────────────────────────────────

# PROMPT_TRANSCRIBIR = """Transcribe EXACTAMENTE el texto de esta factura o recibo.
# Incluye TODOS los artículos con sus cantidades y VALORES.
# Mantén los números EXACTAMENTE como aparecen: NO redondees, NO interpretes, NO calcules.
# Conserva los puntos y comas tal cual están en la imagen.
# Solo transcribe — no clasifiques ni expliques nada."""

# PROMPT_CLASIFICAR_ITEMS = """Analiza el texto de esta factura. Para cada artículo devuelve
# nombre, monto, descuento y categoría.
# [... ver git history para contenido completo ...]
# TEXTO DE LA FACTURA:
# {productos}
# Devuelve ÚNICAMENTE este JSON:
# [
#   {"descripcion": "...", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "..."},
#   ...
# ]"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt gastos manuales — sin cambios
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_GASTO_MANUAL = """El usuario describió uno o varios gastos con sus propias palabras. Extrae TODOS los gastos mencionados.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS PARA EL MONTO:
- Interpreta cualquier formato colombiano: 5k → 5000, 5mil → 5000, 5.000 → 5000, 5,000 → 5000, 5 lucas → 5000.
- El monto siempre es en pesos colombianos (COP).
- Si un gasto no tiene monto claro, usa null.
- Si hay varios gastos, devuelve uno por ítem.

REGLAS PARA LA CATEGORÍA:
CANASTA_VERDURAS  → frutas, verduras, tubérculos, granos secos, legumbres frescas.
CANASTA_PROTEINA  → carnes, pollo, pescado, mariscos, huevos, lácteos (leche, queso, yogur, mantequilla).
CANASTA_ASEO      → detergente, jabón ropa, cloro, limpiapisos, escoba, trapero, desinfectante.
CANASTA_HIGIENE   → desodorante, shampoo, crema, maquillaje, cuidado facial, toallas higiénicas, pañales.
CANASTA           → alimentos básicos que no encajan en verduras ni proteína (arroz, aceite, sal, azúcar, pasta, enlatados, especias).
HOGAR_ARRIENDO    → arriendo, administración.
HOGAR_SERVICIOS   → agua, luz, gas, internet, teléfono fijo.
HOGAR_REPARACIONES → plomería, electricista, pintura, puertas, ventanas. NUNCA electrodomésticos ni TV.
HOGAR             → gastos del hogar sin clasificación clara.
MEDICAMENTOS      → drogas, farmacia, consulta médica, parafarmacia.
OCIO              → entretenimiento, streaming, deporte, videojuegos, viajes.
ANTOJO            → gaseosa, snacks, dulces, comida por placer, restaurante, domicilio.
TRANSPORTE        → bus, taxi, Uber, gasolina, peaje, parqueadero.
TECNOLOGÍA        → celular, computador, televisor, electrodomésticos (nevera, lavadora, microondas), software, datos móviles.
ROPA              → ropa, zapatos, accesorios de vestir.
EDUCACIÓN         → libros, cursos, útiles, matrícula.
MASCOTAS          → veterinario, concentrado, accesorios mascotas.

DESCRIPCIÓN: {descripcion}

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
[
  {{"categoria": "NOMBRE_CATEGORIA", "monto": valor_numerico_o_null, "descripcion": "descripcion corta del item"}},
  ...
]"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validar_gastos_manuales(data: Any) -> list:
    """Sin cambios — sigue igual para gastos manuales."""
    if not isinstance(data, list):
        data = [data]
    resultado = []
    for item in data:
        categoria   = str(item.get("categoria", "")).upper().strip()
        if categoria not in CATEGORIAS_VALIDAS:
            continue
        monto_raw   = item.get("monto")
        monto       = round(float(monto_raw), 2) if monto_raw is not None else None
        descripcion = item.get("descripcion", "").strip() or None
        resultado.append({
            "categoria":   categoria,
            "monto":       monto,
            "descripcion": descripcion,
        })
    if not resultado:
        raise ValueError("El modelo no devolvió ningún gasto válido.")
    return resultado


def _agrupar_por_categoria(items: list[dict]) -> dict[str, float]:
    """
    Recibe ítems con monto, descuento y categoria.
    Corrige inversión si descuento > monto.
    Python aplica descuento y agrupa totales por categoría.
    """
    categorias: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("categoria", "")).upper().strip()
        if cat not in CATEGORIAS_VALIDAS:
            cat = "CANASTA"
        try:
            monto     = float(item.get("monto", 0) or 0)
            descuento = float(item.get("descuento", 0) or 0)
            if descuento > monto and monto > 0:
                monto, descuento = descuento, monto
            elif monto == 0 and descuento > 0:
                monto, descuento = descuento, 0
            neto = round(monto - descuento, 2)
            if neto <= 0:
                continue
        except (TypeError, ValueError):
            continue
        categorias[cat] = round(categorias.get(cat, 0.0) + neto, 2)
    return categorias


def _extraer_resultado(data: Any) -> tuple[list, str | None, str | None]:
    """Extrae items, comercio y fecha del JSON devuelto por el modelo."""
    if isinstance(data, list):
        return data, None, None
    if isinstance(data, dict):
        comercio = data.get("comercio") or None
        fecha    = data.get("fecha") or None
        items    = data.get("items") or []
        return items, comercio, fecha
    return [], None, None


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request LUKA
# ─────────────────────────────────────────────────────────────────────────────
class TextoRequest(BaseModel):
    texto: str

class ImagenRequest(BaseModel):
    imagen_b64: str

class GastoManualRequest(BaseModel):
    descripcion: str


# ─────────────────────────────────────────────────────────────────────────────
# Router LUKA
# ─────────────────────────────────────────────────────────────────────────────
class LukaRouter(BaseRouter):
    """
    Router de inferencia IA para LUKA.
    Todos los endpoints quedan bajo el prefijo /luka/.

    La luka-api (Docker) debe apuntar a:
        KR_HOST:KR_PORT/luka/categorizar-factura-texto
        KR_HOST:KR_PORT/luka/categorizar-factura-imagen
        KR_HOST:KR_PORT/luka/categorizar-gasto-manual
    """

    prefix = "/luka"

    def _registrar_rutas(self) -> None:

        @self.router.post("/categorizar-factura-texto")
        def categorizar_factura_texto(req: TextoRequest):
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
            try:
                prompt = PROMPT_TEXTO_DIRECTO.replace("{{contenido}}", req.texto.strip())
                raw    = self.motor.texto(prompt, max_tokens=1200)
                logger.info("RESPUESTA TEXTO DIRECTO:\n%s", raw)
                data   = self.motor.extraer_json(raw)
                items, comercio, fecha = _extraer_resultado(data)
                categorias = _agrupar_por_categoria(items)
                if not categorias:
                    raise ValueError("No se pudieron clasificar los ítems.")
                return {
                    "categorias":    categorias,
                    "comercio":      comercio,
                    "fecha":         fecha,
                    "total_factura": round(sum(categorias.values()), 2),
                }
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-factura-imagen")
        def categorizar_factura_imagen(req: ImagenRequest):
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                # ── v3: modelo de visión → JSON directo ───────────────────────
                raw = self.motor.imagen(
                    PROMPT_IMAGEN_DIRECTA,
                    req.imagen_b64,
                    max_tokens=2400,
                )
                logger.info("RESPUESTA IMAGEN DIRECTA:\n%s", raw)
                data  = self.motor.extraer_json(raw)
                items, comercio, fecha = _extraer_resultado(data)
                categorias = _agrupar_por_categoria(items)
                if not categorias:
                    raise ValueError("No se pudieron clasificar los ítems.")
                return {
                    "categorias":    categorias,
                    "comercio":      comercio,
                    "fecha":         fecha,
                    "total_factura": round(sum(categorias.values()), 2),
                }
                # ── ROLLBACK v2: descomentar bloque y comentar v3 ─────────────
                # texto_transcrito = self.motor.imagen(
                #     PROMPT_TRANSCRIBIR, req.imagen_b64, max_tokens=2400,
                # )
                # if not texto_transcrito.strip():
                #     raise ValueError("No se pudo transcribir el contenido de la imagen.")
                # logger.info("TRANSCRIPCIÓN PASADA 1:\n%s", texto_transcrito)
                # prompt = PROMPT_CLASIFICAR_ITEMS.format(productos=texto_transcrito.strip())
                # raw    = self.motor.texto(prompt, max_tokens=2400)
                # logger.info("CLASIFICACIÓN PASADA 2:\n%s", raw)
                # data   = self.motor.extraer_json(raw)
                # if not isinstance(data, list):
                #     raise ValueError("El modelo no devolvió una lista de clasificaciones.")
                # categorias = _agrupar_por_categoria(data)
                # if not categorias:
                #     raise ValueError("No se pudieron clasificar los ítems.")
                # return {
                #     "categorias":    categorias,
                #     "comercio":      None,
                #     "fecha":         None,
                #     "total_factura": round(sum(categorias.values()), 2),
                # }
                # ── FIN ROLLBACK v2 ───────────────────────────────────────────

            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-gasto-manual")
        def categorizar_gasto_manual(req: GastoManualRequest):
            """Sin cambios — gastos manuales no necesitan aritmética."""
            if not req.descripcion.strip():
                raise HTTPException(status_code=422, detail="La descripción no puede estar vacía.")
            try:
                data = self.motor.extraer_json(
                    self.motor.texto(
                        PROMPT_GASTO_MANUAL.format(descripcion=req.descripcion.strip()),
                        max_tokens=400,
                    )
                )
                return _validar_gastos_manuales(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))


# luka_ai_router.py no tiene punto de entrada.
# El servidor se levanta desde main.py — ver ~/projects/AIBase/main.py