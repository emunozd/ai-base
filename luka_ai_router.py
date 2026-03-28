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
"""
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, KingsrowAI, MotorInferencia

# ─────────────────────────────────────────────────────────────────────────────
# Constantes LUKA
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIAS_VALIDAS = {
    "HOGAR", "CANASTA", "MEDICAMENTOS", "OCIO", "ANTOJO",
    "TRANSPORTE", "TECNOLOGÍA", "ROPA", "EDUCACIÓN", "MASCOTAS",
}

SYSTEM_PROMPT = (
    "Eres LUKA, el motor de análisis financiero de una app de finanzas personales para colombianos. "
    "Tu única función es clasificar gastos en categorías y devolver JSON válido. "
    "Responde SIEMPRE en español. NUNCA uses otro idioma. "
    "NUNCA expliques tu razonamiento. SOLO devuelve el JSON solicitado, sin texto adicional."
)

PROMPT_FACTURA = """Analiza el siguiente contenido de una factura o recibo y clasifica el gasto total por categorías.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, CANASTA, MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS:
- CANASTA: mercado del día a día, alimentos básicos, aseo del hogar.
- ANTOJO: comida por placer, restaurantes, domicilios, dulces, snacks.
- HOGAR: arriendo, servicios públicos, reparaciones, muebles.
- MEDICAMENTOS: farmacia, consultas médicas, parafarmacia.
- OCIO: streaming, entretenimiento, deportes, viajes, videojuegos.
- TRANSPORTE: gasolina, taxi, bus, peajes, Uber.
- TECNOLOGÍA: celulares, computadores, software, internet.
- ROPA: ropa, calzado, accesorios de vestir.
- EDUCACIÓN: cursos, libros, útiles escolares, matrículas.
- MASCOTAS: veterinario, concentrado, accesorios para mascotas.
- Solo incluye categorías con valor > 0.
- Los totales deben ser números con máximo 2 decimales.

{contenido}

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
{{
  "categorias": {{"CATEGORIA": total_en_pesos}},
  "comercio": "nombre del comercio o null",
  "fecha": "YYYY-MM-DD o null",
  "total_factura": total_general_en_pesos
}}"""

PROMPT_GASTO_MANUAL = """El usuario describió uno o varios gastos con sus propias palabras. Extrae TODOS los gastos mencionados.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, CANASTA, MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS PARA EL MONTO:
- Interpreta cualquier formato colombiano: 5k → 5000, 5mil → 5000, 5.000 → 5000, 5,000 → 5000, 5 lucas → 5000.
- El monto siempre es en pesos colombianos (COP).
- Si un gasto no tiene monto claro, usa null.
- Si hay varios gastos, devuelve uno por ítem.

REGLAS PARA LA CATEGORÍA:
- CANASTA: pan, arroz, leche, huevos, frutas, verduras, mercado básico.
- ANTOJO: gaseosa, snacks, dulces, comida por placer, restaurante, domicilio.
- MASCOTAS: comida para perro/gato, veterinario, accesorios de mascotas.
- HOGAR: servicios públicos, arriendo, reparaciones.
- MEDICAMENTOS: drogas, farmacia, consulta médica.
- OCIO: entretenimiento, streaming, deporte, videojuegos.
- TRANSPORTE: bus, taxi, Uber, gasolina, peaje.
- TECNOLOGÍA: celular, computador, internet, software.
- ROPA: ropa, zapatos, accesorios de vestir.
- EDUCACIÓN: libros, cursos, útiles, matrícula.

DESCRIPCIÓN: {descripcion}

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
[
  {{"categoria": "NOMBRE_CATEGORIA", "monto": valor_numerico_o_null, "descripcion": "descripcion corta del item"}},
  ...
]"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de validación (privados a LUKA)
# ─────────────────────────────────────────────────────────────────────────────
def _validar_categorias(categorias: dict) -> dict:
    resultado = {}
    for cat, total in categorias.items():
        cat_upper = cat.upper().strip()
        if cat_upper in CATEGORIAS_VALIDAS:
            try:
                valor = float(total)
                if valor > 0:
                    resultado[cat_upper] = round(valor, 2)
            except (TypeError, ValueError):
                pass  # valor inválido, se ignora silenciosamente
        # categorías desconocidas se ignoran
    if not resultado:
        raise ValueError("El modelo no devolvió ninguna categoría válida.")
    return resultado


def _validar_gastos_manuales(data: Any) -> list:
    if not isinstance(data, list):
        data = [data]
    resultado = []
    for item in data:
        categoria = str(item.get("categoria", "")).upper().strip()
        if categoria not in CATEGORIAS_VALIDAS:
            continue
        monto_raw = item.get("monto")
        monto     = round(float(monto_raw), 2) if monto_raw is not None else None
        descripcion = item.get("descripcion", "").strip() or None
        resultado.append({
            "categoria":   categoria,
            "monto":       monto,
            "descripcion": descripcion,
        })
    if not resultado:
        raise ValueError("El modelo no devolvió ningún gasto válido.")
    return resultado


def _procesar_resultado_factura(data: dict) -> dict:
    categorias = _validar_categorias(data.get("categorias", {}))
    return {
        "categorias":    categorias,
        "comercio":      data.get("comercio") or None,
        "fecha":         data.get("fecha") or None,
        "total_factura": round(float(data.get("total_factura") or sum(categorias.values())), 2),
    }


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

    Nota Docker Desktop: Docker Desktop en Mac enruta tráfico saliente
    de contenedores a través de la red del host. Los contenedores pueden
    alcanzar KR_HOST:KR_PORT directamente. NO usar host.docker.internal
    aquí porque ese alias apunta a 127.0.0.1 del host, y el servidor ya
    no escucha en 127.0.0.1.
    """

    prefix = "/luka"

    def _registrar_rutas(self) -> None:

        @self.router.post("/categorizar-factura-texto")
        def categorizar_factura_texto(req: TextoRequest):
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
            try:
                prompt = PROMPT_FACTURA.format(
                    contenido=f"TEXTO DE LA FACTURA:\n{req.texto.strip()}"
                )
                data = self.motor.extraer_json(
                    self.motor.texto(prompt, max_tokens=600)
                )
                return _procesar_resultado_factura(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-factura-imagen")
        def categorizar_factura_imagen(req: ImagenRequest):
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                prompt = PROMPT_FACTURA.format(
                    contenido="Analiza la imagen del recibo o factura que se adjunta."
                )
                data = self.motor.extraer_json(
                    self.motor.imagen(prompt, req.imagen_b64, max_tokens=600)
                )
                return _procesar_resultado_factura(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-gasto-manual")
        def categorizar_gasto_manual(req: GastoManualRequest):
            if not req.descripcion.strip():
                raise HTTPException(status_code=422, detail="La descripción no puede estar vacía.")
            try:
                data = self.motor.extraer_json(
                    self.motor.texto(
                        PROMPT_GASTO_MANUAL.format(descripcion=req.descripcion.strip()),
                        max_tokens=300,
                    )
                )
                return _validar_gastos_manuales(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))


# luka_ai_router.py no tiene punto de entrada.
# El servidor se levanta desde main.py — ver ~/projects/AIBase/main.py

