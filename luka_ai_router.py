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

Estrategia de categorización de facturas:
    El modelo clasifica cada ítem individualmente (no suma).
    Python acumula los totales por categoría → aritmética exacta.
    El formato de salida hacia luka-api no cambia.
"""
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, KingsrowAI, MotorInferencia

import logging
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
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_FACTURA = """Analiza el siguiente contenido de una factura o recibo.
Devuelve CADA ítem de la factura con su monto y categoría. NO sumes — devuelve un objeto por ítem.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE FORMATO NUMÉRICO — MUY IMPORTANTE:
- Esta es una factura colombiana. El punto (.) es separador de MILES, no decimales.
- Ejemplos: 115.316 = 115316 pesos. 42.668 = 42668 pesos.
- NUNCA interpretes el punto como decimal. Siempre es separador de miles.
- Los montos deben ser números enteros en pesos colombianos (COP), sin puntos ni comas.
  Ejemplo: ítem vale 72.500, Descuento 25% 18.126-, el monto final es 72500 - 18126 = 54374.
- Usa SIEMPRE el valor final después de aplicar el descuento.
- El signo "-" al final de un valor indica que es un descuento a restar.

REGLAS DE DESCUENTOS — MUY IMPORTANTE:
- Trata los valores de la factura como una suma acumulativa de arriba a abajo.
- Cualquier valor seguido de "-" es un descuento que se RESTA del ítem inmediatamente anterior.
- El valor junto a "Descuento XX%" con "-" es el monto exacto a restar, no el porcentaje.
- Ejemplo genérico:
    Producto A: 50.000
    Descuento 20% 10.000-
    → monto final de Producto A = 50000 - 10000 = 40000
- NUNCA uses el valor antes del descuento como monto final.
- NUNCA incluyas líneas de descuento como ítems separados.
- Si no hay línea de descuento después de un ítem, el valor mostrado ES el monto final.
- Verifica que la suma de todos los ítems se aproxime al Valor Total de la factura.

REGLAS DE CATEGORÍA:
CANASTA_VERDURAS  → frutas, verduras, tubérculos, granos secos, legumbres frescas.
CANASTA_PROTEINA  → carnes, pollo, pescado, mariscos, huevos, lácteos (leche, queso, yogur, mantequilla).
CANASTA_ASEO      → detergente, jabón para ropa, cloro, limpiapisos, escoba, trapero, desinfectante superficies, vinagre de limpieza.
CANASTA_HIGIENE   → desodorante, shampoo, acondicionador, crema corporal, maquillaje, cuidado facial, toallas higiénicas, pañales, jabón de baño.
CANASTA           → alimentos básicos de difícil clasificación (arroz, aceite, sal, azúcar, pasta, enlatados, especias, salsas, champiñones). Usar si no encaja en CANASTA_VERDURAS ni CANASTA_PROTEINA.
HOGAR_ARRIENDO    → arriendo mensual, administración del edificio.
HOGAR_SERVICIOS   → agua, luz, gas, internet, teléfono fijo.
HOGAR_REPARACIONES → plomería, electricista, pintura, instalación de puertas o ventanas. NUNCA electrodomésticos ni TV.
HOGAR             → elementos del hogar que no encajan en las anteriores (decoración, menaje).
MEDICAMENTOS      → farmacia, medicamentos, consultas médicas, parafarmacia.
OCIO              → streaming, entretenimiento, deportes, viajes, videojuegos, juguetes.
ANTOJO            → comida por placer, restaurantes, domicilios, dulces, snacks, gaseosas.
TRANSPORTE        → gasolina, taxi, bus, peajes, Uber, parqueadero.
TECNOLOGÍA        → celulares, computadores, televisores, electrodomésticos (nevera, lavadora, microondas), software, datos móviles.
ROPA              → ropa, calzado, accesorios de vestir.
EDUCACIÓN         → cursos, libros, útiles escolares, matrículas.
MASCOTAS          → veterinario, concentrado, accesorios para mascotas.

{contenido}

Devuelve ÚNICAMENTE este JSON (lista de ítems), sin explicaciones ni texto adicional:
[
  {{"descripcion": "nombre del ítem", "monto": valor_entero_en_pesos, "categoria": "NOMBRE_CATEGORIA"}},
  ...
]

IMPORTANTE: El campo "monto" debe ser el valor total pagado por ese ítem (precio × cantidad, ya con descuento aplicado), como número entero sin separadores."""

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

PROMPT_TRANSCRIBIR = """Transcribe el texto de esta factura exactamente como aparece, línea por línea.
Incluye todos los nombres de productos, cantidades, precios y descuentos tal como están escritos.
No interpretes, no calcules, no reorganices. Solo copia el texto que ves en la imagen."""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de validación y procesamiento
# ─────────────────────────────────────────────────────────────────────────────

def _validar_gastos_manuales(data: Any) -> list:
    """Sin cambios — sigue igual para gastos manuales."""
    if not isinstance(data, list):
        data = [data]
    resultado = []
    for item in data:
        categoria = str(item.get("categoria", "")).upper().strip()
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


def _procesar_items_factura(items: list, comercio: str = None, fecha: str = None) -> dict:
    """
    Recibe lista de ítems [{descripcion, monto, categoria}] del modelo.
    Python hace la suma por categoría — aritmética exacta, sin depender del modelo.
    Devuelve el mismo formato que antes: {categorias, comercio, fecha, total_factura}.
    """
    categorias: dict[str, float] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("categoria", "")).upper().strip()
        if cat not in CATEGORIAS_VALIDAS:
            continue
        monto_raw = item.get("monto")
        if monto_raw is None:
            continue
        try:
            monto = float(monto_raw)
            if monto <= 0:
                continue
        except (TypeError, ValueError):
            continue
        categorias[cat] = round(categorias.get(cat, 0.0) + monto, 2)

    if not categorias:
        raise ValueError("El modelo no devolvió ningún ítem válido.")

    total = round(sum(categorias.values()), 2)

    return {
        "categorias":    categorias,
        "comercio":      comercio or None,
        "fecha":         fecha or None,
        "total_factura": total,
    }


def _extraer_meta_factura(data: Any) -> tuple[list, str | None, str | None]:
    """
    El modelo devuelve una lista de ítems.
    Opcionalmente puede venir envuelta en un dict con comercio/fecha.
    Soporta ambos formatos para robustez.
    """
    comercio = None
    fecha    = None

    if isinstance(data, list):
        return data, comercio, fecha

    if isinstance(data, dict):
        # Formato alternativo: {"items": [...], "comercio": "...", "fecha": "..."}
        comercio = data.get("comercio") or None
        fecha    = data.get("fecha") or None
        items    = data.get("items") or data.get("categorias") or []
        if isinstance(items, list):
            return items, comercio, fecha
        # Si "categorias" es dict (formato antiguo), convertir para compatibilidad
        if isinstance(items, dict):
            converted = [
                {"descripcion": cat, "monto": val, "categoria": cat}
                for cat, val in items.items()
            ]
            return converted, comercio, fecha

    return [], comercio, fecha


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
                # max_tokens aumentado: ahora el modelo devuelve un ítem por línea
                raw  = self.motor.texto(prompt, max_tokens=800)
                data = self.motor.extraer_json(raw)
                items, comercio, fecha = _extraer_meta_factura(data)
                return _procesar_items_factura(items, comercio, fecha)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-factura-imagen")
        def categorizar_factura_imagen(req: ImagenRequest):
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                # Pasada 1 — transcripción literal
                texto_transcrito = self.motor.imagen(
                    PROMPT_TRANSCRIBIR,
                    req.imagen_b64,
                    max_tokens=2400,
                )
                logger.info("TRANSCRIPCIÓN PASADA 1:\n%s", texto_transcrito)
                if not texto_transcrito.strip():
                    raise ValueError("No se pudo transcribir el contenido de la imagen.")

                # Pasada 2 — clasificación completa incluyendo comercio y fecha
                prompt = PROMPT_FACTURA.format(
                    contenido=f"TEXTO DE LA FACTURA:\n{texto_transcrito.strip()}"
                )
                raw   = self.motor.texto(prompt, max_tokens=900)
                data  = self.motor.extraer_json(raw)
                items, comercio, fecha = _extraer_meta_factura(data)
                return _procesar_items_factura(items, comercio, fecha)
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