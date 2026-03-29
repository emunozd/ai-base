"""
luka_ai_router.py — Router de inferencia IA para LUKA
======================================================
Clase hija de BaseRouter que define los tres endpoints de LUKA:
  POST /luka/categorizar-factura-texto
  POST /luka/categorizar-factura-imagen
  POST /luka/categorizar-gasto-manual

Este archivo NO es el punto de entrada del servidor.
El servidor se levanta desde main.py — ese es el único LaunchDaemon.

Estrategia de categorización de facturas:
  - El modelo lee la imagen/texto y devuelve JSON directamente.
  - El modelo infiere "categoria_comercio" basándose en los productos que vende
    el establecimiento (no en el nombre del comercio).
  - Python aplica descuentos (monto - descuento) y agrupa por categoría.
  - Si suma calculada < total real: la diferencia se agrega en categoria_comercio.
  - Si suma calculada > total real con margen >2%: error, el usuario debe reintentar.
  - El total final SIEMPRE iguala el total real de la factura.
"""
import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

from kingsrow_ai_base import BaseRouter, MotorInferencia

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

TOLERANCIA_PCT = 0.02  # 2% de tolerancia para considerar suma correcta


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_IMAGEN_DIRECTA = """Analiza esta imagen de factura o recibo.
Para cada artículo extrae: nombre, monto, descuento y categoría.

PASO 1 — LEE EL VALOR TOTAL:
Busca el "Valor Total" o "Total" al final de la factura y memorízalo.
Ese es el valor de referencia que SIEMPRE DEBES respetar.

PASO 2 — EXTRAE CADA ARTÍCULO:
- "monto" es el precio del artículo antes de cualquier descuento, como entero sin separadores.
- "descuento" es el valor de la línea de descuento asociada al artículo. Sin descuento usa 0.
- El punto (.) es separador de miles. 42.668 = 42668. NUNCA es decimal.
- UN DESCUENTO SIEMPRE ES MENOR QUE EL MONTO DEL MISMO ARTÍCULO.
  Si tienes dos valores y uno es mayor que el otro, el MAYOR es SIEMPRE el monto.
- Cada descuento pertenece ÚNICAMENTE al artículo que lo precede inmediatamente.
- Los valores con signo "-" al final son descuentos, NO son precios.
- NO restes nada — devuelve monto y descuento por separado como enteros.

PASO 3 — VERIFICACIÓN OBLIGATORIA ANTES DE RESPONDER:
Calcula: SUMA = (monto - descuento) para TODOS los artículos.
Compara esa SUMA con el Valor Total del PASO 1.
Si SUMA difiere del Valor Total en más del 5%:
  - Identifica artículos donde el monto parece demasiado bajo para ese producto.
  - Revisa si confundiste monto con descuento — el monto SIEMPRE es el mayor.
  - Corrige y vuelve a calcular hasta que SUMA ≈ Valor Total.
PROHIBIDO responder si SUMA se aleja más del 5% del Valor Total.

PASO 4 — CATEGORÍA DEL COMERCIO:
Basándote en los productos que vende este establecimiento (NO en su nombre),
determina a qué categoría general pertenece. Ejemplos:
- Si vende alimentos, verduras, carnes → CANASTA
- Si vende medicamentos, productos farmacéuticos → MEDICAMENTOS
- Si vende ropa, calzado → ROPA
- Si vende artículos para el hogar, materiales → HOGAR

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE CATEGORÍA POR ARTÍCULO:
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
  "total_factura": valor_total_real_de_la_factura_como_entero,
  "categoria_comercio": "NOMBRE_CATEGORIA_GENERAL_DEL_ESTABLECIMIENTO",
  "items": [
    {{"descripcion": "nombre del artículo", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "NOMBRE_CATEGORIA"}},
    ...
  ]
}}"""

PROMPT_TEXTO_DIRECTO = """Analiza el siguiente texto de factura o recibo.
Para cada artículo extrae: nombre, monto, descuento y categoría.

REGLAS DE MONTOS:
- El punto (.) es separador de miles. 42.668 = 42668. NUNCA es decimal.
- "monto" es el precio del artículo antes de cualquier descuento, como entero sin separadores.
- "descuento" es el valor de la línea de descuento asociada. Sin descuento usa 0.
- UN DESCUENTO SIEMPRE ES MENOR QUE EL MONTO DEL MISMO ARTÍCULO.
- NO restes nada — devuelve monto y descuento por separado como enteros.

CATEGORÍA DEL COMERCIO:
Basándote en los productos que vende este establecimiento (NO en su nombre),
determina a qué categoría general pertenece.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE CATEGORÍA POR ARTÍCULO:
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
{contenido}

Devuelve ÚNICAMENTE este JSON, sin explicaciones ni texto adicional:
{{
  "comercio": "nombre del comercio o null",
  "fecha": "YYYY-MM-DD o null",
  "total_factura": valor_total_real_de_la_factura_como_entero,
  "categoria_comercio": "NOMBRE_CATEGORIA_GENERAL_DEL_ESTABLECIMIENTO",
  "items": [
    {{"descripcion": "nombre del artículo", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "NOMBRE_CATEGORIA"}},
    ...
  ]
}}"""

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


def _extraer_resultado(data: Any) -> tuple[list, str | None, str | None, float | None, str | None]:
    """Extrae items, comercio, fecha, total_factura y categoria_comercio del JSON del modelo."""
    if isinstance(data, list):
        return data, None, None, None, None
    if isinstance(data, dict):
        comercio     = data.get("comercio") or None
        fecha        = data.get("fecha") or None
        items        = data.get("items") or []
        cat_comercio = None
        total        = None

        raw_cat = data.get("categoria_comercio")
        if raw_cat:
            raw_cat = str(raw_cat).upper().strip()
            if raw_cat in CATEGORIAS_VALIDAS:
                cat_comercio = raw_cat

        try:
            raw_total = data.get("total_factura")
            if raw_total is not None:
                total = float(str(raw_total).replace(".", "").replace(",", ""))
        except (ValueError, TypeError):
            pass

        return items, comercio, fecha, total, cat_comercio

    return [], None, None, None, None


def _agrupar_por_categoria(
    items: list[dict],
    total_real: float | None,
    cat_comercio: str | None,
) -> dict[str, float]:
    """
    Agrupa ítems por categoría aplicando descuentos.
    - Corrige inversión monto/descuento automáticamente.
    - Si suma < total_real: agrega diferencia en cat_comercio (o CANASTA si es None).
    - Si suma > total_real con margen >2%: lanza ValueError para warning al usuario.
    """
    categorias: dict[str, float] = {}

    for item in items:
        #logger.info("EN EL FOR")
        if not isinstance(item, dict):
            continue
        cat = str(item.get("categoria", "")).upper().strip()
        #logger.info("CAT:\n%s", cat)
        if cat not in CATEGORIAS_VALIDAS:
            cat = "CANASTA"
        try:
            #logger.info("TRY")
            monto     = float(item.get("monto", 0) or 0)
            #logger.info("monto:\n%s", monto)
            descuento = float(item.get("descuento", 0) or 0)
            #logger.info("descuento:\n%s", descuento)
            # Corregir inversión: el descuento nunca puede ser mayor que el monto
            if descuento > monto and monto > 0:
                monto, descuento = descuento, monto
            elif monto == 0 and descuento > 0:
                monto, descuento = descuento, 0
            neto = round(monto - descuento, 2)
            #logger.info("neto:\n%s", neto)
            if neto <= 0:
                continue
        except (TypeError, ValueError):
            continue
        #logger.info("despues try")
        categorias[cat] = round(categorias.get(cat, 0.0) + neto, 2)
        #logger.info("categorias[cat]:\n%s", categorias[cat])

    if not categorias:
        raise ValueError("No se pudieron clasificar los ítems.")

    # Sin total real — devolver lo calculado sin ajuste
    if not total_real or total_real <= 0:
        return categorias

    total_calculado = round(sum(categorias.values()), 2)
    logger.info("total_calculado:\n%s", total_calculado)
    diferencia      = round(total_real - total_calculado, 2)
    logger.info("diferencia:\n%s", diferencia)
    tolerancia      = round(total_real * TOLERANCIA_PCT, 2)
    logger.info("tolerancia:\n%s", tolerancia)
    #logger.info("total_calculado:\n%s", total_calculado)
    # Suma mayor que el total real con margen significativo → warning al usuario
    if diferencia < -tolerancia:
        logger.info("error debido a diferencia mejor a tolerancia")
        raise ValueError(
            f"Hubo un problema analizando la foto de la factura. "
            f"Intenta de nuevo con una imagen más clara."
        )

    # Suma menor que el total real → agregar diferencia en categoría del comercio
    if diferencia > tolerancia:
        cat_ajuste = cat_comercio if cat_comercio in CATEGORIAS_VALIDAS else "CANASTA"
        #logger.info("agregando diff a comercioo:\n%s", cat_ajuste)
        categorias[cat_ajuste] = round(categorias.get(cat_ajuste, 0.0) + diferencia, 2)
        logger.info(
            "Diferencia $%s agregada a %s (calculado: $%s → real: $%s)",
            f"{diferencia:,.0f}", cat_ajuste,
            f"{total_calculado:,.0f}", f"{total_real:,.0f}",
        )
    
    logger.info("aqui devuelve categorias")
    return categorias


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request
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
    Prefijo: /luka/

    Endpoints:
      POST /luka/categorizar-factura-texto
      POST /luka/categorizar-factura-imagen
      POST /luka/categorizar-gasto-manual
    """

    prefix = "/luka"

    def _registrar_rutas(self) -> None:

        @self.router.post("/categorizar-factura-texto")
        def categorizar_factura_texto(req: TextoRequest):
            if not req.texto.strip():
                raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")
            try:
                prompt = PROMPT_TEXTO_DIRECTO.format(contenido=req.texto.strip())
                raw    = self.motor.texto(prompt, max_tokens=1200)
                data   = self.motor.extraer_json(raw)
                items, comercio, fecha, total_real, cat_comercio = _extraer_resultado(data)
                categorias = _agrupar_por_categoria(items, total_real, cat_comercio)
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
                raw = self.motor.imagen(
                    PROMPT_IMAGEN_DIRECTA,
                    req.imagen_b64,
                    max_tokens=2400,
                )
                #logger.info("TRANSCRIPCIÓN:\n%s", raw)
                data  = self.motor.extraer_json(raw)
                #logger.info("DATA:\n%s", data)
                items, comercio, fecha, total_real, cat_comercio = _extraer_resultado(data)
                categorias = _agrupar_por_categoria(items, total_real, cat_comercio)
                logger.info("cATEGORIAS:\n%s", categorias)
                return {
                    "categorias":    categorias,
                    "comercio":      comercio,
                    "fecha":         fecha,
                    "total_factura": round(sum(categorias.values()), 2),
                }
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
                        max_tokens=1200,
                    )
                )
                return _validar_gastos_manuales(data)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))


# luka_ai_router.py no tiene punto de entrada.
# El servidor se levanta desde main.py — ver ~/projects/AIBase/main.py