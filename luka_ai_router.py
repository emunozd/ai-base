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

Estrategia de categorización de facturas con imagen:
    Pasada 1 → modelo transcribe la imagen literalmente (visión pura)
    Python   → parsea ítems y aplica descuentos (aritmética exacta)
    Pasada 2 → modelo clasifica cada ítem por nombre (sin tocar montos)
    Python   → une montos exactos con categorías y agrupa totales

Para facturas de texto el flujo es más directo:
    Python   → parsea ítems y descuentos del texto
    Pasada 1 → modelo clasifica cada ítem por nombre
    Python   → agrupa totales por categoría

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
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TRANSCRIBIR = """Transcribe EXACTAMENTE el texto de esta factura o recibo.
Incluye todos los artículos con sus cantidades y valores.
Mantén los números exactamente como aparecen: no redondees, no interpretes, no calcules.
Conserva los puntos y comas tal cual están en la imagen.
Solo transcribe — no clasifiques ni expliques nada."""

PROMPT_CLASIFICAR_ITEMS = """Analiza el texto de esta factura. Para cada artículo devuelve nombre, monto, descuento y categoría.

CATEGORÍAS DISPONIBLES (usa exactamente estos nombres):
HOGAR, HOGAR_ARRIENDO, HOGAR_SERVICIOS, HOGAR_REPARACIONES,
CANASTA, CANASTA_VERDURAS, CANASTA_PROTEINA, CANASTA_ASEO, CANASTA_HIGIENE,
MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

REGLAS DE MONTOS — MUY IMPORTANTE:
- El punto (.) es separador de miles. 42.668 = 42668. NUNCA es decimal.
- "monto" es el valor del artículo tal como aparece en la factura, como número entero sin separadores.
- "descuento" es el valor que aparece en la línea "Descuento XX% VALOR-" que sigue al artículo. Si no hay descuento usa 0.
- NO restes nada — devuelve monto y descuento por separado. Python hará la resta.

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
{productos}

Devuelve ÚNICAMENTE este JSON, sin explicaciones:
[
  {{"descripcion": "nombre del artículo", "monto": valor_entero, "descuento": valor_entero_o_0, "categoria": "NOMBRE_CATEGORIA"}},
  ...
]"""

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
# Parser de texto de factura — Python hace la aritmética
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_texto_factura(texto: str) -> tuple[list[dict], str | None, str | None]:
    lineas = texto.strip().split("\n")
    items = []
    comercio = "Comercio Desconocido"
    fecha = None

    # 1. Regex específicos
    # Busca: Inicio de línea con número + cualquier cosa + espacio + Monto (X.XXX o XX.XXX)
    pat_item = re.compile(r"^\d+\s+.*?\s+([\d]{1,3}(?:\.[\d]{3})+)$")
    pat_desc = re.compile(r"[Dd]escuento\s+[\d,\.]+\s*%\s+([\d\.]+)-?")
    
    # 2. Detectar Comercio y Fecha
    for linea in lineas[:15]:
        linea_s = linea.strip()
        if not fecha:
            m_f = re.search(r"(\d{4}[/\-]\d{2}[/\-]\d{2})", linea_s)
            if m_f: fecha = m_f.group(1).replace("/", "-")
        
        # El comercio suele ser la primera línea relevante que no sea "Factura..."
        if comercio == "Comercio Desconocido" and len(linea_s) > 5:
            if not re.search(r"(Factura|Pedido|Caja|Fecha|Cliente|Direcc|#)", linea_s, re.I):
                comercio = linea_s

    item_actual = None

    for linea in lineas:
        linea_s = linea.strip()
        if not linea_s or any(x in linea_s for x in ["Subtotal", "Total IVA", "Valor Total"]):
            continue

        # ¿Es una línea de ítem? (Ej: 8 240062700000 0 0 6,86 115.316)
        m_item = pat_item.match(linea_s)
        if m_item:
            if item_actual: items.append(item_actual)
            
            monto = int(m_item.group(1).replace(".", ""))
            item_actual = {"descripcion": "", "monto": monto}
            continue

        # ¿Es una línea de descuento?
        m_desc = pat_desc.search(linea_s)
        if m_desc and item_actual:
            desc_val = int(m_desc.group(1).replace(".", ""))
            item_actual["monto"] = max(0, item_actual["monto"] - desc_val)
            continue

        # Si tenemos un ítem pero no tiene descripción, esta línea lo es
        if item_actual and not item_actual["descripcion"]:
            # Filtramos que no sea una línea de puros números o etiquetas conocidas
            if not re.match(r"^[0-9\s,.-]+$", linea_s) and len(linea_s) > 2:
                item_actual["descripcion"] = linea_s

    if item_actual: items.append(item_actual)

    return items, comercio, fecha


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de validación y procesamiento
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


def _agrupar_por_categoria(clasificaciones: list[dict]) -> dict[str, float]:
    """
    Recibe ítems con monto, descuento y categoria del modelo.
    Python aplica descuento y agrupa totales por categoría.
    """
    categorias: dict[str, float] = {}
    for item in clasificaciones:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("categoria", "")).upper().strip()
        if cat not in CATEGORIAS_VALIDAS:
            cat = "CANASTA"
        try:
            monto     = float(item.get("monto", 0) or 0)
            descuento = float(item.get("descuento", 0) or 0)
            neto      = round(monto - descuento, 2)
            if neto <= 0:
                continue
        except (TypeError, ValueError):
            continue
        categorias[cat] = round(categorias.get(cat, 0.0) + neto, 2)
    return categorias


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
                # Python parsea el texto y aplica descuentos exactamente
                items_con_monto, comercio, fecha = _parsear_texto_factura(req.texto.strip())
                if not items_con_monto:
                    raise ValueError("No se pudieron extraer ítems del texto.")

                # Modelo solo clasifica por nombre
                lista_productos = "\n".join(
                    f"- {item['descripcion']}" for item in items_con_monto
                )
                prompt = PROMPT_CLASIFICAR_ITEMS.format(productos=lista_productos)
                raw    = self.motor.texto(prompt, max_tokens=800)
                data   = self.motor.extraer_json(raw)
                if not isinstance(data, list):
                    raise ValueError("El modelo no devolvió una lista de clasificaciones.")

                categorias = _agrupar_por_categoria(data)
                if not categorias:
                    raise ValueError("No se pudieron clasificar los ítems.")

                return {
                    "categorias":    categorias,
                    "comercio":      None,
                    "fecha":         None,
                    "total_factura": round(sum(categorias.values()), 2),
                }
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @self.router.post("/categorizar-factura-imagen")
        def categorizar_factura_imagen(req: ImagenRequest):
            if not req.imagen_b64.strip():
                raise HTTPException(status_code=422, detail="La imagen no puede estar vacía.")
            try:
                # Pasada 1 — transcripción literal de la imagen
                texto_transcrito = self.motor.imagen(
                    PROMPT_TRANSCRIBIR,
                    req.imagen_b64,
                    max_tokens=2400,
                )
                if not texto_transcrito.strip():
                    raise ValueError("No se pudo transcribir el contenido de la imagen.")
                logger.info("TRANSCRIPCIÓN PASADA 1:\n%s", texto_transcrito)

                # Python parsea el texto transcrito y aplica descuentos exactamente
                items_con_monto, comercio, fecha = _parsear_texto_factura(texto_transcrito)
                if not items_con_monto:
                    raise ValueError("No se pudieron extraer ítems de la imagen.")

                #logger.info("Ítems parseados: %d — total Python: $%s", len(items_con_monto), f"{sum(i['monto'] for i in items_con_monto):,.0f}")

                # Pasada 2 — modelo solo clasifica por nombre de producto
                lista_productos = "\n".join(
                    f"- {item['descripcion']}" for item in items_con_monto
                )
                prompt = PROMPT_CLASIFICAR_ITEMS.format(productos=texto_transcrito.strip())
                raw    = self.motor.texto(prompt, max_tokens=800)
                logger.info("CLASIFICACIÓN PASADA 2:\n%s", raw)
                data   = self.motor.extraer_json(raw)
                if not isinstance(data, list):
                    raise ValueError("El modelo no devolvió una lista de clasificaciones.")

                categorias = _agrupar_por_categoria(data)
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