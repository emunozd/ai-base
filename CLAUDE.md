# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arquitectura del Proyecto

**Kingsrow AI** es un servidor de inferencia IA centralizado que carga un modelo MLX una sola vez en memoria RAM y lo comparte entre múltiples \"proyectos\" (routers).

### Estructura de archivos

- **kingsrow_ai_base.py** - Motor base compartido: singleton del modelo MLX, endpoints base (/health, /v1/messages, /v1/chat/completions), clase BaseRouter para routers hijos
- **main.py** - ÚNICO punto de entrada del servidor. Arranca el servidor y registra todos los routers
- **luka_ai_router.py** - Router de LUKA: endpoint de clasificación de facturas (texto, imagen, gasto manual)

### Patrón de diseño

```
main.py (LaunchDaemon)
    ↓ carga KingsrowAI()
    ↓ registra LukaRouter (y otros routers en el futuro)
    ↓ build() + run()
        ↓ _ModeloMLX.cargar() - singleton MLX (19GB en RAM)
        ↓ endpoints base + endpoints de cada router
```

### Para agregar un nuevo proyecto

1. Crea `otro_proyecto_router.py` con una clase que extienda `BaseRouter`
2. Importa la clase en `main.py`
3. Llama a `server.registrar(OtroProyectoRouter)` en `main.py`
4. Reinicia el servidor - el modelo NO se recarga, ya está en RAM

## Comandos de Desarrollo

### Arrancar el servidor

```bash
source ~/mlx-env/bin/activate
python main.py
```

### Variables de entorno (opcionales)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `KR_MODEL_PATH` | `mlx-community/Qwen3.5-35B-A3B-4bit` | Ruta del modelo MLX |
| `KR_HOST` | `127.0.0.1` | IP de escucha |
| `KR_PORT` | `8181` | Puerto |
| `KR_IMG_MAX` | `1024` | Tamaño máximo de imagen en píxeles |
| `KR_API_KEY` | `` | API key requerida en header `X-API-Key` |
| `KR_MAX_TOKENS_CHAT` | `8192` | Máximo tokens para chat |
| `KR_MAX_TOKENS_OPENAI` | `4096` | Máximo tokens para endpoint OpenAI |
| `KR_MAX_TOKENS_LUKA` | `600` | Máximo tokens para LUKA |

## Rutas Disponibles

### Endpoints base

- `GET /health` - Health check
- `GET /v1/models` - List models
- `POST /v1/messages` - Anthropic-compatible (Claude Code)
- `POST /v1/chat/completions` - OpenAI-compatible (AnythingLLM)

### Rutas de LUKA

- `POST /luka/categorizar-factura-texto` - Clasifica texto de factura
- `POST /luka/categorizar-factura-imagen` - Clasifica imagen de factura
- `POST /luka/categorizar-gasto-manual` - Clasifica descripción de gasto

## Dependencias

```bash
pip install fastapi uvicorn mlx-vlm pillow duckduckgo-search httpx beautifulsoup4
```

## Endpoints de LUKA

### Categorías válidas

HOGAR, CANASTA, MEDICAMENTOS, OCIO, ANTOJO, TRANSPORTE, TECNOLOGÍA, ROPA, EDUCACIÓN, MASCOTAS

### Ejemplos de uso

```bash
# Clasificar factura por texto
curl -X POST http://KR_HOST:KR_PORT/luka/categorizar-factura-texto \\
  -H \"Content-Type: application/json\" \\
  -d '{\"texto\": \"Compre 2 panes a 2000 cada uno\"}'

# Clasificar factura por imagen
curl -X POST http://KR_HOST:KR_PORT/luka/categorizar-factura-imagen \\
  -H \"Content-Type: application/json\" \\
  -d '{\"imagen_b64\": \"base64...\"}'

# Clasificar gasto manual
curl -X POST http://KR_HOST:KR_PORT/luka/categorizar-gasto-manual \\
  -H \"Content-Type: application/json\" \\
  -d '{\"descripcion\": \"Gasté 5k en el supermercado\"}'
```

## Notas Técnicas

- El modelo se carga una sola vez en el singleton `_ModeloMLX`
- Los routers hijos comparten el mismo modelo en memoria
- Búsqueda web automática via DuckDuckGo para queries que requieren datos actualizados
- Truncamiento inteligente del historial de mensajes cuando el contexto excede los tokens máximos
- Soporte para streaming SSE en ambos endpoints Anthropic y OpenAI
- Middleware de autenticación opcional con API key

## Archivos Clave

- **kingsrow_ai_base.py:79-96** - Singleton del modelo MLX
- **kingsrow_ai_base.py:515-571** - Clase MotorInferencia (API pública para routers)
- **kingsrow_ai_base.py:577-587** - Clase BaseRouter (abstracta)
- **kingsrow_ai_base.py:621-830** - Clase KingsrowAI (servidor principal)
- **luka_ai_router.py:167-231** - Router LUKA con sus 3 endpoints
- **main.py:30-35** - Punto de entrada único

## Limitaciones

- El modelo Qwen3.5-35B-A3B-4bit empieza a degradarse por encima de ~20k tokens de entrada
- Las URLs con años anteriores a 2025 en el path son rechazadas automáticamente
- Las queries de fecha/hora no se buscan (el servidor ya las tiene disponibles)
"}