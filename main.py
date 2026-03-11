"""
main.py — Punto de entrada único del servidor Kingsrow AI
==========================================================
Este es el ÚNICO archivo que arranca el servidor y el ÚNICO LaunchDaemon.
Un proceso → un modelo en RAM (19 GB) → todos los proyectos.

Arranque manual:
    source ~/mlx-env/bin/activate
    python ~/projects/AIBase/main.py

Variables de entorno disponibles (todas opcionales):
    KR_MODEL_PATH   ruta/nombre del modelo MLX  (default: mlx-community/Qwen3.5-35B-A3B-4bit)
    KR_HOST         IP de escucha               (default: 192.168.0.90)
    KR_PORT         puerto                      (default: 8181)
    KR_IMG_MAX      tamaño máximo de imagen px  (default: 1024)
    KR_API_KEY      API key requerida en header  X-API-Key (default: vacío = sin auth)

Para agregar un proyecto nuevo:
    1. Crea ~/projects/AIBase/otro_proyecto_router.py extendiendo BaseRouter.
    2. Importa la clase aquí y agrégala con server.registrar().
    3. Reinicia el LaunchDaemon — el modelo no se recarga, ya está en RAM.
"""

from kingsrow_ai_base import KingsrowAI
from luka_ai_router import LukaRouter

# ── Registra aquí todos los proyectos ────────────────────────────────────────
# from otro_proyecto_router import OtroProyectoRouter

if __name__ == "__main__":
    server = KingsrowAI()
    server.registrar(LukaRouter)
    # server.registrar(OtroProyectoRouter)   # descomentar cuando llegue
    server.build()
    server.run()

