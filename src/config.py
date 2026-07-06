import os

from dotenv import load_dotenv

load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://visioncore_minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "plate-bucket")

API_URL = os.getenv("API_URL")
CAMERA_ID = os.getenv("CAMERA_ID", "camera_default")

REDIS_HOST = os.getenv("REDIS_HOST", "parking_redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "camera:portaria:queue")

# ===========================================================================
# GPU
# GPU_PROVIDER controla qual Execution Provider o onnxruntime usará.
# Valores aceitos: "none" (CPU puro, padrão) | "nvidia" (CUDAExecutionProvider)
# USE_GPU é repassado ao easyocr para ativar/desativar aceleração via torch-CUDA.
# ===========================================================================
GPU_PROVIDER = os.getenv("GPU_PROVIDER", "none").lower()
USE_GPU = GPU_PROVIDER == "nvidia"

# ===========================================================================
# OCR — Fase 1 do plano de confiabilidade
# Leituras em formato BR válido com confiança abaixo deste limiar recebem
# status "revisar" (fila de anotação humana) em vez de "sucesso".
# ===========================================================================
OCR_CONF_MINIMA_SUCESSO = float(os.getenv("OCR_CONF_MINIMA_SUCESSO", "0.5"))

# ===========================================================================
# Detecção — modelo YOLOv8 (ONNX) usado pelo ONNXDetector.
# Default: yasirfaizahmed/license-plate-object-detection (Apache-2.0), que no
# gate de 43 imagens supera o modelo anterior (Koushim) — estrita 46,5% vs
# 37,2%, CER 20,9% vs 29,6%, sucessos-errados 3 vs 7, sem_deteccao 2 vs 5.
# Configurável para permitir troca de detector (ex.: fallback chain da Fase 2)
# sem alterar código. Ver relatorios/2026-07-05-confiabilidade-lpr/.
# ===========================================================================
MODELO_PLACAS_PATH = os.getenv("MODELO_PLACAS_PATH", "models/modelo_placas_yasir.onnx")
