import time
import json
from datetime import datetime, timezone
from ultralytics import YOLO
import easyocr

from src.config import CAMERA_ID
from src.services.ia_service import detectar_placa
from src.services.ocr_service import ler_texto_placa
from src.services.storage_service import upload_imagem_s3, baixar_imagem_s3
from src.services.api_service import enviar_para_api
from src.services.redis_service import conectar_com_retry, aguardar_evento

print("⏳ Inicializando modelos pesados de IA...")
modelo_yolo = YOLO("models/modelo_placas.pt")
leitor_ocr = easyocr.Reader(['pt', 'en'], gpu=False)
print("✅ Modelos carregados!")


def processar_evento(evento: dict) -> None:
    """
    Processa um único evento recebido da fila Redis.
    Espera receber: { "path": "dataset/img.jpg", "camera_id": "...", "timestamp": "..." }
    """
    chave_arquivo = evento.get("path")
    camera_id_evento = evento.get("camera_id", CAMERA_ID)

    if not chave_arquivo:
        print(f"[WORKER] Evento sem campo 'path', ignorado: {evento}")
        return

    print(f"\n🚗 Processando: {chave_arquivo} (câmera: {camera_id_evento})")

    imagem_original = baixar_imagem_s3(chave_arquivo)
    if imagem_original is None:
        print("[WORKER] Falha ao baixar imagem, evento descartado.")
        return

    placa_crop, confianca_yolo = detectar_placa(imagem_original, modelo_yolo)
    if placa_crop is None:
        print("[WORKER] Nenhuma placa detectada na imagem.")
        return

    texto_placa = ler_texto_placa(placa_crop, leitor_ocr)
    if not texto_placa or len(texto_placa) != 7:
        print(f"[WORKER] OCR falhou ou placa fora do padrão: '{texto_placa}'")
        return

    print(f"✅ Placa lida: {texto_placa} (Confiança YOLO: {confianca_yolo:.2f})")

    url_recorte = upload_imagem_s3(placa_crop, texto_placa)
    if not url_recorte:
        print("[WORKER] Falha ao fazer upload do recorte no MinIO.")
        return

    payload = {
        "camera_id": camera_id_evento,
        "arquivo_origem": chave_arquivo,
        "placa": texto_placa,
        "confianca": round(confianca_yolo, 4),
        "imagem_url": url_recorte,
        "data_hora": datetime.now(timezone.utc).isoformat(),
    }

    enviar_para_api(payload)


def iniciar_loop_eventos() -> None:
    """
    Loop principal orientado a eventos.
    Aguarda mensagens via BLPOP na fila Redis e processa cada uma.
    """
    cliente_redis = conectar_com_retry()
    print("\n🎯 Worker em modo escuta. Aguardando eventos da fila Redis...\n")

    while True:
        try:
            evento = aguardar_evento(cliente_redis, timeout=5)
            if evento is None:
                # Timeout normal — sem mensagens no momento
                continue
            processar_evento(evento)
        except Exception as e:
            print(f"[WORKER] ⚠️  Erro inesperado no loop: {e}")
            print("[WORKER] Aguardando 3s antes de reiniciar o loop...")
            time.sleep(3)


if __name__ == "__main__":
    print("🚀 VisionCore Worker Portaria — Iniciado!")
    iniciar_loop_eventos()