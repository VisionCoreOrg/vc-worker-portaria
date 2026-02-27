import time
from datetime import datetime, timezone
from ultralytics import YOLO
import easyocr

from src.config import CAMERA_ID
from src.services.ia_service import detectar_placa
from src.services.ocr_service import ler_texto_placa
from src.services.storage_service import upload_imagem_s3, listar_imagens_s3, baixar_imagem_s3
from src.services.api_service import enviar_para_api

print("⏳ Inicializando modelos pesados de IA...")
modelo_yolo = YOLO("models/modelo_placas.pt")
leitor_ocr = easyocr.Reader(['pt', 'en'], gpu=False)
print("Modelos carregados!")

def processar_pipeline_nuvem():
    print("\n Buscando novas imagens no MinIO (pasta dataset/)...")
    arquivos_no_bucket = listar_imagens_s3("dataset/")
    
    if not arquivos_no_bucket:
        print("Nenhuma imagem encontrada na pasta 'dataset/'.")
        return

    print(f"📸 {len(arquivos_no_bucket)} imagens encontradas. Iniciando processamento...")
    print("-" * 50)

    for chave_arquivo in arquivos_no_bucket:
        print(f"\n🚗 Processando: {chave_arquivo}")
        
        imagem_original = baixar_imagem_s3(chave_arquivo)
        if imagem_original is None:
            continue

        placa_crop, confianca_yolo = detectar_placa(imagem_original, modelo_yolo)
        if placa_crop is None:
            print("Nenhuma placa detectada.")
            continue

        texto_placa = ler_texto_placa(placa_crop, leitor_ocr)
        if not texto_placa or len(texto_placa) != 7:
            print(f"OCR falhou ou placa fora do padrão: '{texto_placa}'")
            continue

        print(f"✅ Placa lida com sucesso: {texto_placa} (Confiança YOLO: {confianca_yolo:.2f})")

        url_recorte = upload_imagem_s3(placa_crop, texto_placa)
        if not url_recorte:
            continue

        payload = {
            "camera_id": CAMERA_ID,
            "arquivo_origem": chave_arquivo,
            "placa": texto_placa,
            "confianca": round(confianca_yolo, 4),
            "imagem_url": url_recorte,
            "data_hora": datetime.now(timezone.utc).isoformat()
        }
        
        enviar_para_api(payload)

if __name__ == "__main__":
    print("VisionCore Worker Iniciado!")
    
    processar_pipeline_nuvem()
    
    print("\n🎉 Processamento em lote finalizado.")