from concurrent.futures import ThreadPoolExecutor
import time
import easyocr

from src.config import (
    CAMERA_ID,
    USE_GPU,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    BUCKET_NAME,
    API_URL,
    OCR_CONF_MINIMA_SUCESSO,
)
from src.core.logger import configurar_logger
from src.core.use_cases import ProcessarEventoUseCase
from src.core.task_limiter import LimitedExecutor
from src.services.ia_service import ONNXDetector
from src.services.ocr_service import EasyOCRReader
from src.services.storage_service import MinIOStorage
from src.services.api_service import FastAPIClient
from src.services.redis_service import (
    aguardar_evento,
    conectar_com_retry,
    confirmar_evento,
    recuperar_eventos_orfaos,
)

logger = configurar_logger("VisionCoreWorker")

def processar_e_confirmar(use_case, cliente_redis, evento, mensagem_bruta):
    """Processa o evento e faz o ack (LREM) mesmo em caso de falha —
    um evento com erro irrecuperável não deve voltar para a fila em loop."""
    try:
        use_case.executar(evento)
    finally:
        confirmar_evento(cliente_redis, mensagem_bruta)

def main():
    logger.info("Inicializando VisionCore Worker Portaria.")
    
    # 1. Inicialização de IA e Modelos pesados
    logger.info("Carregando modelos de Deep Learning (YOLOv8 ONNX e EasyOCR).")
    try:
        detector = ONNXDetector("models/modelo_placas.onnx")
        leitor_ocr_interno = easyocr.Reader(['pt', 'en'], gpu=USE_GPU, model_storage_directory='models')
        ocr_reader = EasyOCRReader(leitor_ocr_interno)
        logger.info("Modelos de Inteligência Artificial carregados com sucesso.")
    except Exception as e:
        logger.critical(f"Falha fatal ao carregar os modelos de IA: {e}")
        return

    # 2. Inicialização de Infraestrutura de Storage e API
    storage = MinIOStorage(
        endpoint_url=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket_name=BUCKET_NAME
    )
    api_client = FastAPIClient(api_url=API_URL)

    # 3. Inicialização do Caso de Uso Core
    use_case = ProcessarEventoUseCase(
        detector=detector,
        ocr_reader=ocr_reader,
        storage=storage,
        api_client=api_client,
        camera_id_default=CAMERA_ID,
        conf_minima_sucesso=OCR_CONF_MINIMA_SUCESSO,
    )

    # 4. Conexão ao Broker de Eventos (Redis)
    try:
        cliente_redis = conectar_com_retry()
    except Exception as e:
        logger.critical(f"Falha fatal ao conectar ao Broker Redis: {e}")
        return

    logger.info("Worker em modo escuta. Aguardando eventos da fila Redis.")

    # Sobra de crash anterior: eventos que estavam em processamento voltam à fila
    recuperar_eventos_orfaos(cliente_redis)

    # 5. Execução Concorrente via ThreadPoolExecutor com backpressure
    # 4 threads concorrentes aproveitam o paralelismo C++ do ONNX Runtime e evitam travar em I/O.
    # O LimitedExecutor segura o BLMOVE quando o pool está cheio; eventos em
    # voo ficam retidos na lista :processing até o ack, então um crash não os perde.
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="WorkerThread") as executor:
        executor_limitado = LimitedExecutor(executor, max_in_flight=max_workers * 2)
        try:
            while True:
                try:
                    resultado = aguardar_evento(cliente_redis, timeout=5)
                    if resultado is None:
                        continue
                    evento, mensagem_bruta = resultado

                    # Submete o processamento do evento para o pool de threads
                    executor_limitado.submit(
                        processar_e_confirmar, use_case, cliente_redis, evento, mensagem_bruta
                    )
                except Exception as e:
                    logger.error(f"Erro inesperado ao gerenciar fila no loop principal: {e}")
                    time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("Sinal de interrupção recebido. Iniciando encerramento gracioso.")
        finally:
            logger.info("Aguardando finalização das threads de processamento ativas.")

if __name__ == "__main__":
    main()