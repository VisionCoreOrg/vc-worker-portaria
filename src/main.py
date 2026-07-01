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
)
from src.core.logger import configurar_logger
from src.core.use_cases import ProcessarEventoUseCase
from src.core.task_limiter import LimitedExecutor
from src.services.ia_service import ONNXDetector
from src.services.ocr_service import EasyOCRReader
from src.services.storage_service import MinIOStorage
from src.services.api_service import FastAPIClient
from src.services.redis_service import conectar_com_retry, aguardar_evento

logger = configurar_logger("VisionCoreWorker")

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
        camera_id_default=CAMERA_ID
    )

    # 4. Conexão ao Broker de Eventos (Redis)
    try:
        cliente_redis = conectar_com_retry()
    except Exception as e:
        logger.critical(f"Falha fatal ao conectar ao Broker Redis: {e}")
        return

    logger.info("Worker em modo escuta. Aguardando eventos da fila Redis.")

    # 5. Execução Concorrente via ThreadPoolExecutor com backpressure
    # 4 threads concorrentes aproveitam o paralelismo C++ do ONNX Runtime e evitam travar em I/O.
    # O LimitedExecutor segura o BRPOP quando o pool está cheio, evitando
    # acumular em memória eventos já removidos do Redis.
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="WorkerThread") as executor:
        executor_limitado = LimitedExecutor(executor, max_in_flight=max_workers * 2)
        try:
            while True:
                try:
                    evento = aguardar_evento(cliente_redis, timeout=5)
                    if evento is None:
                        continue

                    # Submete o processamento do evento para o pool de threads
                    executor_limitado.submit(use_case.executar, evento)
                except Exception as e:
                    logger.error(f"Erro inesperado ao gerenciar fila no loop principal: {e}")
                    time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("Sinal de interrupção recebido. Iniciando encerramento gracioso.")
        finally:
            logger.info("Aguardando finalização das threads de processamento ativas.")

if __name__ == "__main__":
    main()