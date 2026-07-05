import time
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np

from src.core.interfaces import Detector, OCRReader, StorageRepository, APIClient
from src.core.logger import configurar_logger
from src.core.text_utils import decidir_status, escolher_leitura

logger = configurar_logger("ProcessarEventoUseCase")

class ProcessarEventoUseCase:
    """
    Caso de Uso central que orquestra o processamento do evento de detecção de veículos.
    Livre de dependências de infraestrutura concreta (boto3, requests, easyocr, redis).
    """

    def __init__(
        self,
        detector: Detector,
        ocr_reader: OCRReader,
        storage: StorageRepository,
        api_client: APIClient,
        camera_id_default: str = "camera_default",
        conf_minima_sucesso: float = 0.5,
    ):
        self.detector = detector
        self.ocr_reader = ocr_reader
        self.storage = storage
        self.api_client = api_client
        self.camera_id_default = camera_id_default
        self.conf_minima_sucesso = conf_minima_sucesso

    def executar(self, evento: dict) -> None:
        """
        Processa um evento oriundo da fila contendo metadados do frame.
        """
        chave_arquivo = evento.get("path")
        camera_id_evento = evento.get("camera_id", self.camera_id_default)

        if not chave_arquivo:
            logger.warning(f"Evento sem campo 'path' ignorado: {evento}")
            return

        logger.info(f"Iniciando processamento do arquivo: {chave_arquivo} (Camera: {camera_id_evento})")

        # 1. Download do frame original
        imagem_original = self.storage.baixar_imagem(chave_arquivo)
        if imagem_original is None:
            logger.error(f"Falha ao baixar imagem '{chave_arquivo}'. Evento descartado.")
            return

        # 2. Inferência de Visão Computacional (YOLOv8)
        placa_crop, confianca_yolo = self.detector.detectar(imagem_original)
        if placa_crop is None:
            logger.info(f"Nenhuma placa de veículo detectada na imagem '{chave_arquivo}'.")
            return

        # 3. OCR multi-variante (adapter devolve leituras cruas candidatas)
        leituras, img_binarizada = self.ocr_reader.ler_texto(placa_crop)

        # 4. Regras de Domínio: melhor leitura + validação do formato BR
        decisao = escolher_leitura(leituras)

        status, motivo_filtro = decidir_status(decisao, self.conf_minima_sucesso)

        placa_salvar = decisao.placa if decisao.placa else "—"

        logger.info(
            f"Filtro Aplicado -> [{status.upper()}] Placa final: {placa_salvar} "
            f"(YOLO: {confianca_yolo:.2f} | OCR: {decisao.confianca_ocr:.2f})"
        )

        # 5. Upload das mídias com resiliência de retentativas curtas
        url_recorte = self._upload_com_retry(placa_crop, placa_salvar, sufixo="")
        if not url_recorte:
            logger.warning(f"Falha no upload do recorte colorido após retentativas. Prosseguindo sem mídia.")
            # Definimos uma flag para registrar na API que houve falha de mídia
            falha_midia = True
        else:
            falha_midia = False

        url_binarizada = self._upload_com_retry(img_binarizada, placa_salvar, sufixo="bin")
        if not url_binarizada:
            logger.warning(f"Falha no upload do recorte binarizado secundário.")

        # 6. Consolidação do Payload de Domínio
        payload = {
            "camera_id": camera_id_evento,
            "arquivo_origem": chave_arquivo,
            "placa": placa_salvar,
            "confianca": round(float(confianca_yolo), 4),
            "confianca_ocr": round(float(decisao.confianca_ocr), 4),
            "imagem_url": url_recorte if url_recorte else "",
            "imagem_processada_url": url_binarizada if url_binarizada else "",
            "status": status,
            "motivo_filtro": motivo_filtro,
            "falha_midia": falha_midia,
            "data_hora": datetime.now(timezone.utc).isoformat(),
        }

        # 7. Registro na API Core com resiliência
        sucesso_envio = self._registrar_com_retry(payload)
        if sucesso_envio:
            logger.info(f"Registro da placa '{placa_salvar}' enviado com sucesso para a API Core.")
        else:
            logger.error(f"Erro fatal: Falha no envio do registro da placa '{placa_salvar}' para a API após retentativas.")

    def _upload_com_retry(
        self,
        imagem: np.ndarray,
        placa: str,
        sufixo: str = "",
        max_tentativas: int = 3,
        delay: float = 0.5
    ) -> Optional[str]:
        """Tenta fazer upload da imagem no storage com retry exponencial simples."""
        for tentativa in range(1, max_tentativas + 1):
            try:
                url = self.storage.upload_recorte(imagem, placa, sufixo)
                if url:
                    return url
            except Exception as e:
                logger.warning(f"Tentativa {tentativa}/{max_tentativas} - Erro ao fazer upload de mídia: {e}")
            if tentativa < max_tentativas:
                time.sleep(delay * tentativa)
        return None

    def _registrar_com_retry(
        self,
        payload: dict,
        max_tentativas: int = 3,
        delay: float = 1.0
    ) -> bool:
        """Tenta enviar o payload para a API com retry exponencial simples."""
        for tentativa in range(1, max_tentativas + 1):
            try:
                if self.api_client.registrar_passagem(payload):
                    return True
            except Exception as e:
                logger.warning(f"Tentativa {tentativa}/{max_tentativas} - Erro ao enviar registro para a API Core: {e}")
            if tentativa < max_tentativas:
                time.sleep(delay * tentativa)
        return False
