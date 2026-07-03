from typing import Protocol, Tuple, Optional
import numpy as np

class Detector(Protocol):
    def detectar(self, imagem: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detecta a região da placa na imagem original.
        Retorna:
            (recorte_da_placa, score_de_confiança_yolo) ou (None, 0.0) se não detectada.
        """
        ...


class OCRReader(Protocol):
    def ler_texto(self, crop: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Realiza pré-processamento e leitura OCR sobre o recorte da placa.
        Retorna:
            (texto_bruto, imagem_binarizada_processada).
        A correção do padrão brasileiro é regra de domínio e vive no caso de
        uso — o adapter devolve o texto cru.
        """
        ...


class StorageRepository(Protocol):
    def baixar_imagem(self, chave_arquivo: str) -> Optional[np.ndarray]:
        """
        Baixa a imagem original do storage S3/MinIO direto para um array NumPy (OpenCV).
        Retorna:
            np.ndarray contendo a imagem em formato BGR, ou None se falhar.
        """
        ...

    def upload_recorte(self, imagem_numpy: np.ndarray, string_placa: str, sufixo: str = "") -> Optional[str]:
        """
        Realiza o upload direto de uma imagem NumPy da memória para o storage, retornando a URL pública.
        Retorna:
            URL pública/caminho absoluto do arquivo carregado, ou None se falhar.
        """
        ...


class APIClient(Protocol):
    def registrar_passagem(self, payload: dict) -> bool:
        """
        Envia os metadados do veículo e urls das mídias detectadas para a API Core.
        Retorna:
            True se enviado com sucesso, False em caso de falha de rede/API offline.
        """
        ...
