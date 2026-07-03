from typing import Tuple
import numpy as np
import easyocr

from src.core.logger import configurar_logger
from src.utils.image_utils import pre_processar_imagem_ocr

logger = configurar_logger("EasyOCRReader")

class EasyOCRReader:
    """
    Implementa o leitor OCR usando a biblioteca EasyOCR.
    Conforme com o Protocol 'OCRReader'.
    """

    def __init__(self, leitor: easyocr.Reader):
        self.leitor = leitor

    def ler_texto(self, crop: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Realiza pré-processamento OpenCV e leitura OCR sobre o recorte da placa.
        Retorna:
            (texto_bruto, imagem_binarizada).
        """
        # Pré-processamento OpenCV via Image Utils
        img_binarizada = pre_processar_imagem_ocr(crop)

        # Leitura via EasyOCR
        try:
            resultado_ocr = self.leitor.readtext(img_binarizada)
        except Exception as e:
            logger.error(f"Falha ao realizar inferência no EasyOCR: {e}")
            return "", img_binarizada

        texto_bruto = ""
        for (bbox, texto, conf_ocr) in resultado_ocr:
            texto_bruto += texto

        return texto_bruto, img_binarizada
