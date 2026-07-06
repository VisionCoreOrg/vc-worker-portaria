from typing import Tuple
import numpy as np
import easyocr

from src.core.logger import configurar_logger
from src.utils.image_utils import variantes_para_ocr

logger = configurar_logger("EasyOCRReader")

# Placas BR só contêm A-Z e 0-9; restringir o vocabulário do EasyOCR elimina
# pontuação, minúsculas e Unicode (ex.: 'ª') na origem.
ALLOWLIST_PLACA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Caixas com menos caracteres que isso são quase sempre ruído (parafusos,
# moldura); participam da leitura concatenada mas não viram candidata própria.
MIN_CHARS_CANDIDATA = 5


class EasyOCRReader:
    """
    Implementa o leitor OCR usando a biblioteca EasyOCR sobre múltiplas
    variantes de pré-processamento. Conforme com o Protocol 'OCRReader'.
    """

    def __init__(self, leitor: easyocr.Reader):
        self.leitor = leitor

    def ler_texto(self, crop: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Executa OCR sobre as variantes do recorte e devolve leituras candidatas.
        Retorna:
            (leituras, imagem_binarizada) — leituras é uma lista de tuplas
            (texto_cru, confianca_ocr); a escolha da melhor leitura é regra
            de domínio (escolher_leitura) e vive no caso de uso.
        """
        variantes = variantes_para_ocr(crop)
        img_binarizada = dict(variantes)["otsu"]

        leituras: list[tuple[str, float]] = []
        for nome, img in variantes:
            try:
                resultado = self.leitor.readtext(img, allowlist=ALLOWLIST_PLACA)
            except Exception as e:
                logger.error(f"Falha no EasyOCR na variante '{nome}': {e}")
                continue
            if not resultado:
                continue

            # O EasyOCR não garante ordem de leitura entre caixas — ordenar
            # da esquerda para a direita antes de concatenar.
            caixas = sorted(resultado, key=lambda r: min(p[0] for p in r[0]))
            concatenado = "".join(texto for _, texto, _ in caixas)
            # A confiança da leitura concatenada deve refletir o corpo da placa,
            # não o mínimo global (caixas de ruído como "BRASIL"/moldura deprimem
            # leituras corretas para 'revisar'). Usamos a conf da MAIOR caixa por
            # nº de caracteres — a que provavelmente é a placa; empate → primeira.
            caixa_maior = max(caixas, key=lambda r: len(r[1]))
            conf_leitura = float(caixa_maior[2])
            leituras.append((concatenado, conf_leitura))

            if len(caixas) > 1:
                for _, texto, conf in caixas:
                    if len(texto) >= MIN_CHARS_CANDIDATA:
                        leituras.append((texto, float(conf)))

        return leituras, img_binarizada
