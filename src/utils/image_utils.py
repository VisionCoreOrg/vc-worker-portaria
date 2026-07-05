import cv2
import numpy as np

def letterbox(img: np.ndarray, tamanho: int = 640):
    """
    Redimensiona a imagem mantendo o aspect ratio e adiciona padding cinza
    até atingir (tamanho x tamanho). Retorna também os parâmetros para
    reverter as coordenadas de volta ao espaço original.
    """
    h, w = img.shape[:2]
    escala = tamanho / max(h, w)
    novo_w = int(w * escala)
    novo_h = int(h * escala)

    img_redim = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)

    tela = np.full((tamanho, tamanho, 3), 114, dtype=np.uint8)
    pad_top = (tamanho - novo_h) // 2
    pad_left = (tamanho - novo_w) // 2
    tela[pad_top:pad_top + novo_h, pad_left:pad_left + novo_w] = img_redim

    return tela, escala, pad_top, pad_left


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Non-Maximum Suppression simples implementado em numpy."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    ordem = scores.argsort()[::-1]

    mantidos = []
    while ordem.size > 0:
        i = ordem[0]
        mantidos.append(i)

        xx1 = np.maximum(x1[i], x1[ordem[1:]])
        yy1 = np.maximum(y1[i], y1[ordem[1:]])
        xx2 = np.minimum(x2[i], x2[ordem[1:]])
        yy2 = np.minimum(y2[i], y2[ordem[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersecao = inter_w * inter_h
        iou = intersecao / (areas[i] + areas[ordem[1:]] - intersecao)

        ordem = ordem[np.where(iou <= iou_threshold)[0] + 1]

    return mantidos


def pre_processar_imagem_ocr(img: np.ndarray) -> np.ndarray:
    """Aplica grayscaling, CLAHE, resizing 2x, bilateralFilter e binarização Otsu para melhorar OCR."""
    # 1. Conversão para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE para normalizar iluminação e sombras de forma adaptativa local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 3. Ampliação da imagem (2x) para melhorar a leitura de caracteres pequenos
    largura = int(gray_clahe.shape[1] * 2)
    altura = int(gray_clahe.shape[0] * 2)
    ampliada = cv2.resize(
        gray_clahe, (largura, altura), interpolation=cv2.INTER_CUBIC
    )

    # 4. Filtro Bilateral para suavizar ruído e sujeira sem borrar as bordas dos caracteres
    suave = cv2.bilateralFilter(ampliada, d=5, sigmaColor=75, sigmaSpace=75)

    # 5. Binarização de Otsu
    _, binarizada = cv2.threshold(
        suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binarizada


def variantes_para_ocr(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Gera variantes de pré-processamento do crop para OCR multi-variante.

    A binarização de Otsu global destrói placas legíveis sob iluminação
    irregular e escolhe a polaridade sozinha conforme o histograma; por isso
    o OCR roda também na variante em cinza (sem threshold) e na binarizada
    invertida — a melhor leitura é escolhida no domínio (escolher_leitura).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    largura = int(gray_clahe.shape[1] * 2)
    altura = int(gray_clahe.shape[0] * 2)
    ampliada = cv2.resize(gray_clahe, (largura, altura), interpolation=cv2.INTER_CUBIC)

    suave = cv2.bilateralFilter(ampliada, d=5, sigmaColor=75, sigmaSpace=75)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        ("cinza_clahe", suave),
        ("otsu", otsu),
        ("otsu_invertida", cv2.bitwise_not(otsu)),
    ]
