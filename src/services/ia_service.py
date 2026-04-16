import os
import cv2
import numpy as np
import onnxruntime as ort

from src.config import GPU_PROVIDER


def _get_providers() -> list[str]:
    """Retorna a lista de Execution Providers conforme GPU_PROVIDER."""
    if GPU_PROVIDER == "nvidia":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def carregar_modelo(caminho: str) -> ort.InferenceSession:
    """Carrega o modelo ONNX e retorna uma sessão de inferência."""
    providers = _get_providers()
    sessao = ort.InferenceSession(caminho, providers=providers)
    provider_ativo = sessao.get_providers()[0]
    print(f"[IA] Modelo ONNX carregado. Provider ativo: {provider_ativo}")
    return sessao


def _letterbox(img: np.ndarray, tamanho: int = 640):
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


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
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


def detectar_placa(
    img: np.ndarray,
    sessao: ort.InferenceSession,
    conf_threshold: float = 0.5,
):
    """
    Detecta a região da placa na imagem e retorna o recorte e a confiança.

    Assume saída YOLOv8 no formato ONNX: [1, 4+num_classes, num_anchors]
    onde as 4 primeiras linhas são (cx, cy, w, h) no espaço do letterbox.
    """
    h_orig, w_orig = img.shape[:2]

    # — Pré-processamento —
    img_lb, escala, pad_top, pad_left = _letterbox(img)
    # BGR → RGB, float32 normalizado, HWC → NCHW
    entrada = img_lb[:, :, ::-1].astype(np.float32) / 255.0
    entrada = np.transpose(entrada, (2, 0, 1))[np.newaxis]

    # — Inferência —
    nome_entrada = sessao.get_inputs()[0].name
    saida = sessao.run(None, {nome_entrada: entrada})[0]  # [1, 5, 8400]

    pred = saida[0]  # [5, 8400] — remove dimensão batch

    # Separa coordenadas e scores (suporta multi-classe pegando o máximo)
    coords = pred[:4]           # cx, cy, w, h — shape [4, 8400]
    class_scores = pred[4:]     # shape [num_classes, 8400]
    scores = class_scores.max(axis=0)  # score máximo por anchor

    # — Filtra por confiança —
    mask = scores >= conf_threshold
    if not mask.any():
        return None, 0.0

    cx = coords[0][mask]
    cy = coords[1][mask]
    w  = coords[2][mask]
    h  = coords[3][mask]
    scores_filtrados = scores[mask]

    # — cxcywh → xyxy (espaço letterbox 640x640) —
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # — NMS —
    indices = _nms(boxes, scores_filtrados)
    if not indices:
        return None, 0.0

    melhor = indices[0]
    bx1, by1, bx2, by2 = boxes[melhor]
    confianca = float(scores_filtrados[melhor])

    # — Converte coordenadas de volta para o espaço da imagem original —
    bx1 = (bx1 - pad_left) / escala
    by1 = (by1 - pad_top) / escala
    bx2 = (bx2 - pad_left) / escala
    by2 = (by2 - pad_top) / escala

    # — Margem em torno da detecção —
    larg = bx2 - bx1
    alt  = by2 - by1
    bx1 = max(0, int(bx1 - larg * 0.10))
    by1 = max(0, int(by1 - alt  * 0.15))
    bx2 = min(w_orig, int(bx2 + larg * 0.10))
    by2 = min(h_orig, int(by2 + alt  * 0.15))

    recorte = img[by1:by2, bx1:bx2]
    return recorte, confianca