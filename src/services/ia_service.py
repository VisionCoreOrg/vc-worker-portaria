import os
from typing import Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from src.config import GPU_PROVIDER
from src.core.logger import configurar_logger
from src.utils.image_utils import letterbox, nms

logger = configurar_logger("ONNXDetector")

class ONNXDetector:
    """
    Implementa o detector de placas usando o modelo YOLOv8 no ONNX Runtime.
    Conforme com o Protocol 'Detector'.
    """

    def __init__(self, caminho_modelo: str):
        self.caminho_modelo = caminho_modelo
        self.sessao = self._carregar_modelo()
        self.nome_entrada = self.sessao.get_inputs()[0].name

    def _get_providers(self) -> list[str]:
        """Retorna a lista de Execution Providers conforme GPU_PROVIDER."""
        if GPU_PROVIDER == "nvidia":
            logger.info("Configurando ONNX Runtime para aceleração via NVIDIA GPU (CUDA)...")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _carregar_modelo(self) -> ort.InferenceSession:
        """Carrega o modelo ONNX e inicializa a sessão de inferência."""
        providers = self._get_providers()
        try:
            sessao = ort.InferenceSession(self.caminho_modelo, providers=providers)
            provider_ativo = sessao.get_providers()[0]
            logger.info(f"Sessão do ONNX inicializada com sucesso. Provider ativo: {provider_ativo}")
            return sessao
        except Exception as e:
            logger.critical(f"Falha crítica ao carregar o modelo ONNX no caminho '{self.caminho_modelo}': {e}")
            raise e

    def detectar(self, img: np.ndarray, conf_threshold: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """
        Detecta a região da placa na imagem original.
        Retorna:
            (recorte_da_placa, score_de_confiança) ou (None, 0.0) se nenhuma for encontrada.
        """
        h_orig, w_orig = img.shape[:2]

        # — Pré-processamento via Image Utils —
        img_lb, escala, pad_top, pad_left = letterbox(img)
        
        # BGR → RGB, float32 normalizado, HWC → NCHW
        entrada = img_lb[:, :, ::-1].astype(np.float32) / 255.0
        entrada = np.transpose(entrada, (2, 0, 1))[np.newaxis]

        # — Inferência no ONNX Runtime —
        try:
            saida = self.sessao.run(None, {self.nome_entrada: entrada})[0]  # [1, 5, 8400]
        except Exception as e:
            logger.error(f"Erro inesperado durante a execução do ONNX Runtime: {e}")
            return None, 0.0

        pred = saida[0]  # [5, 8400] — remove dimensão batch

        # Separa coordenadas e scores (suporta multi-classe pegando o máximo por anchor)
        coords = pred[:4]           # cx, cy, w, h
        class_scores = pred[4:]     
        scores = class_scores.max(axis=0)

        # — Filtro de confiança —
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

        # — Non-Maximum Suppression (NMS) via Image Utils —
        indices = nms(boxes, scores_filtrados)
        if not indices:
            return None, 0.0

        # Seleciona a detecção mais provável (melhor score pós-NMS)
        melhor = indices[0]
        bx1, by1, bx2, by2 = boxes[melhor]
        confianca = float(scores_filtrados[melhor])

        # — Converte coordenadas do letterbox de volta para a imagem original —
        bx1 = (bx1 - pad_left) / escala
        by1 = (by1 - pad_top) / escala
        bx2 = (bx2 - pad_left) / escala
        by2 = (by2 - pad_top) / escala

        # — Margem de expansão do crop (10% larg, 15% alt) para facilitar OCR —
        larg = bx2 - bx1
        alt  = by2 - by1
        bx1 = max(0, int(bx1 - larg * 0.10))
        by1 = max(0, int(by1 - alt  * 0.15))
        bx2 = min(w_orig, int(bx2 + larg * 0.10))
        by2 = min(h_orig, int(by2 + alt  * 0.15))

        recorte = img[by1:by2, bx1:bx2]
        return recorte, confianca