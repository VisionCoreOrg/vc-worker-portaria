"""Testes da decisão de status e montagem do payload com fakes dos Protocols."""

import numpy as np

from src.core.use_cases import ProcessarEventoUseCase


class FakeDetector:
    def detectar(self, imagem):
        return np.zeros((10, 30, 3), dtype=np.uint8), 0.9


class FakeOCR:
    def __init__(self, leituras):
        self._leituras = leituras

    def ler_texto(self, crop):
        return self._leituras, np.zeros((10, 30), dtype=np.uint8)


class FakeStorage:
    def baixar_imagem(self, chave):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def upload_recorte(self, imagem, placa, sufixo=""):
        return f"http://minio/{placa}{sufixo}.jpg"


class FakeAPI:
    def __init__(self):
        self.payloads = []

    def registrar_passagem(self, payload):
        self.payloads.append(payload)
        return True


def _executar(leituras, conf_minima=0.5):
    api = FakeAPI()
    caso = ProcessarEventoUseCase(
        detector=FakeDetector(),
        ocr_reader=FakeOCR(leituras),
        storage=FakeStorage(),
        api_client=api,
        conf_minima_sucesso=conf_minima,
    )
    caso.executar({"path": "dataset/1.jpg", "camera_id": "cam_teste"})
    assert len(api.payloads) == 1
    return api.payloads[0]


def test_leitura_valida_confiante_vira_sucesso():
    payload = _executar([("BRASILABC1234", 0.9)])
    assert payload["status"] == "sucesso"
    assert payload["placa"] == "ABC1234"
    assert payload["confianca_ocr"] == 0.9
    assert payload["motivo_filtro"] is None


def test_leitura_valida_com_conf_baixa_vira_revisar():
    payload = _executar([("ABC1234", 0.3)])
    assert payload["status"] == "revisar"
    assert payload["placa"] == "ABC1234"
    assert "0.30" in payload["motivo_filtro"]


def test_nenhuma_leitura_valida_vira_filtrado():
    payload = _executar([("F172", 0.9)])
    assert payload["status"] == "filtrado"
    assert payload["placa"] == "F172"
    assert "melhor esforco" in payload["motivo_filtro"]


def test_ocr_sem_leituras_vira_filtrado_com_motivo_especifico():
    payload = _executar([])
    assert payload["status"] == "filtrado"
    assert payload["placa"] == "—"
    assert payload["motivo_filtro"] == "OCR nao identificou nenhum caractere"


def test_payload_mantem_confianca_yolo_separada_da_ocr():
    payload = _executar([("ABC1234", 0.7)])
    assert payload["confianca"] == 0.9      # YOLO (detector)
    assert payload["confianca_ocr"] == 0.7  # OCR (leitura)
