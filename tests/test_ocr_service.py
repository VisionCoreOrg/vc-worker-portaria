"""Testes do adapter EasyOCR multi-variante com leitor fake injetado."""

import numpy as np

from src.services.ocr_service import ALLOWLIST_PLACA, EasyOCRReader


def _crop():
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(40, 120, 3), dtype=np.uint8)


def _caixa(x, texto, conf):
    # bbox no formato do EasyOCR: 4 pontos [x, y]
    return ([[x, 0], [x + 50, 0], [x + 50, 10], [x, 10]], texto, conf)


class FakeLeitor:
    """Devolve um resultado pré-definido por chamada (uma por variante)."""

    def __init__(self, resultados):
        self._resultados = list(resultados)
        self.allowlists = []

    def readtext(self, img, allowlist=None):
        self.allowlists.append(allowlist)
        return self._resultados.pop(0)


def test_passa_allowlist_em_todas_as_variantes():
    fake = FakeLeitor([[], [], []])
    EasyOCRReader(fake).ler_texto(_crop())
    assert fake.allowlists == [ALLOWLIST_PLACA] * 3


def test_concatena_caixas_em_ordem_esquerda_direita():
    # Regressão do caso real 24.jpg: caixas fora de ordem viravam 'J17...QEX7'
    fake = FakeLeitor([[_caixa(100, "J17", 0.8), _caixa(0, "QEX7", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    # confiança = a da MAIOR caixa (QEX7, 4 chars, conf 0.9), não o mínimo global
    assert ("QEX7J17", 0.9) in leituras


def test_caixas_grandes_viram_candidatas_individuais():
    fake = FakeLeitor([[_caixa(0, "BRASIL", 0.5), _caixa(60, "ABC1234", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("ABC1234", 0.9) in leituras
    assert ("BRASIL", 0.5) in leituras
    # concatenação herda a conf da MAIOR caixa (ABC1234, 7 chars, 0.9), não 0.5
    assert ("BRASILABC1234", 0.9) in leituras


def test_caixa_unica_nao_duplica_candidata():
    fake = FakeLeitor([[_caixa(0, "ABC1234", 0.9)], [], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert leituras.count(("ABC1234", 0.9)) == 1


def test_erro_em_uma_variante_nao_derruba_as_demais():
    class LeitorFalhaPrimeira(FakeLeitor):
        def readtext(self, img, allowlist=None):
            if not self.allowlists:
                self.allowlists.append(allowlist)
                raise RuntimeError("boom")
            return super().readtext(img, allowlist=allowlist)

    fake = LeitorFalhaPrimeira([[_caixa(0, "ABC1234", 0.9)], []])
    leituras, _ = EasyOCRReader(fake).ler_texto(_crop())
    assert ("ABC1234", 0.9) in leituras


def test_retorna_imagem_binarizada_para_upload():
    fake = FakeLeitor([[], [], []])
    _, binarizada = EasyOCRReader(fake).ler_texto(_crop())
    assert set(np.unique(binarizada)) <= {0, 255}
