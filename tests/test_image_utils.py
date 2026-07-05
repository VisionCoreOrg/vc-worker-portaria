"""Testes das variantes de pré-processamento para OCR multi-variante."""

import numpy as np

from src.utils.image_utils import variantes_para_ocr


def _crop_sintetico():
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(40, 120, 3), dtype=np.uint8)


def test_retorna_tres_variantes_na_ordem_esperada():
    nomes = [nome for nome, _ in variantes_para_ocr(_crop_sintetico())]
    assert nomes == ["cinza_clahe", "otsu", "otsu_invertida"]


def test_todas_ampliadas_2x_grayscale_uint8():
    for nome, img in variantes_para_ocr(_crop_sintetico()):
        assert img.shape == (80, 240), nome
        assert img.dtype == np.uint8, nome


def test_otsu_e_binaria_e_invertida_e_complemento():
    variantes = dict(variantes_para_ocr(_crop_sintetico()))
    assert set(np.unique(variantes["otsu"])) <= {0, 255}
    assert (variantes["otsu_invertida"] == 255 - variantes["otsu"]).all()
