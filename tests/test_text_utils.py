"""Testes da heurística de correção de placas brasileiras (regra de domínio)."""

from src.core.text_utils import corrigir_placa


def test_placa_antiga_valida_permanece():
    assert corrigir_placa("ABC1234") == "ABC1234"


def test_placa_mercosul_valida_permanece():
    assert corrigir_placa("ABC1D23") == "ABC1D23"


def test_substitui_digitos_por_letras_no_prefixo():
    # Posições 0-2 são sempre letras: 4→A, 8→B
    assert corrigir_placa("48C1234") == "ABC1234"


def test_substitui_letras_por_digitos_nas_posicoes_numericas():
    # Posições 3, 5 e 6 são sempre dígitos: I→1, Z→2
    assert corrigir_placa("ABCI2Z4") == "ABC1224"


def test_posicao_4_nao_e_alterada():
    # Posição 4 pode ser letra (Mercosul) ou dígito (antiga) — forçar quebraria um dos formatos
    assert corrigir_placa("ABC1O23") == "ABC1O23"
    assert corrigir_placa("ABC1023") == "ABC1023"


def test_ruido_e_removido_e_mantem_ultimos_7():
    assert corrigir_placa("BR ABC-1234") == "ABC1234"


def test_minusculas_sao_normalizadas():
    assert corrigir_placa("abc1234") == "ABC1234"


def test_tamanho_diferente_de_7_retorna_sem_correcao():
    assert corrigir_placa("AB12") == "AB12"
    assert corrigir_placa("") == ""
