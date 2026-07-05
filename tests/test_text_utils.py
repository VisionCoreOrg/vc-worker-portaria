"""Testes das regras de domínio de extração de placas brasileiras."""

from src.core.text_utils import (
    Decisao,
    Extracao,
    corrigir_janela,
    eh_formato_valido,
    escolher_leitura,
    extrair_placa,
    normalizar_texto_ocr,
)

# --- normalizar_texto_ocr ---------------------------------------------------

def test_normaliza_uppercase_e_remove_simbolos():
    assert normalizar_texto_ocr("br abc-1234!") == "BRABC1234"


def test_remove_alfanumericos_unicode():
    # str.isalnum() aceitaria 'ª' — regressão do caso real 'KYXDD28ª' (7.jpg)
    assert normalizar_texto_ocr("KYXDD28ª") == "KYXDD28"


# --- corrigir_janela ---------------------------------------------------------

def test_corrige_digitos_no_prefixo_de_letras():
    assert corrigir_janela("48C1234") == "ABC1234"


def test_corrige_letras_nas_posicoes_de_digito():
    assert corrigir_janela("ABCI2Z4") == "ABC1224"


def test_posicao_4_nao_e_alterada():
    # Pode ser letra (Mercosul) ou dígito (antiga) — forçar quebraria um formato
    assert corrigir_janela("ABC1O23") == "ABC1O23"
    assert corrigir_janela("ABC1023") == "ABC1023"


# --- eh_formato_valido -------------------------------------------------------

def test_valida_formato_antigo_e_mercosul():
    assert eh_formato_valido("ABC1234")
    assert eh_formato_valido("ABC1D23")


def test_rejeita_formatos_impossiveis():
    # Regressão do caso real '7UB5D38' (10.jpg), aceito pelo filtro antigo de len==7
    assert not eh_formato_valido("7UB5D38")
    assert not eh_formato_valido("AB1C234")
    assert not eh_formato_valido("ABC12345")
    assert not eh_formato_valido("")


# --- extrair_placa -----------------------------------------------------------

def test_placa_exata_permanece():
    assert extrair_placa("ABC1234") == Extracao("ABC1234", True, 0)


def test_minusculas_sao_normalizadas():
    assert extrair_placa("abc1d23") == Extracao("ABC1D23", True, 0)


def test_ignora_prefixo_brasil():
    # Regressão: bruto real 'BRasILPOX4G21' (38.jpg) virava 'XAG218R' com os últimos-7
    ext = extrair_placa("BRasILPOX4G21")
    assert ext.placa == "POX4G21"
    assert ext.valida


def test_ignora_prefixo_aplicando_correcao():
    # Caso real (8.jpg): '191[00V0d55' → OOV0D55 (0→O duas vezes no prefixo de letras)
    ext = extrair_placa("191[00V0d55")
    assert ext.placa == "OOV0D55"
    assert ext.valida
    assert ext.correcoes == 2


def test_janela_com_menos_correcoes_vence():
    # Em '5HRFB4D54' (11.jpg), 'HRFB4D5' também validaria com 2 correções;
    # 'RFB4D54' vence com 0
    ext = extrair_placa("5hRFB4D54")
    assert ext.placa == "RFB4D54"
    assert ext.correcoes == 0


def test_empate_escolhe_janela_mais_a_direita():
    # Lixo observado nos dados é majoritariamente prefixo ('BRASIL', sujeira)
    ext = extrair_placa("ABC1234DEF5678")
    assert ext.placa == "DEF5678"


def test_texto_curto_retorna_invalido_sem_correcao():
    assert extrair_placa("AK1JD") == Extracao("AK1JD", False, 0)
    assert extrair_placa("") == Extracao("", False, 0)


# --- escolher_leitura --------------------------------------------------------

def test_lista_vazia_retorna_decisao_vazia():
    assert escolher_leitura([]) == Decisao("", False, 0.0, "")


def test_leitura_valida_vence_invalida_mesmo_com_conf_menor():
    decisao = escolher_leitura([("F172", 0.9), ("BRASILABC1234", 0.6)])
    assert decisao.placa == "ABC1234"
    assert decisao.valida
    assert decisao.confianca_ocr == 0.6
    assert decisao.texto_bruto == "BRASILABC1234"


def test_menos_correcoes_vence_confianca():
    decisao = escolher_leitura([("ABC1234", 0.5), ("48C1234", 0.9)])
    assert decisao.placa == "ABC1234"
    assert decisao.texto_bruto == "ABC1234"


def test_confianca_desempata_leituras_equivalentes():
    decisao = escolher_leitura([("ABC1234", 0.5), ("XYZ0A11", 0.9)])
    assert decisao.placa == "XYZ0A11"


def test_nenhuma_valida_retorna_melhor_esforco():
    decisao = escolher_leitura([("F172", 0.9)])
    assert decisao.placa == "F172"
    assert not decisao.valida
