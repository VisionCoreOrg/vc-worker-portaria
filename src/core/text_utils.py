"""Regras de domínio para extração e correção de placas brasileiras."""

import re
from dataclasses import dataclass

# Dicionários de mapeamento de caracteres ambíguos no OCR de placas brasileiras.
# Nas posições de dígito, 'Q' e 'D' mapeiam ambos para '0' intencionalmente: são
# as duas confusões de OCR mais comuns com o zero e não há mapeamento melhor.
dict_int_para_letra = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_letra_para_int = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}

# Padrão único que cobre placa antiga (AAA0000) e Mercosul (AAA0A00):
# a posição 4 aceita letra ou dígito.
PADRAO_PLACA_BR = re.compile(r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$")
_NAO_ALFANUM_ASCII = re.compile(r"[^A-Z0-9]")


@dataclass
class Extracao:
    placa: str      # melhor janela corrigida (ou o texto normalizado, se < 7 chars)
    valida: bool    # True se casa com PADRAO_PLACA_BR
    correcoes: int  # quantos caracteres o mapa de ambiguidade alterou


@dataclass
class Decisao:
    placa: str
    valida: bool
    confianca_ocr: float
    texto_bruto: str  # leitura crua que originou a placa escolhida


def normalizar_texto_ocr(texto: str) -> str:
    """Uppercase + remove tudo que não for ASCII A-Z/0-9.

    Não usar str.isalnum(): ele aceita alfanuméricos Unicode (ex.: 'ª'),
    que já contaminaram placas gravadas no banco.
    """
    return _NAO_ALFANUM_ASCII.sub("", texto.upper())


def corrigir_janela(janela: str) -> str:
    """Aplica o mapa posicional de ambiguidade a uma janela de exatamente 7 chars."""
    corrigida = []
    for i, char in enumerate(janela):
        if i in (0, 1, 2):
            corrigida.append(dict_int_para_letra.get(char, char))
        elif i in (3, 5, 6):
            corrigida.append(dict_letra_para_int.get(char, char))
        else:  # posição 4: letra (Mercosul) ou dígito (antiga) — não forçar
            corrigida.append(char)
    return "".join(corrigida)


def eh_formato_valido(placa: str) -> bool:
    return bool(PADRAO_PLACA_BR.fullmatch(placa))


def extrair_placa(texto_ocr: str) -> Extracao:
    """Encontra a melhor janela de 7 caracteres dentro do texto cru do OCR.

    Substitui a heurística antiga de "últimos 7 chars", que desalinhava a
    leitura quando havia texto extra no crop (ex.: 'BRASIL', molduras de
    concessionária). Critério de escolha entre janelas: formato válido >
    menos correções aplicadas > mais à direita (o lixo observado nos dados
    reais é majoritariamente prefixo).
    """
    texto = normalizar_texto_ocr(texto_ocr)
    if len(texto) < 7:
        return Extracao(placa=texto, valida=False, correcoes=0)

    melhor_chave = None
    melhor = None
    for i in range(len(texto) - 6):
        janela = texto[i:i + 7]
        corrigida = corrigir_janela(janela)
        correcoes = sum(1 for a, b in zip(janela, corrigida) if a != b)
        chave = (eh_formato_valido(corrigida), -correcoes, i)
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = Extracao(placa=corrigida, valida=chave[0], correcoes=correcoes)
    return melhor


def escolher_leitura(leituras: list[tuple[str, float]]) -> Decisao:
    """Escolhe a melhor leitura entre candidatas (texto_cru, confianca_ocr).

    Critério: extração em formato válido > menos correções > maior
    confiança do OCR.
    """
    if not leituras:
        return Decisao(placa="", valida=False, confianca_ocr=0.0, texto_bruto="")

    melhor_chave = None
    melhor = None
    for texto, confianca in leituras:
        extracao = extrair_placa(texto)
        chave = (extracao.valida, -extracao.correcoes, confianca)
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor = Decisao(
                placa=extracao.placa,
                valida=extracao.valida,
                confianca_ocr=confianca,
                texto_bruto=texto,
            )
    return melhor


def decidir_status(decisao: Decisao, conf_minima_sucesso: float) -> tuple[str, str | None]:
    """Decide o status do registro a partir da leitura escolhida.

    Regra única compartilhada entre o worker (ProcessarEventoUseCase) e o
    harness de avaliação offline — se ela mudar, os dois mudam juntos.
    Retorna (status, motivo_filtro).
    """
    if decisao.valida and decisao.confianca_ocr >= conf_minima_sucesso:
        return "sucesso", None
    if decisao.valida:
        return "revisar", f"Confianca OCR baixa ({decisao.confianca_ocr:.2f})"
    if decisao.placa:
        return "filtrado", f"Nenhuma leitura em formato BR (melhor esforco: '{decisao.placa[:20]}')"
    return "filtrado", "OCR nao identificou nenhum caractere"
