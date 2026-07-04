# Dicionários de mapeamento de caracteres ambíguos no OCR de placas brasileiras.
# Nas posições de dígito, 'Q' e 'D' mapeiam ambos para '0' intencionalmente: são
# as duas confusões de OCR mais comuns com o zero e não há mapeamento melhor.
dict_int_para_letra = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_letra_para_int = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}


def corrigir_placa(texto_ocr: str) -> str:
    """
    Limpa caracteres especiais e aplica heurísticas de correção para o padrão brasileiro
    de placas (AAA0000 ou AAA0A00 - Mercosul).
    """
    texto = "".join(e for e in texto_ocr if e.isalnum()).upper()
    if len(texto) > 7:
        texto = texto[-7:]
    if len(texto) != 7:
        return texto

    placa_corrigida = ""
    for i, char in enumerate(texto):
        if i in [0, 1, 2]:
            placa_corrigida += dict_int_para_letra.get(char, char)
        elif i in [3, 5, 6]:
            placa_corrigida += dict_letra_para_int.get(char, char)
        elif i == 4:
            placa_corrigida += char
    return placa_corrigida
