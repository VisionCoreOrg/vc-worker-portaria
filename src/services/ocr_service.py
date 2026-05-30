# pyrefly: ignore [missing-import]
import cv2
import re

dict_int_para_letra = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_letra_para_int = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}

def pre_processar_imagem(img):
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

    # 5. Binarização de Otsu (que agora se torna extremamente precisa devido à iluminação uniforme do CLAHE)
    _, binarizada = cv2.threshold(
        suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binarizada


def corrigir_placa(texto_ocr):
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

def ler_texto_placa(placa_crop, leitor_ocr):
    """Recebe a imagem recortada e devolve (texto_corrigido, texto_bruto, img_processada)."""
    img_processada = pre_processar_imagem(placa_crop)
    resultado_ocr = leitor_ocr.readtext(img_processada)
    
    texto_bruto = ""
    for (bbox, texto, conf_ocr) in resultado_ocr:
        texto_bruto += texto
        
    # Aplica a heurística
    texto_placa = corrigir_placa(texto_bruto)
    return texto_placa, texto_bruto, img_processada