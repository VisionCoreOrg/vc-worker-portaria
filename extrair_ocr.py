import os
import cv2
import easyocr
import json
import re


dict_int_para_letra = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_letra_para_int = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}

def corrigir_placa(texto_ocr):
    # 1. Limpeza pesada: tira tudo que não for letra e número
    texto = "".join(e for e in texto_ocr if e.isalnum()).upper()
    
    # Se o OCR leu a palavra "BRASIL" na tarja azul ou algo a mais, vamos tentar pegar só os 7 últimos caracteres
    if len(texto) > 7:
        texto = texto[-7:]
        
    # Se leu menos de 7, a placa está cortada ou ilegível, retornamos como está
    if len(texto) != 7:
        return texto

    placa_corrigida = ""
    
    # 2. Aplicar a regra posição por posição (Index 0 a 6)
    for i, char in enumerate(texto):
        # Posições 0, 1, 2: SEMPRE LETRAS
        if i in [0, 1, 2]:
            if char in dict_int_para_letra:
                placa_corrigida += dict_int_para_letra[char]
            else:
                placa_corrigida += char
                
        # Posições 3, 5, 6: SEMPRE NÚMEROS (no index 3, 5 e 6 da string)
        elif i in [3, 5, 6]:
            if char in dict_letra_para_int:
                placa_corrigida += dict_letra_para_int[char]
            else:
                placa_corrigida += char
                
        # Posição 4: Letra (Mercosul) ou Número (Antiga)
        elif i == 4:
            # Aqui deixamos o que o OCR leu, pois pode ser ambos.
            placa_corrigida += char

    return placa_corrigida

def pre_processar_imagem(img):
    # 1. Grayscale: O primeiro passo para qualquer binarização
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Redimensionamento: Dobrar o tamanho usando interpolação bicúbica
    # Isso "estica" a imagem de forma suave, dando mais pixels para o OCR analisar
    largura = int(gray.shape[1] * 2)
    altura = int(gray.shape[0] * 2)
    ampliada = cv2.resize(gray, (largura, altura), interpolation=cv2.INTER_CUBIC)
    
    # 3. Suavização Leve: Tira pequenos "farelos" de ruído antes de binarizar
    desfoque = cv2.GaussianBlur(ampliada, (3, 3), 0)
    
    # 4. Binarização de Otsu: 
    # O método de Otsu é inteligente, ele acha o meio-termo ideal do contraste sozinho
    _, binarizada = cv2.threshold(desfoque, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binarizada

def extrair_textos_ocr(pasta_recortes, arquivo_json_saida):
    print("Carregando modelo EasyOCR...")
    leitor_ocr = easyocr.Reader(['pt', 'en'], gpu=False)
    
    resultados_json = []
    pasta_debug = "placas_binarizadas"
    
    if not os.path.exists(pasta_recortes):
        print(f"Erro: A pasta '{pasta_recortes}' não foi encontrada.")
        return

    # Cria a pasta para você ver como ficaram as imagens processadas
    if not os.path.exists(pasta_debug):
        os.makedirs(pasta_debug)

    arquivos = [f for f in os.listdir(pasta_recortes) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nIniciando extração com Binarização em {len(arquivos)} imagens...\n")
    print("-" * 50)
    
    for idx, nome_arquivo in enumerate(arquivos):
        caminho_imagem = os.path.join(pasta_recortes, nome_arquivo)
        
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            continue
            
        # Aplica nosso novo filtro de contraste pesado
        img_processada = pre_processar_imagem(imagem)
        
        # Salva a imagem binarizada para você poder inspecionar os resultados
        caminho_debug = os.path.join(pasta_debug, f"bin_{nome_arquivo}")
        cv2.imwrite(caminho_debug, img_processada)
            
        # Passa a imagem já binarizada para o OCR
        resultado_ocr = leitor_ocr.readtext(img_processada)
        
        texto_bruto = ""
        for (bbox, texto, conf_ocr) in resultado_ocr:
            texto_bruto += texto
            
        # Aplica a heurística inteligente!
        texto_placa = corrigir_placa(texto_bruto)
        
        texto_placa = ""
        for (bbox, texto, conf_ocr) in resultado_ocr:
            texto_limpo = "".join(e for e in texto if e.isalnum()).upper()
            texto_placa += texto_limpo
        
        if texto_placa:
            print(f"[LIDA] {nome_arquivo} -> {texto_placa}")
            
            objeto_placa = {
                "id": idx + 1,
                "arquivo_origem": nome_arquivo,
                "placa": texto_placa
            }
            resultados_json.append(objeto_placa)
        else:
            print(f"[VAZIA] {nome_arquivo} -> O OCR não encontrou texto, mesmo após binarizar.")

    with open(arquivo_json_saida, 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, ensure_ascii=False, indent=4)
        
    print("-" * 50)
    print(f"Sucesso! Dados salvos em '{arquivo_json_saida}'.")
    print(f"Abra a pasta '{pasta_debug}' para ver como as placas ficaram em preto e branco!")

if __name__ == "__main__":
    PASTA_RECORTES = "placas_recortadas"
    ARQUIVO_JSON = "dados_placas.json"
    
    extrair_textos_ocr(PASTA_RECORTES, ARQUIVO_JSON)