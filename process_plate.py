import os
import cv2
import easyocr
from ultralytics import YOLO

def processar_dataset(pasta_entrada, pasta_saida, caminho_modelo):
    # 1. Carregar o modelo YOLOv8 que você baixou do Hugging Face
    print("Carregando modelo YOLO local...")
    modelo_yolo = YOLO(caminho_modelo)
    
    # 2. Carregar o EasyOCR (lê pt e en para cobrir o padrão Mercosul)
    print("Carregando EasyOCR...")
    leitor_ocr = easyocr.Reader(['pt', 'en'], gpu=False)

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nIniciando processamento de {len(arquivos)} imagens...\n")
    print("-" * 50)

    for nome_arquivo in arquivos:
        caminho_imagem = os.path.join(pasta_entrada, nome_arquivo)
        img = cv2.imread(caminho_imagem)
        
        if img is None:
            continue

        # 3. Inferência do YOLO na imagem
        # O conf=0.5 garante que ele só pegue o que tem mais de 50% de certeza de ser uma placa
        resultados = modelo_yolo(img, conf=0.5, verbose=False)
        placa_encontrada = False

        for resultado in resultados:
            caixas = resultado.boxes
            
            for resultado in resultados:
                caixas = resultado.boxes
            
            # Pegar a altura e largura da imagem original para as travas de segurança
            altura_img, largura_img, _ = img.shape
            
            for caixa in caixas:
                placa_encontrada = True
                
                # Pegar coordenadas originais do bounding box
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                confianca_yolo = float(caixa.conf[0])
                
                # --- INÍCIO DA LÓGICA DE MARGEM ---
                # Calcula a largura e altura atuais da caixa
                largura_caixa = x2 - x1
                altura_caixa = y2 - y1
                
                # Define a margem (ex: 15% da altura para cima/baixo e 10% da largura para os lados)
                # Você pode ajustar esses multiplicadores (0.15 e 0.10) conforme os testes
                margem_y = int(altura_caixa * 0.15)
                margem_x = int(largura_caixa * 0.10)
                
                # Aplica a margem garantindo que não vaze das bordas da imagem original
                novo_x1 = max(0, x1 - margem_x)
                novo_y1 = max(0, y1 - margem_y)
                novo_x2 = min(largura_img, x2 + margem_x)
                novo_y2 = min(altura_img, y2 + margem_y)
                
                # Recortar a placa com a nova margem
                placa_recortada = img[novo_y1:novo_y2, novo_x1:novo_x2]
                # --- FIM DA LÓGICA DE MARGEM ---
                
                # Salvar o recorte na pasta de resultados
                nome_saida_recorte = f"recorte_{nome_arquivo}"
                caminho_recorte = os.path.join(pasta_saida, nome_saida_recorte)
                cv2.imwrite(caminho_recorte, placa_recortada)

                # 4. Passar o recorte para o OCR ler
                resultado_ocr = leitor_ocr.readtext(placa_recortada)
                
                # ... (o resto do código do OCR continua igual)
                
                texto_placa = ""
                for (bbox, texto, conf_ocr) in resultado_ocr:
                    # Remove espaços e hifens, deixando apenas letras e números em maiúsculo
                    texto_limpo = "".join(e for e in texto if e.isalnum()).upper()
                    texto_placa += texto_limpo

                if texto_placa:
                    print(f"[SUCESSO] {nome_arquivo} -> Placa lida: {texto_placa} (Confiança YOLO: {confianca_yolo:.2f})")
                else:
                    print(f"[ALERTA] {nome_arquivo} -> Placa isolada e salva, mas o OCR falhou em ler as letras.")

        if not placa_encontrada:
            print(f"[FALHA] {nome_arquivo} -> O YOLO não detectou nenhuma placa.")

    print("-" * 50)
    print("Processamento concluído! Verifique a pasta de recortes.")

if __name__ == "__main__":
    PASTA_DATASET = "dataset"
    PASTA_RESULTADOS = "placas_recortadas"
    MODELO_YOLO_PLACAS = "modelo_placas.pt"
    
    processar_dataset(PASTA_DATASET, PASTA_RESULTADOS, MODELO_YOLO_PLACAS)