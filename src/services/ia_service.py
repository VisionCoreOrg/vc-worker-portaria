# src/services/ia_service.py
import cv2

def detectar_placa(img, modelo_yolo):
    """Recebe uma imagem (NumPy Array) e retorna o recorte da placa e a confiança."""
    
    resultados = modelo_yolo(img, conf=0.5, verbose=False)
    altura_img, largura_img, _ = img.shape
    
    for resultado in resultados:
        caixas = resultado.boxes
        
        for caixa in caixas:
            # Pegar coordenadas originais
            x1, y1, x2, y2 = map(int, caixa.xyxy[0])
            confianca_yolo = float(caixa.conf[0])
            
            # Lógica de Margem
            largura_caixa = x2 - x1
            altura_caixa = y2 - y1
            
            margem_y = int(altura_caixa * 0.15)
            margem_x = int(largura_caixa * 0.10)
            
            novo_x1 = max(0, x1 - margem_x)
            novo_y1 = max(0, y1 - margem_y)
            novo_x2 = min(largura_img, x2 + margem_x)
            novo_y2 = min(altura_img, y2 + margem_y)
            
            # Corta a imagem na memória
            placa_recortada = img[novo_y1:novo_y2, novo_x1:novo_x2]
            
            return placa_recortada, confianca_yolo
            
    return None, 0.0