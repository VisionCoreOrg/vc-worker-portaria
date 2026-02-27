import boto3
import cv2
import io
import uuid
import numpy as np
from botocore.client import Config
from src.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME

def obter_cliente_s3():
    """Configura e retorna o cliente do Boto3 apontando para o MinIO local"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1' # O MinIO ignora a região, mas o boto3 exige que ela exista
    )

def listar_imagens_s3(prefixo="dataset/"):
    """Lista todos os arquivos de imagem dentro de uma pasta específica no bucket."""
    cliente_s3 = obter_cliente_s3()
    try:
        resposta = cliente_s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefixo)
        
        if 'Contents' not in resposta:
            return []
            
        arquivos = [obj['Key'] for obj in resposta['Contents'] if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
        return arquivos
        
    except Exception as e:
        print(f"[STORAGE] Erro ao listar arquivos: {e}")
        return []

def baixar_imagem_s3(chave_arquivo):
    """Baixa a imagem do MinIO direto para a memória RAM (formato OpenCV NumPy Array)."""
    cliente_s3 = obter_cliente_s3()
    try:
        resposta = cliente_s3.get_object(Bucket=BUCKET_NAME, Key=chave_arquivo)
        bytes_imagem = resposta['Body'].read()
        
        array_np = np.frombuffer(bytes_imagem, np.uint8)
        imagem_cv2 = cv2.imdecode(array_np, cv2.IMREAD_COLOR)
        
        return imagem_cv2
        
    except Exception as e:
        print(f"[STORAGE] Erro ao baixar imagem '{chave_arquivo}': {e}")
        return None

def upload_imagem_s3(imagem_numpy, string_placa):
    """
    Recebe o crop da placa (NumPy Array) e a string do OCR.
    Faz o upload direto da memória para a pasta "recortes/" e retorna a URL pública.
    """
    cliente_s3 = obter_cliente_s3()
    
    nome_arquivo = f"recortes/{string_placa}_{uuid.uuid4().hex[:8]}.jpg"
    
    sucesso, buffer_imagem = cv2.imencode('.jpg', imagem_numpy)
    if not sucesso:
        print("[STORAGE] Erro ao codificar a imagem para upload.")
        return None
    
    bytes_io = io.BytesIO(buffer_imagem.tobytes())
    
    try:
        cliente_s3.upload_fileobj(
            bytes_io, 
            BUCKET_NAME, 
            nome_arquivo,
            ExtraArgs={'ContentType': 'image/jpeg'} 
        )
        
        url_publica = f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{nome_arquivo}"
        return url_publica
        
    except Exception as e:
        print(f"[STORAGE] Erro ao fazer upload no MinIO: {e}")
        return None