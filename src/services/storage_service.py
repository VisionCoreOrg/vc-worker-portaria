import io
import uuid
from datetime import datetime
from typing import Optional
import boto3
import cv2
import numpy as np
from botocore.client import Config

from src.core.logger import configurar_logger

logger = configurar_logger("MinIOStorage")

class MinIOStorage:
    """
    Implementa o StorageRepository apontando para o MinIO local via Boto3.
    Conforme com o Protocol 'StorageRepository'.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        url_prefix: str = "/storage"
    ):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.url_prefix = url_prefix
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version='s3v4'),
                region_name='us-east-1'  # O MinIO ignora a região, mas o boto3 exige
            )
            logger.info(f"Cliente S3 do MinIO inicializado com sucesso em: {self.endpoint_url}")
        except Exception as e:
            logger.critical(f"Falha ao conectar e inicializar cliente MinIO no endpoint '{self.endpoint_url}': {e}")
            raise e

    def baixar_imagem(self, chave_arquivo: str) -> Optional[np.ndarray]:
        """Baixa a imagem original do MinIO direto para a memória RAM (BGR array NumPy)."""
        try:
            resposta = self.s3_client.get_object(Bucket=self.bucket_name, Key=chave_arquivo)
            bytes_imagem = resposta['Body'].read()
            
            array_np = np.frombuffer(bytes_imagem, np.uint8)
            imagem_cv2 = cv2.imdecode(array_np, cv2.IMREAD_COLOR)
            
            return imagem_cv2
        except Exception as e:
            logger.error(f"Erro ao baixar imagem '{chave_arquivo}' do bucket '{self.bucket_name}': {e}")
            return None

    def upload_recorte(self, imagem_numpy: np.ndarray, string_placa: str, sufixo: str = "") -> Optional[str]:
        """Upload do crop da placa ou binarizada direto da memória para o MinIO, retornando a URL parametrizada."""
        suf = f"_{sufixo}" if sufixo else ""
        nome_arquivo = f"placa_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{suf}.jpg"    
        
        sucesso, buffer_imagem = cv2.imencode('.jpg', imagem_numpy)
        if not sucesso:
            logger.error(f"Erro ao codificar recorte da placa '{string_placa}' para formato JPEG.")
            return None
        
        bytes_io = io.BytesIO(buffer_imagem.tobytes())
        
        try:
            self.s3_client.upload_fileobj(
                bytes_io, 
                self.bucket_name, 
                nome_arquivo,
                ExtraArgs={'ContentType': 'image/jpeg'} 
            )
            # URL pública gerada dinamicamente utilizando o prefixo injetado (desacoplado do Nginx)
            url_publica = f"{self.url_prefix.rstrip('/')}/{self.bucket_name}/{nome_arquivo}"
            return url_publica
        except Exception as e:
            logger.error(f"Erro ao fazer upload do recorte da placa '{string_placa}' no MinIO: {e}")
            return None

    def listar_imagens(self, prefixo="dataset/") -> list[str]:
        """Lista todos os arquivos de imagem dentro de uma pasta específica no bucket."""
        try:
            resposta = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefixo)
            if 'Contents' not in resposta:
                return []
            arquivos = [obj['Key'] for obj in resposta['Contents'] if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
            return arquivos
        except Exception as e:
            logger.error(f"Erro ao listar arquivos com prefixo '{prefixo}': {e}")
            return []