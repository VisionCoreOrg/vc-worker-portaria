# Imagem do Python otimizada
FROM python:3.15.0a6-slim

# Define variáveis de ambiente para o Python
# PYTHONDONTWRITEBYTECODE: Impede o Python de gravar arquivos .pyc
# PYTHONUNBUFFERED: Garante que os logs do Celery/Python saiam no terminal em tempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala as dependências de sistema operacional necessárias para o OpenCV e Tesseract
# A flag --no-install-recommends ajuda a manter a imagem menor
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do worker para dentro do container
COPY . .

# Cria um usuário não-root por questões de segurança (opcional, mas recomendado)
RUN useradd -m workeruser
USER workeruser

# Comando padrão para iniciar o Worker do Celery
# Substituir 'tasks' pelo nome do arquivo Python onde a instância do Celery está definida (ex: tasks.py)
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]