# ===========================================================================
# GPU_BUILD controla a imagem base e qual onnxruntime instalar:
#   none    → CPU puro (python:3.12-slim)
#   nvidia  → CUDA (nvidia/cuda + python3.12 + onnxruntime-gpu)
#
# Uso:
#   docker build .                              # CPU
#   docker build --build-arg GPU_BUILD=nvidia . # NVIDIA GPU
# ===========================================================================
ARG GPU_BUILD=none

# --- Stage 1: CPU Base ---
FROM python:3.12-slim AS base-none
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: NVIDIA GPU Base ---
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-nvidia
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common curl \
    libgl1 libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.12 python3.12-dev python3.12-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 3: Final (selecionada dinamicamente) ---
FROM base-${GPU_BUILD} AS final

# Re-declara a variável ARG após o comando FROM
ARG GPU_BUILD=none

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$GPU_BUILD" = "nvidia" ]; then \
        pip install --no-cache-dir onnxruntime-gpu==1.20.0; \
    else \
        pip install --no-cache-dir onnxruntime==1.20.0; \
    fi

COPY . .

CMD ["python3", "-m", "src.main"]