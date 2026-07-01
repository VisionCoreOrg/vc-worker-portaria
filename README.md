# 🚗 Worker 01 — Portaria (Real-Time OCR)

Microsserviço responsável pelo processamento de Visão Computacional e OCR das câmeras da guarita de entrada do estacionamento.

Opera em modo **orientado a eventos**: aguarda mensagens via fila Redis e processa cada imagem na chegada, sem polling.

## 🏗️ Arquitetura e Fluxo

```
[vc-camera-mock ou câmera real]
        ↓ LPUSH  (camera:portaria:queue)
[redis_broker — parking-infra]
        ↓ BRPOP (bloqueante)
[Worker Portaria]
    → Baixa imagem do MinIO
    → ONNX Runtime detecta placa (modelo YOLOv8)
    → EasyOCR lê o texto
    → Upload do recorte no MinIO
    → POST /api/vagas/registro (via Nginx → api_core)
```

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Mensageria | Redis (LPUSH/BRPOP — fila FIFO) via `parking-infra` |
| IA / Detecção | YOLOv8 exportado para ONNX + ONNX Runtime |
| OCR | EasyOCR |
| Storage | MinIO (API compatível com S3) |
| Infraestrutura | Docker + Docker Compose |

## 🚀 Como Rodar

> **Pré-requisito:** O `parking-infra` **deve estar rodando** antes deste serviço. Ele cria a rede `parking_global_net` e o broker Redis.

### 1. Suba o parking-infra primeiro

```bash
cd ../parking-infra
docker-compose up -d
```

### 2. Configure o `.env`

```bash
cp .env.example .env
# Edite .env com suas senhas de MinIO e Redis
```

### 3. Prepare o modelo ONNX

O worker utiliza o modelo no formato `.onnx`. Se você só possui o arquivo `.pt`, execute o script de exportação abaixo (requer apenas Docker, sem instalações adicionais):

```bash
chmod +x scripts/export_model.sh
./scripts/export_model.sh
```

Isso gera `models/modelo_placas.onnx` automaticamente usando a imagem oficial do Ultralytics.

> **Nota:** Os arquivos de modelo (`.pt` e `.onnx`) são ignorados pelo Git. Consulte o time ou o repositório de modelos para obter a versão mais recente.

### 4. Suba o worker (CPU — padrão para todo o time)

```bash
docker-compose up -d --build
```

Isso sobe:
- `visioncore_minio` — armazenamento de imagens (porta 9000 / console 9001)
- `vc_worker_portaria` — worker em modo escuta da fila Redis, rodando inferência em CPU

### 4b. Rodar com GPU NVIDIA (opcional)

Se você possui uma GPU NVIDIA e o `nvidia-container-toolkit` instalado no host, use o arquivo de override dedicado:

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Esse override:
- Compila a imagem com `onnxruntime-gpu` no lugar de `onnxruntime`
- Passa `GPU_PROVIDER=nvidia` ao container, ativando o `CUDAExecutionProvider` e o GPU mode do EasyOCR
- Reserva 1 GPU NVIDIA para o container

### 5. Acompanhe os logs

```bash
docker-compose logs -f worker
```

## 📁 Estrutura

```
scripts/
└── export_model.sh         # Exporta modelo .pt → .onnx via Docker
src/
├── config.py               # Variáveis de ambiente (inclui GPU_PROVIDER)
├── main.py                 # Loop principal (event-driven)
└── services/
    ├── ia_service.py       # Detecção YOLO via ONNX Runtime
    ├── ocr_service.py      # OCR EasyOCR
    ├── redis_service.py    # Consumo da fila Redis (BRPOP)
    ├── storage_service.py  # Upload/download MinIO
    └── api_service.py      # Envio para API Core
models/
├── modelo_placas.pt        # Modelo original (ignorado pelo Git)
└── modelo_placas.onnx      # Modelo exportado para inferência (ignorado pelo Git)
docker-compose.yml          # Configuração padrão (CPU)
docker-compose.gpu.yml      # Override para GPU NVIDIA
```

## 🔑 Variáveis de Ambiente

| Variável | Descrição | Padrão |
|---|---|---|
| `MINIO_ENDPOINT` | URL do MinIO | `http://visioncore_minio:9000` |
| `MINIO_ROOT_USER` | Usuário MinIO | — |
| `MINIO_ROOT_PASSWORD` | Senha MinIO | — |
| `MINIO_BUCKET_NAME` | Bucket de imagens | `plate-bucket` |
| `API_URL` | Endpoint da API Core | — |
| `CAMERA_ID` | Identificador da câmera | `camera_default` |
| `REDIS_HOST` | Host do broker Redis | `parking_redis` |
| `REDIS_PORT` | Porta Redis | `6379` |
| `REDIS_PASSWORD` | Senha do Redis | — |
| `REDIS_QUEUE` | Nome da fila de eventos | `camera:portaria:queue` |
| `GPU_PROVIDER` | Provider de GPU para inferência (`none` \| `nvidia`) | `none` |
