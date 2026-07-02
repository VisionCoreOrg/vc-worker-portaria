# 🚗 Worker 01 — Portaria (Real-Time OCR)

Microsserviço responsável pelo processamento de Visão Computacional e OCR das câmeras da guarita de entrada do estacionamento.

Opera em modo **orientado a eventos**: aguarda mensagens via fila Redis e processa cada imagem na chegada, sem polling.

## 🏗️ Arquitetura e Fluxo

```
[vc-camera-mock ou câmera real]
        ↓ LPUSH  (camera:portaria:queue)
[redis — compose local ou stack raiz]
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
| Mensageria | Redis (LPUSH/BRPOP — fila FIFO) (broker no próprio compose standalone ou no stack raiz) |
| IA / Detecção | YOLOv8 exportado para ONNX + ONNX Runtime |
| OCR | EasyOCR |
| Storage | MinIO (API compatível com S3) |
| Infraestrutura | Docker + Docker Compose |

## 🚀 Como Rodar

> O compose standalone é autossuficiente (Redis e MinIO próprios). Para o sistema completo, use o compose raiz do workspace.

### 1. Configure o `.env`

```bash
cp .env.example .env
# Edite .env com suas senhas de MinIO e Redis
```

### 2. Prepare o modelo ONNX

O worker utiliza o modelo no formato `.onnx`. Se você só possui o arquivo `.pt`, execute o script de exportação abaixo (requer apenas Docker, sem instalações adicionais):

```bash
chmod +x scripts/export_model.sh
./scripts/export_model.sh
```

Isso gera `models/modelo_placas.onnx` automaticamente usando a imagem oficial do Ultralytics.

> **Nota:** Os arquivos de modelo (`.pt` e `.onnx`) são ignorados pelo Git. Consulte o time ou o repositório de modelos para obter a versão mais recente.

### 3. Suba o worker (CPU — padrão para todo o time)

```bash
docker-compose up -d --build
```

Isso sobe:
- `redis` — broker da fila (uso interno, sem porta publicada)
- `minio` + `minio_init` — armazenamento de imagens e criação do bucket `plate-bucket` (uso interno, sem porta publicada)
- `worker` — em modo escuta da fila Redis, rodando inferência em CPU

### 3b. Rodar com GPU NVIDIA (opcional)

GPU: requer build com o arg GPU_BUILD=nvidia e configuração de runtime NVIDIA — não há override de compose versionado.

### 4. Acompanhe os logs

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
