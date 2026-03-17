# 🚗 Worker 01 — Portaria (Real-Time OCR)

Microsserviço responsável pelo processamento de Visão Computacional e OCR das câmeras da guarita de entrada do estacionamento.

Opera em modo **orientado a eventos**: aguarda mensagens via fila Redis e processa cada imagem na chegada, sem polling.

## 🏗️ Arquitetura e Fluxo

```
[vc-camera-mock ou câmera real]
        ↓ LPUSH  (camera:portaria:queue)
[redis_broker — parking-infra]
        ↓ BLPOP (bloqueante)
[Worker Portaria]
    → Baixa imagem do MinIO
    → YOLO detecta placa
    → EasyOCR lê o texto
    → Upload do recorte no MinIO
    → POST /api/vagas/registro (via Nginx → api_core)
```

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.10 |
| Mensageria | Redis (BLPOP) via `parking-infra` |
| IA / Detecção | YOLO v8 (Ultralytics) |
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
# Edite .env se necessário (senhas MinIO, Redis etc.)
```

### 3. Suba o worker e o MinIO

```bash
docker-compose up -d
```

Isso sobe:
- `visioncore_minio` — armazenamento de imagens (porta 9000 / console 9001)
- `vc_worker_portaria` — worker em modo escuta da fila Redis

### 4. Acompanhe os logs

```bash
docker-compose logs -f worker
```

## 📁 Estrutura

```
src/
├── config.py               # Variáveis de ambiente
├── main.py                 # Loop principal (event-driven)
└── services/
    ├── redis_service.py    # Consumo da fila Redis (BLPOP)
    ├── storage_service.py  # Upload/download MinIO
    ├── ia_service.py       # Detecção YOLO
    ├── ocr_service.py      # OCR EasyOCR
    └── api_service.py      # Envio para API Core
models/
└── modelo_placas.pt        # Modelo YOLO treinado
```

## 🔑 Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `MINIO_ENDPOINT` | URL do MinIO (ex: `http://visioncore_minio:9000`) |
| `MINIO_ROOT_USER` | Usuário MinIO |
| `MINIO_ROOT_PASSWORD` | Senha MinIO |
| `MINIO_BUCKET_NAME` | Bucket de imagens (ex: `plate-bucket`) |
| `API_URL` | Endpoint da API Core (ex: `http://api_core:8000/api/vagas/registro`) |
| `CAMERA_ID` | Identificador da câmera (ex: `portaria_principal`) |
| `REDIS_HOST` | Host do broker Redis (ex: `parking_redis`) |
| `REDIS_PORT` | Porta Redis (padrão: `6379`) |
| `REDIS_PASSWORD` | Senha do Redis |
| `REDIS_QUEUE` | Nome da fila (padrão: `camera:portaria:queue`) |
