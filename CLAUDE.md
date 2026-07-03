# vc-worker-portaria

Worker de detecção e OCR de placas veiculares. Escuta uma fila Redis, baixa imagens do MinIO, detecta a placa com YOLOv8 (ONNX), lê o texto com EasyOCR e envia o resultado ao `vc-api-core`.

## Stack

- **Linguagem**: Python 3.12
- **Mensageria**: Redis (LPUSH/BLMOVE — fila FIFO com ack (lista de processamento)) (broker no próprio compose standalone ou no stack raiz)
- **Detecção (IA)**: YOLOv8 via ONNX Runtime (suporte a CPU e GPU NVIDIA)
- **OCR**: EasyOCR (PT + EN) com heurísticas para placas brasileiras
- **Storage**: MinIO via boto3
- **HTTP**: requests
- **Container**: Docker multi-stage (CPU padrão, GPU opcional)

## Estrutura

```
src/
├── main.py                  # Loop principal: BLMOVE → processar → ack
├── config.py                # Carrega variáveis de ambiente
└── services/
    ├── ia_service.py        # Inferência ONNX (YOLOv8), NMS, letterbox
    ├── ocr_service.py       # EasyOCR + pré-processamento + correção de placa BR
    ├── redis_service.py     # Conexão Redis com retry, BLMOVE/ack, recuperação de órfãos
    ├── storage_service.py   # Download/upload MinIO via boto3
    └── api_service.py       # POST para /api/vagas/registro
models/
├── modelo_placas.pt         # Modelo YOLOv8 PyTorch (git-ignored)
└── modelo_placas.onnx       # Modelo exportado ONNX (git-ignored, gerado por scripts/)
scripts/
└── export_model.sh          # Converte .pt → .onnx via Docker (sem instalar PyTorch local)
```

## Variáveis de Ambiente (.env)

| Variável | Valor Padrão | Descrição |
|----------|-------------|-----------|
| `REDIS_HOST` | `parking_redis` | Hostname do Redis (serviço Docker do parking-infra) |
| `REDIS_PORT` | `6379` | Porta Redis |
| `REDIS_PASSWORD` | — | Deve ser igual ao `REDIS_PASSWORD` do `parking-infra` |
| `REDIS_QUEUE` | `camera:portaria:queue` | Nome da fila (consumida via BLMOVE) |
| `API_URL` | `http://api_core:8000/api/vagas/registro` | Endpoint de registro |
| `CAMERA_ID` | `portaria_principal` | Identificador desta câmera |
| `MINIO_ENDPOINT` | `http://visioncore_minio:9000` | URL do MinIO |
| `MINIO_ROOT_USER` | — | Credencial MinIO |
| `MINIO_ROOT_PASSWORD` | — | Credencial MinIO |
| `MINIO_BUCKET_NAME` | `plate-bucket` | Bucket para recortes de placas |
| `GPU_PROVIDER` | `none` | `none` (CPU) ou `nvidia` (CUDA) |

## Pipeline de Processamento

```
Evento Redis (JSON)
    ↓
{ "path": "dataset/img.jpg", "camera_id": "...", "timestamp": "..." }
    ↓
MinIO: download imagem → NumPy array (OpenCV)
    ↓
YOLOv8 ONNX: letterbox → inferência → NMS → recorte + confiança
    ↓
EasyOCR: pré-processamento → leitura → heurística BR → 7 chars
    ↓
MinIO: upload recorte → URL pública
    ↓
POST /api/vagas/registro
```

## Detalhes de IA

### ia_service.py — Detecção YOLOv8 ONNX

- Input: imagem 640×640 com letterbox (mantém proporção, adiciona padding)
- Output ONNX: `[1, 5, 8400]` → extrai (cx, cy, w, h, conf)
- Confiança mínima: `0.5`
- NMS IoU threshold: `0.45`
- Retorna: recorte da placa + score de confiança

### ocr_service.py — OCR com Heurísticas BR

Pré-processamento OpenCV (`pre_processar_imagem_ocr` em `src/utils/image_utils.py`), nesta ordem:

1. **Grayscale** (`cv2.cvtColor` BGR→GRAY)
2. **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8,8)`) — equaliza contraste localmente, compensa iluminação/sombras
3. **Upscale 2×** (`cv2.resize`, interpolação `INTER_CUBIC`)
4. **Filtro bilateral** (`d=5`, `sigmaColor=75`, `sigmaSpace=75`) — suaviza ruído **preservando as bordas** dos caracteres (não é blur gaussiano)
5. **Binarização de Otsu** (`cv2.threshold` com `THRESH_BINARY + THRESH_OTSU`)

Correção de placa (`corrigir_placa` em `src/utils/text_utils.py`) — formato BR `AAA0000` (antigo) ou `AAA0A00` (Mercosul):
- Remove não-alfanuméricos, faz `upper()`, e mantém os **últimos 7** caracteres se vier mais que isso
- Posições 0-2 (letras): `{'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '8':'B'}`
- Posições 3, 5, 6 (dígitos): `{'O':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'B':'8', 'Q':'0', 'D':'0'}`
- **Posição 4 não é alterada** — pode ser letra (Mercosul) ou dígito (antigo), então forçar quebraria um dos formatos

Validação: texto final deve ter exatamente 7 caracteres.

## Redis — Formato do Evento

```json
{
  "path": "dataset/frame_001.jpg",
  "camera_id": "portaria_principal",
  "timestamp": "2026-05-13T12:34:56Z"
}
```

O campo `path` é a chave do objeto no MinIO bucket.

### Fila confiável (ack)

O consumo usa `BLMOVE camera:portaria:queue → camera:portaria:queue:processing`:
o evento fica retido na lista de processamento até o ack (`LREM`) após o
processamento — com sucesso ou falha tratada. No startup, eventos órfãos em
`:processing` (sobras de crash) voltam para a fila. Semântica resultante:
**at-least-once** — reprocessamento eventual é aceitável (a API tem regra de
placa duplicada).

## Docker

### CPU (padrão)

```bash
docker-compose up -d --build
```

GPU: requer build com o arg GPU_BUILD=nvidia e configuração de runtime NVIDIA — não há override de compose versionado.

### Exportar modelo para ONNX

```bash
# Requer apenas Docker instalado (sem PyTorch local)
bash scripts/export_model.sh
# Gera models/modelo_placas.onnx a partir de models/modelo_placas.pt
```

## Desenvolvimento Local

```bash
pip install -r requirements.txt

# Redis e MinIO devem estar rodando (via parking-infra e docker-compose local)
# Ajustar .env com REDIS_HOST=localhost e MINIO_ENDPOINT=http://localhost:9000

python -m src.main
```

## Resiliência

- **Redis**: retry automático com 10 tentativas e backoff de 3s
- **Eventos inválidos**: campo `path` ausente ou JSON inválido → ack + log + skip (não voltam para a fila)
- **Falha de download**: log + skip (não bloqueia a fila)
- **Placa não detectada**: log + skip
- **OCR inválido** (≠7 chars): log + skip
- **API Core indisponível**: log warning + continua processando (não bloqueia fila)
- **Worker restart**: `restart: "no"` em dev (política alinhada ao stack raiz)
- **Crash do worker**: eventos em voo ficam em `camera:portaria:queue:processing` e voltam à fila no próximo start

## Dependências do Serviço

Standalone: nenhuma dependência externa — Redis e MinIO sobem no próprio compose (api-core ausente; POST tolerado com warning). Stack raiz: Redis/MinIO/api-core do compose raiz.
