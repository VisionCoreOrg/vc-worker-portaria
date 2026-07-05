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
├── core/
│   ├── interfaces.py        # Protocols: Detector, OCRReader, StorageRepository, APIClient
│   ├── logger.py            # configurar_logger (cores ANSI, níveis, timestamp)
│   ├── task_limiter.py      # LimitedExecutor — backpressure do pool
│   ├── text_utils.py        # extrair_placa/escolher_leitura — janela deslizante + validação BR (domínio)
│   └── use_cases.py         # ProcessarEventoUseCase
├── services/
│   ├── ia_service.py        # Inferência ONNX (YOLOv8), NMS, letterbox
│   ├── ocr_service.py       # EasyOCR + pré-processamento (devolve texto cru)
│   ├── redis_service.py     # Conexão Redis com retry, BLMOVE/ack, recuperação de órfãos
│   ├── storage_service.py   # Download/upload MinIO via boto3
│   └── api_service.py       # POST para /api/vagas/registro
└── utils/
    └── image_utils.py       # pré-processamento OpenCV para OCR
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
| `OCR_CONF_MINIMA_SUCESSO` | `0.5` | Confiança mínima do OCR para status `sucesso`; abaixo disso (com formato válido) vira `revisar` |

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
EasyOCR: 3 variantes de pré-processamento → leituras candidatas → escolher_leitura → status sucesso/revisar/filtrado
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

Pré-processamento multi-variante (`variantes_para_ocr` em `src/utils/image_utils.py`) — gera **3 variantes** a partir do mesmo recorte, pois a binarização de Otsu global sozinha destrói placas sob iluminação irregular e escolhe a polaridade sem garantia de acerto:

1. **Grayscale + CLAHE** (`clipLimit=2.0`, `tileGridSize=(8,8)`) — equaliza contraste local
2. **Upscale 2×** (`cv2.resize`, `INTER_CUBIC`) + **filtro bilateral** (`d=5`, `sigmaColor=75`, `sigmaSpace=75`) — suaviza preservando bordas dos caracteres
3. A partir daí, três variantes seguem para o OCR: `cinza_clahe` (sem threshold), `otsu` (`THRESH_BINARY + THRESH_OTSU`) e `otsu_invertida` (`cv2.bitwise_not` da anterior)

`EasyOCRReader.ler_texto` (`src/services/ocr_service.py`) roda o EasyOCR em cada variante com **allowlist** restrita a `A-Z0-9` (elimina pontuação, minúsculas e Unicode na origem) e devolve uma lista de **leituras candidatas** `(texto_cru, confianca_ocr)` — inclui a concatenação de todas as caixas por variante e, quando há múltiplas caixas, cada caixa com ≥5 caracteres também vira candidata própria. Não escolhe a melhor leitura nem valida formato — isso é regra de domínio e vive no caso de uso.

Extração e escolha de placa (`src/core/text_utils.py`), aplicadas pelo `ProcessarEventoUseCase`:
- `extrair_placa`: janela deslizante de 7 caracteres sobre o texto normalizado (upper + só `A-Z0-9`), testando o padrão único `^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$` (cobre `AAA0000` antigo e `AAA0A00` Mercosul — posição 4 aceita letra ou dígito e nunca é forçada). Substitui a heurística antiga de "últimos 7 chars", que desalinhava a leitura quando havia texto extra no crop (ex.: `BRASIL`, moldura de concessionária). Critério entre janelas: formato válido > menos correções aplicadas > mais à direita.
  - Mapa de correção por posição: letras (posições 0-2) `{'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '8':'B'}`; dígitos (posições 3, 5, 6) `{'O':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'B':'8', 'Q':'0', 'D':'0'}`
- `escolher_leitura`: entre as leituras candidatas do OCR, escolhe a melhor por extração em formato válido > menos correções > maior confiança do OCR.

Status final no `ProcessarEventoUseCase`, conforme validade e `confianca_ocr` da leitura escolhida frente ao limiar `OCR_CONF_MINIMA_SUCESSO` (padrão `0.5`):
- **`sucesso`**: formato válido e `confianca_ocr >= OCR_CONF_MINIMA_SUCESSO`
- **`revisar`**: formato válido mas `confianca_ocr` abaixo do limiar
- **`filtrado`**: nenhuma leitura em formato BR válido (ou OCR não identificou nenhum caractere)

O payload enviado à API inclui `confianca_ocr` (confiança do OCR, separada de `confianca` = confiança do YOLO).

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
- **Nenhuma leitura em formato BR válido**: status `filtrado` (registrado na API com `motivo_filtro`, não é skip silencioso)
- **API Core indisponível**: log warning + continua processando (não bloqueia fila)
- **Worker restart**: `restart: "no"` em dev (política alinhada ao stack raiz)
- **Crash do worker**: eventos em voo ficam em `camera:portaria:queue:processing` e voltam à fila no próximo start

## Dependências do Serviço

Standalone: nenhuma dependência externa — Redis e MinIO sobem no próprio compose (api-core ausente; POST tolerado com warning). Stack raiz: Redis/MinIO/api-core do compose raiz.
