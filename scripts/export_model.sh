#!/usr/bin/env bash
# =============================================================================
# scripts/export_model.sh
#
# Exporta um modelo YOLOv8 (.pt) para o formato ONNX (.onnx) usando a imagem
# oficial do Ultralytics via Docker — sem precisar instalar nada localmente.
#
# Uso:
#   chmod +x scripts/export_model.sh
#   ./scripts/export_model.sh [nome_do_modelo]   # sem .pt; default: modelo_placas_yasir
#
# Exemplos:
#   ./scripts/export_model.sh                     # models/modelo_placas_yasir.pt → .onnx (detector padrão)
#   ./scripts/export_model.sh modelo_placas       # models/modelo_placas.pt → .onnx (Koushim, fallback)
#
# O detector padrão (modelo_placas_yasir) vem de:
#   https://huggingface.co/yasirfaizahmed/license-plate-object-detection (Apache-2.0)
# Baixe o peso antes de exportar:
#   curl -L -o models/modelo_placas_yasir.pt \
#     https://huggingface.co/yasirfaizahmed/license-plate-object-detection/resolve/main/best.pt
# =============================================================================

set -e

NOME_MODELO="${1:-modelo_placas_yasir}"
MODELO_PT="models/${NOME_MODELO}.pt"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"

if [ ! -f "$PROJECT_DIR/$MODELO_PT" ]; then
    echo "❌ Modelo não encontrado: $PROJECT_DIR/$MODELO_PT"
    echo "   Certifique-se de que o arquivo .pt está na pasta models/ antes de exportar."
    exit 1
fi

echo "🔄 Exportando $MODELO_PT para ONNX..."

docker run --rm \
    -v "$PROJECT_DIR/models:/models" \
    ultralytics/ultralytics:latest \
    yolo export model="/models/${NOME_MODELO}.pt" format=onnx imgsz=640

echo "✅ Exportado com sucesso: models/${NOME_MODELO}.onnx"
