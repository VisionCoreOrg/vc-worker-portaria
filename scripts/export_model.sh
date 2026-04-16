#!/usr/bin/env bash
# =============================================================================
# scripts/export_model.sh
#
# Exporta o modelo YOLOv8 (.pt) para o formato ONNX (.onnx) usando a imagem
# oficial do Ultralytics via Docker — sem precisar instalar nada localmente.
#
# Uso:
#   chmod +x scripts/export_model.sh
#   ./scripts/export_model.sh
#
# O arquivo gerado será: models/modelo_placas.onnx
# =============================================================================

set -e

MODELO_PT="models/modelo_placas.pt"
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
    yolo export model=/models/modelo_placas.pt format=onnx imgsz=640

echo "✅ Exportado com sucesso: models/modelo_placas.onnx"
