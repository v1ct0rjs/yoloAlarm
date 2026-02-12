#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$ROOT_DIR/yolo_model"
MODEL_PT_PATH="$MODEL_DIR/yolov8n.pt"
MODEL_ONNX_PATH="$MODEL_DIR/yolov8n.onnx"
COCO_PATH="$MODEL_DIR/coco.names"
YOLO_MODEL_URL="${YOLO_MODEL_URL:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

require_downloader() {
    if command -v curl &> /dev/null; then DOWNLOADER="curl";
    elif command -v wget &> /dev/null; then DOWNLOADER="wget";
    else error "Se necesita curl o wget para descargar YOLOv8."; exit 1; fi
}

download_file() {
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fL --retry 3 --retry-delay 2 -o "$2" "$1"
    else
        wget -O "$2" "$1"
    fi
}

echo "============================================================="
echo "   INSTALADOR - Detector de Movimiento + YOLOv8"
echo "============================================================="

info "Verificando Python..."
if command -v python3 &> /dev/null; then PYTHON=python3; PIP=pip3;
elif command -v python &> /dev/null; then PYTHON=python; PIP=pip;
else error "Python no encontrado. Por favor instala Python 3.8+"; exit 1; fi
info "Python encontrado: $($PYTHON --version)"
require_downloader

info "Detectando GPU..."
GPU_TYPE="CPU"; ONNX_PACKAGE="onnxruntime"
if command -v nvidia-smi &> /dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$GPU_NAME" ]; then GPU_TYPE="NVIDIA"; ONNX_PACKAGE="onnxruntime-gpu"; info "GPU NVIDIA detectada: $GPU_NAME"; fi
fi
[ "$GPU_TYPE" = "CPU" ] && warn "No se detectó GPU compatible. Se usará CPU."

read -p "¿Crear entorno virtual? (recomendado) [S/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-S}
if [[ "$CREATE_VENV" =~ ^[Ss]$ ]]; then
  info "Creando entorno virtual..."
  $PYTHON -m venv "$ROOT_DIR/venv"
  source "$ROOT_DIR/venv/bin/activate"
  PIP="pip"
fi

info "Instalando dependencias..."
$PIP install --upgrade pip --quiet
$PIP install -r "$ROOT_DIR/requirements.txt" --quiet
$PIP install "$ONNX_PACKAGE" ultralytics --quiet

mkdir -p "$MODEL_DIR"
if [ -n "$YOLO_MODEL_URL" ]; then
    YOLO_MODEL_URLS=("$YOLO_MODEL_URL")
else
    YOLO_MODEL_URLS=(
      "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
      "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    )
fi

info "Descargando YOLOv8n desde releases de Ultralytics (independiente)..."
DOWNLOAD_OK=0
for url in "${YOLO_MODEL_URLS[@]}"; do
  info "Intentando: $url"
  if download_file "$url" "$MODEL_PT_PATH"; then DOWNLOAD_OK=1; break; fi
  warn "No se pudo descargar desde esa URL, probando alternativa..."
done
[ "$DOWNLOAD_OK" -eq 1 ] || { error "No se pudo descargar yolov8n.pt"; exit 1; }

info "Convirtiendo yolov8n.pt a ONNX en su carpeta de modelo..."
$PYTHON <<PYEOF
from ultralytics import YOLO
from pathlib import Path
pt = Path(r"$MODEL_PT_PATH")
out_dir = pt.parent
model = YOLO(str(pt))
model.export(format="onnx", imgsz=640, simplify=True)
root_onnx = Path("yolov8n.onnx")
if root_onnx.exists():
    root_onnx.replace(out_dir / "yolov8n.onnx")
print("ONNX listo en", out_dir / "yolov8n.onnx")
PYEOF

info "Descargando etiquetas COCO..."
download_file "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" "$COCO_PATH"

info "Verificando instalación..."
$PYTHON - <<PYEOF
import os, cv2, numpy, onnxruntime
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('ONNX Runtime:', onnxruntime.__version__)
assert os.path.exists(r"$MODEL_ONNX_PATH"), 'No existe yolov8n.onnx'
assert os.path.exists(r"$MODEL_PT_PATH"), 'No existe yolov8n.pt'
assert os.path.exists(r"$COCO_PATH"), 'No existe coco.names'
print('Modelo PT+ONNX y etiquetas OK')
PYEOF

echo "\nInstalación completada. Ejecuta: python main.py"
