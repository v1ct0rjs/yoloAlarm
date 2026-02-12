#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$ROOT_DIR/yolo_model"
MODEL_PT_PATH="$MODEL_DIR/yolov8n.pt"
MODEL_ONNX_PATH="$MODEL_DIR/yolov8n.onnx"
COCO_PATH="$MODEL_DIR/coco.names"

YOLO_PT_URL="${YOLO_PT_URL:-https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt}"
YOLO_ONNX_URL="${YOLO_ONNX_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx}"
EXPORT_WITH_ULTRALYTICS="${EXPORT_WITH_ULTRALYTICS:-0}"

info(){ echo -e "\033[0;32m[INFO]\033[0m $1"; }
warn(){ echo -e "\033[1;33m[WARN]\033[0m $1"; }
error(){ echo -e "\033[0;31m[ERROR]\033[0m $1"; }

need_downloader(){
  if command -v curl >/dev/null 2>&1; then D='curl';
  elif command -v wget >/dev/null 2>&1; then D='wget';
  else error "Instala curl o wget"; exit 1; fi
}

download(){
  if [ "$D" = 'curl' ]; then curl -fL --retry 3 --retry-delay 2 -o "$2" "$1";
  else wget -O "$2" "$1"; fi
}

echo "============================================================="
echo "   INSTALADOR - Detector de Movimiento + YOLOv8"
echo "============================================================="

if command -v python3 >/dev/null 2>&1; then PY=python3; PIP=pip3;
elif command -v python >/dev/null 2>&1; then PY=python; PIP=pip;
else error "Python 3.8+ no encontrado"; exit 1; fi

need_downloader

GPU_TYPE="CPU"; ONNX_PACKAGE="onnxruntime"
if command -v nvidia-smi >/dev/null 2>&1; then ONNX_PACKAGE="onnxruntime-gpu"; GPU_TYPE="NVIDIA"; fi
info "Python: $($PY --version)"
info "GPU detectada: $GPU_TYPE"

read -p "¿Crear entorno virtual? (recomendado) [S/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-S}
if [[ "$CREATE_VENV" =~ ^[Ss]$ ]]; then
  $PY -m venv "$ROOT_DIR/venv"
  source "$ROOT_DIR/venv/bin/activate"
  PIP=pip
fi

info "Instalando dependencias (modo ligero, sin cache)..."
$PIP install --upgrade pip --quiet --no-cache-dir
$PIP install -r "$ROOT_DIR/requirements.txt" --quiet --no-cache-dir
$PIP install "$ONNX_PACKAGE" --quiet --no-cache-dir

mkdir -p "$MODEL_DIR"
info "Descargando peso YOLOv8n (.pt) en carpeta local..."
download "$YOLO_PT_URL" "$MODEL_PT_PATH"

if [ "$EXPORT_WITH_ULTRALYTICS" = "1" ]; then
  info "EXPORT_WITH_ULTRALYTICS=1 -> intentando convertir PT a ONNX..."
  $PIP install ultralytics --quiet --no-cache-dir
  $PY <<PYEOF
from ultralytics import YOLO
from pathlib import Path
pt = Path(r"$MODEL_PT_PATH")
YOLO(str(pt)).export(format='onnx', imgsz=640, simplify=True)
out = Path('yolov8n.onnx')
if out.exists():
    out.replace(Path(r"$MODEL_ONNX_PATH"))
PYEOF
else
  warn "Saltando conversión PT->ONNX para ahorrar espacio en disco."
  warn "Descargando ONNX preexportado para ejecución inmediata..."
  download "$YOLO_ONNX_URL" "$MODEL_ONNX_PATH"
fi

info "Descargando etiquetas COCO..."
download "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" "$COCO_PATH"

$PY - <<PYEOF
import os, cv2, numpy, onnxruntime
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('ONNX Runtime:', onnxruntime.__version__)
for p in [r"$MODEL_PT_PATH", r"$MODEL_ONNX_PATH", r"$COCO_PATH"]:
    assert os.path.exists(p), f'No existe: {p}'
print('✓ Modelo y etiquetas listos en yolo_model/')
PYEOF

info "Instalación completada. Ejecuta: python main.py"
