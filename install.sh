#!/bin/bash
# =============================================================
# INSTALADOR AUTOMÁTICO - Detector de Movimiento + YOLOv8
# Compatible con: Linux (Arch, Ubuntu, Debian, Fedora) y macOS
# =============================================================

set -e

echo "============================================================="
echo "   INSTALADOR - Detector de Movimiento + YOLOv8"
echo "============================================================="
echo ""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================
# 1. VERIFICAR PYTHON
# =============================================================
info "Verificando Python..."

if command -v python3 &> /dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PYTHON=python
    PIP=pip
else
    error "Python no encontrado. Por favor instala Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
info "Python encontrado: $($PYTHON --version)"

# =============================================================
# 2. DETECTAR GPU
# =============================================================
info "Detectando GPU..."

GPU_TYPE="CPU"
ONNX_PACKAGE="onnxruntime"

# Detectar NVIDIA
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        GPU_TYPE="NVIDIA"
        ONNX_PACKAGE="onnxruntime-gpu"
        info "GPU NVIDIA detectada: $GPU_NAME"
    fi
fi

# Detectar AMD (ROCm) en Linux
if [ "$GPU_TYPE" = "CPU" ] && [ -d "/opt/rocm" ]; then
    GPU_TYPE="AMD"
    # ROCm usa onnxruntime específico
    warn "GPU AMD detectada. ONNX Runtime para ROCm requiere instalación manual."
    warn "Ver: https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html"
    ONNX_PACKAGE="onnxruntime"
fi

# Detectar macOS con Apple Silicon
if [ "$(uname)" = "Darwin" ]; then
    CHIP=$(uname -m)
    if [ "$CHIP" = "arm64" ]; then
        GPU_TYPE="Apple Silicon"
        # CoreML provider viene incluido en onnxruntime para macOS
        ONNX_PACKAGE="onnxruntime"
        info "Apple Silicon detectado (M1/M2/M3)"
    else
        info "macOS Intel detectado"
    fi
fi

if [ "$GPU_TYPE" = "CPU" ]; then
    warn "No se detectó GPU compatible. Se usará CPU."
fi

echo ""
echo "============================================================="
echo "   Configuración detectada:"
echo "   - Sistema: $(uname -s)"
echo "   - GPU: $GPU_TYPE"
echo "   - ONNX Runtime: $ONNX_PACKAGE"
echo "============================================================="
echo ""

# =============================================================
# 3. CREAR ENTORNO VIRTUAL (opcional)
# =============================================================
read -p "¿Crear entorno virtual? (recomendado) [S/n]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-S}

if [[ "$CREATE_VENV" =~ ^[Ss]$ ]]; then
    info "Creando entorno virtual..."
    $PYTHON -m venv venv

    # Activar según sistema
    if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
        source venv/bin/activate
    fi

    info "Entorno virtual creado y activado"
    PIP="pip"
else
    warn "Instalando en el sistema global..."
fi

# =============================================================
# 4. INSTALAR DEPENDENCIAS
# =============================================================
info "Instalando dependencias..."

# Actualizar pip
$PIP install --upgrade pip --quiet

# Instalar dependencias base
$PIP install opencv-python numpy --quiet
info "OpenCV y NumPy instalados"

# Instalar ONNX Runtime (GPU o CPU)
info "Instalando $ONNX_PACKAGE..."
$PIP install $ONNX_PACKAGE --quiet

# Verificar ONNX Runtime
$PYTHON -c "import onnxruntime as ort; providers = ort.get_available_providers(); print(f'ONNX Runtime OK - Providers: {providers}')"

# =============================================================
# 5. DESCARGAR MODELO YOLOv8
# =============================================================
info "Descargando modelo YOLOv8..."

# Instalar ultralytics temporalmente para convertir
$PIP install ultralytics --quiet

mkdir -p yolo_model

$PYTHON << 'EOF'
import os
import shutil

try:
    from ultralytics import YOLO

    print("Descargando YOLOv8n...")
    model = YOLO("yolov8n.pt")

    print("Convirtiendo a ONNX...")
    model.export(format="onnx", imgsz=640, simplify=True)

    if os.path.exists("yolov8n.onnx"):
        shutil.move("yolov8n.onnx", "yolo_model/yolov8n.onnx")
        print("✓ Modelo guardado en: yolo_model/yolov8n.onnx")

    # Limpiar
    for f in ["yolov8n.pt"]:
        if os.path.exists(f):
            os.remove(f)

except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOF

# Descargar nombres de clases
info "Descargando etiquetas COCO..."
curl -sL -o yolo_model/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# =============================================================
# 6. VERIFICAR INSTALACIÓN
# =============================================================
echo ""
info "Verificando instalación..."

$PYTHON << 'EOF'
import sys

# Verificar imports
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError:
    print("✗ OpenCV no instalado")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError:
    print("✗ NumPy no instalado")
    sys.exit(1)

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"✓ ONNX Runtime: {ort.__version__}")

    if 'CUDAExecutionProvider' in providers:
        print("  └─ GPU CUDA disponible")
    elif 'CoreMLExecutionProvider' in providers:
        print("  └─ Apple CoreML disponible")
    else:
        print("  └─ Usando CPU")
except ImportError:
    print("✗ ONNX Runtime no instalado")
    sys.exit(1)

import os
if os.path.exists("yolo_model/yolov8n.onnx"):
    size = os.path.getsize("yolo_model/yolov8n.onnx") / (1024*1024)
    print(f"✓ Modelo YOLOv8: {size:.1f} MB")
else:
    print("✗ Modelo no encontrado")
    sys.exit(1)

print("\n✓ Instalación completada correctamente!")
EOF

# =============================================================
# 7. INSTRUCCIONES FINALES
# =============================================================
echo ""
echo "============================================================="
echo "   ¡INSTALACIÓN COMPLETADA!"
echo "============================================================="
echo ""
echo "Para ejecutar el detector:"
echo ""

if [[ "$CREATE_VENV" =~ ^[Ss]$ ]]; then
    echo "  1. Activar entorno virtual:"
    echo "     source venv/bin/activate"
    echo ""
    echo "  2. Ejecutar:"
fi

echo "     python detector_movimiento_yolo.py"
echo ""
echo "Controles:"
echo "  ESC   = Salir"
echo "  P     = Pausar"
echo "  R     = Reiniciar video"
echo "  S/W   = Sensibilidad YOLO"
echo "  A/D   = Sensibilidad movimiento"
echo "============================================================="