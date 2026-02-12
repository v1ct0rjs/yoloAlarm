@echo off
REM =============================================================
REM INSTALADOR AUTOMÁTICO - Detector de Movimiento + YOLOv8
REM Compatible con: Windows 10/11
REM =============================================================

echo =============================================================
echo    INSTALADOR - Detector de Movimiento + YOLOv8
echo =============================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.8+ desde python.org
    pause
    exit /b 1
)

echo [INFO] Python encontrado
python --version

REM Detectar GPU NVIDIA
set GPU_TYPE=CPU
set ONNX_PACKAGE=onnxruntime

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [INFO] GPU NVIDIA detectada
    set GPU_TYPE=NVIDIA
    set ONNX_PACKAGE=onnxruntime-gpu
) else (
    echo [WARN] No se detectó GPU NVIDIA. Se usará CPU.
)

echo.
echo =============================================================
echo    GPU: %GPU_TYPE%
echo    ONNX Runtime: %ONNX_PACKAGE%
echo =============================================================
echo.

REM Crear entorno virtual
set /p CREATE_VENV="Crear entorno virtual? (S/n): "
if /i "%CREATE_VENV%"=="" set CREATE_VENV=S

if /i "%CREATE_VENV%"=="S" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Entorno virtual activado
)

REM Instalar dependencias
echo [INFO] Instalando dependencias...
pip install --upgrade pip --quiet
pip install opencv-python numpy --quiet
echo [INFO] OpenCV y NumPy instalados

echo [INFO] Instalando %ONNX_PACKAGE%...
pip install %ONNX_PACKAGE% --quiet

REM Descargar modelo
echo [INFO] Descargando modelo YOLOv8...
pip install ultralytics --quiet

if not exist yolo_model mkdir yolo_model

python -c "from ultralytics import YOLO; import shutil, os; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=640, simplify=True); shutil.move('yolov8n.onnx', 'yolo_model/yolov8n.onnx') if os.path.exists('yolov8n.onnx') else None; os.remove('yolov8n.pt') if os.path.exists('yolov8n.pt') else None"

REM Descargar etiquetas
echo [INFO] Descargando etiquetas COCO...
curl -sL -o yolo_model\coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

REM Verificar
echo.
echo [INFO] Verificando instalación...
python -c "import cv2, numpy, onnxruntime; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('ONNX Runtime:', onnxruntime.__version__); print('Providers:', onnxruntime.get_available_providers())"

echo.
echo =============================================================
echo    INSTALACION COMPLETADA!
echo =============================================================
echo.
echo Para ejecutar:

if /i "%CREATE_VENV%"=="S" (
    echo   1. Activar entorno: venv\Scripts\activate
    echo   2. Ejecutar: python detector_movimiento_yolo.py
) else (
    echo   python detector_movimiento_yolo.py
)

echo.
pause