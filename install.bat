@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
set MODEL_DIR=%ROOT_DIR%yolo_model
set MODEL_PT_PATH=%MODEL_DIR%\yolov8n.pt
set MODEL_ONNX_PATH=%MODEL_DIR%\yolov8n.onnx
set COCO_PATH=%MODEL_DIR%\coco.names

if "%YOLO_PT_URL%"=="" set YOLO_PT_URL=https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
if "%YOLO_ONNX_URL%"=="" set YOLO_ONNX_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
if "%EXPORT_WITH_ULTRALYTICS%"=="" set EXPORT_WITH_ULTRALYTICS=0

echo =============================================================
echo    INSTALADOR - Detector de Movimiento + YOLOv8
echo =============================================================

python --version >nul 2>&1 || (echo [ERROR] Python 3.8+ no encontrado & pause & exit /b 1)

set ONNX_PACKAGE=onnxruntime
nvidia-smi >nul 2>&1 && set ONNX_PACKAGE=onnxruntime-gpu

set /p CREATE_VENV=Crear entorno virtual? (S/n): 
if /i "%CREATE_VENV%"=="" set CREATE_VENV=S
if /i "%CREATE_VENV%"=="S" (
    python -m venv "%ROOT_DIR%venv"
    call "%ROOT_DIR%venv\Scripts\activate.bat"
)

echo [INFO] Instalando dependencias (modo ligero, sin cache)...
pip install --upgrade pip --quiet --no-cache-dir
pip install -r "%ROOT_DIR%requirements.txt" --quiet --no-cache-dir
pip install %ONNX_PACKAGE% --quiet --no-cache-dir

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo [INFO] Descargando yolov8n.pt...
curl -fL -o "%MODEL_PT_PATH%" "%YOLO_PT_URL%" >nul 2>&1 || (echo [ERROR] No se pudo descargar yolov8n.pt & pause & exit /b 1)

if "%EXPORT_WITH_ULTRALYTICS%"=="1" (
    echo [INFO] Convirtiendo PT a ONNX con ultralytics...
    pip install ultralytics --quiet --no-cache-dir
    python -c "from ultralytics import YOLO; from pathlib import Path; YOLO(str(Path(r'%MODEL_PT_PATH%'))).export(format='onnx', imgsz=640, simplify=True); o=Path('yolov8n.onnx'); o.replace(Path(r'%MODEL_ONNX_PATH%')) if o.exists() else None" || (echo [ERROR] Fallo conversion & pause & exit /b 1)
) else (
    echo [WARN] Saltando conversion para ahorrar espacio.
    echo [INFO] Descargando ONNX preexportado...
    curl -fL -o "%MODEL_ONNX_PATH%" "%YOLO_ONNX_URL%" >nul 2>&1 || (echo [ERROR] No se pudo descargar ONNX & pause & exit /b 1)
)

curl -fL -o "%COCO_PATH%" https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names >nul 2>&1 || (echo [ERROR] No se pudo descargar coco.names & pause & exit /b 1)

python -c "import os, cv2, numpy, onnxruntime; print('OpenCV',cv2.__version__); print('NumPy',numpy.__version__); print('ONNX Runtime',onnxruntime.__version__); assert os.path.exists(r'%MODEL_PT_PATH%'); assert os.path.exists(r'%MODEL_ONNX_PATH%'); assert os.path.exists(r'%COCO_PATH%'); print('Modelo y etiquetas listos en yolo_model/')" || (echo [ERROR] Verificacion final fallo & pause & exit /b 1)

echo.
echo Instalacion completada. Ejecuta: python main.py
pause
