@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
set MODEL_DIR=%ROOT_DIR%yolo_model
set MODEL_PT_PATH=%MODEL_DIR%\yolov8n.pt
set MODEL_ONNX_PATH=%MODEL_DIR%\yolov8n.onnx
set COCO_PATH=%MODEL_DIR%\coco.names

echo =============================================================
echo    INSTALADOR - Detector de Movimiento + YOLOv8
echo =============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.8+ desde python.org
    pause
    exit /b 1
)

set ONNX_PACKAGE=onnxruntime
nvidia-smi >nul 2>&1
if not errorlevel 1 set ONNX_PACKAGE=onnxruntime-gpu

set /p CREATE_VENV=Crear entorno virtual? (S/n): 
if /i "%CREATE_VENV%"=="" set CREATE_VENV=S
if /i "%CREATE_VENV%"=="S" (
    python -m venv "%ROOT_DIR%venv"
    call "%ROOT_DIR%venv\Scripts\activate.bat"
)

echo [INFO] Instalando dependencias...
pip install --upgrade pip --quiet
pip install -r "%ROOT_DIR%requirements.txt" --quiet
pip install %ONNX_PACKAGE% ultralytics --quiet

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

set URL1=https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
set URL2=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
set URL3=https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
if not "%YOLO_MODEL_URL%"=="" (
    set URL1=%YOLO_MODEL_URL%
    set URL2=
    set URL3=
)

set DOWNLOAD_OK=0
for %%U in ("!URL1!" "!URL2!" "!URL3!") do (
    if not "%%~U"=="" (
        echo [INFO] Descargando: %%~U
        curl -fL -o "%MODEL_PT_PATH%" "%%~U" >nul 2>&1
        if not errorlevel 1 (
            set DOWNLOAD_OK=1
            goto :pt_downloaded
        )
    )
)

:pt_downloaded
if "%DOWNLOAD_OK%"=="0" (
    echo [ERROR] No se pudo descargar yolov8n.pt
    pause
    exit /b 1
)

echo [INFO] Convirtiendo yolov8n.pt a ONNX en yolo_model...
python -c "from ultralytics import YOLO; from pathlib import Path; pt=Path(r'%MODEL_PT_PATH%'); YOLO(str(pt)).export(format='onnx', imgsz=640, simplify=True); root=Path('yolov8n.onnx'); root.replace(Path(r'%MODEL_ONNX_PATH%')) if root.exists() else None"
if errorlevel 1 (
    echo [ERROR] Fallo conversion a ONNX
    pause
    exit /b 1
)

curl -fL -o "%COCO_PATH%" https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se pudo descargar coco.names
    pause
    exit /b 1
)

python -c "import os, cv2, numpy, onnxruntime; print('OpenCV',cv2.__version__); print('NumPy',numpy.__version__); print('ONNX Runtime',onnxruntime.__version__); assert os.path.exists(r'%MODEL_PT_PATH%'); assert os.path.exists(r'%MODEL_ONNX_PATH%'); assert os.path.exists(r'%COCO_PATH%'); print('Modelo PT+ONNX y etiquetas OK')"
if errorlevel 1 (
    echo [ERROR] Verificacion final fallo
    pause
    exit /b 1
)

echo.
echo Instalacion completada. Ejecuta: python main.py
pause
