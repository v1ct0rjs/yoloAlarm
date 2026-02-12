# Detector de Movimiento + YOLOv8

Sistema de detección de movimiento e identificación de objetos (personas, perros, gatos) usando YOLOv8 con soporte para GPU.

## Características

-  **Detección de movimiento** en tiempo real
-  **YOLOv8** para identificar personas, perros y gatos
-  **Soporte GPU** automático (NVIDIA CUDA)
-  **Optimizado** para video en tiempo real y streaming
- ️ **Controles en tiempo real** para ajustar sensibilidad

## Requisitos

- **Python** 3.8 o superior
- **Sistema operativo:** Windows 10/11, Linux (cualquier distro), macOS
- **GPU (opcional):** NVIDIA con CUDA para aceleración

## Instalación Rápida

### Linux / macOS

```bash
# Dar permisos y ejecutar
chmod +x instalar.sh
./instalar.sh
```

### Windows

```batch
# Doble clic en instalar.bat o ejecutar en CMD:
instalar.bat
```

## Archivos del Proyecto

```
proyecto/
├── main.py  # Programa principal
├── install.sh                  # Instalador Linux/macOS
├── install.bat                 # Instalador Windows
├── requirements.txt             # Dependencias
├── README.md                    # Este archivo
└── yolo_model/                  # (creado automáticamente)
    ├── yolov8n.onnx            # Modelo YOLOv8
    └── coco.names              # Etiquetas de clases
```

## Uso

### 1. Activar entorno virtual (si lo creaste)

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Configurar fuente de video

Edita `detector_movimiento_yolo.py` y cambia la variable `URL`:

```python
# Para archivo local:
URL = "mi_video.mp4"

# Para streaming:
URL = "https://url-del-streaming/video"

# Para webcam:
URL = 0  # o 1 si tienes múltiples cámaras
```

### 3. Ejecutar

```bash
python detector_movimiento_yolo.py
```

## Controles

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir |
| `P` | Pausar / Reanudar |
| `R` | Reiniciar video |
| `S` | Subir sensibilidad YOLO |
| `W` | Bajar sensibilidad YOLO |
| `A` | Subir sensibilidad movimiento |
| `D` | Bajar sensibilidad movimiento |
| `Q` | YOLO más frecuente |
| `E` | YOLO menos frecuente |

## Colores de Detección

| Objeto | Color |
|--------|-------|
| 🟢 Persona | Verde |
| 🟠 Perro | Naranja |
| 🟣 Gato | Magenta |

## Configuración Avanzada

### Añadir más clases de detección

Edita el diccionario `CLASES_DETECTAR` en el código:

```python
CLASES_DETECTAR = {
    "person": {"nombre": "PERSONA", "color": (0, 255, 0)},
    "dog": {"nombre": "PERRO", "color": (0, 165, 255)},
    "cat": {"nombre": "GATO", "color": (255, 0, 255)},
    "car": {"nombre": "COCHE", "color": (255, 0, 0)},  # Añadir
}

# Y añadir el ID en IDS_DETECTAR (ver lista COCO)
IDS_DETECTAR = {0: "person", 15: "cat", 16: "dog", 2: "car"}
```

### IDs de clases COCO comunes

| ID | Clase | ID | Clase |
|----|-------|----|-------|
| 0 | person | 2 | car |
| 1 | bicycle | 3 | motorcycle |
| 5 | bus | 7 | truck |
| 14 | bird | 15 | cat |
| 16 | dog | 17 | horse |

## Solución de Problemas

### "No se detectó GPU"

1. Verifica que tienes drivers NVIDIA: `nvidia-smi`
2. Reinstala onnxruntime-gpu: `pip install onnxruntime-gpu --force-reinstall`

### "No module named cv2"

```bash
pip install opencv-python
```

### "No module named onnxruntime"

```bash
# Con GPU NVIDIA:
pip install onnxruntime-gpu

# Sin GPU:
pip install onnxruntime
```

### Video entrecortado

- Aumenta el intervalo de YOLO con la tecla `E`
- O usa un modelo más pequeño

## Rendimiento Esperado

| Hardware | Tiempo/frame | FPS YOLO |
|----------|-------------|----------|
| CPU (i7) | ~80ms | ~12 |
| GPU (GTX 1060) | ~15ms | ~66 |
| GPU (RTX 3090) | ~5ms | ~200 |

## Créditos

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)