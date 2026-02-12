# Detector de Movimiento + YOLOv8

Sistema de detección de movimiento e identificación de objetos (personas, perros, gatos) usando YOLOv8 con soporte para GPU y alertas por Telegram/Email.

## Características

- Detección de movimiento en tiempo real
- YOLOv8 para identificar personas, perros y gatos
- Soporte GPU automático (NVIDIA CUDA)
- Alertas por Telegram con foto
- Alertas por Email con foto adjunta
- Horarios programables para activar/desactivar alarma
- Bot de Telegram para control remoto
- Optimizado para video en tiempo real y streaming

## Requisitos

- Python 3.8 o superior
- Sistema operativo: Windows 10/11, Linux, macOS
- GPU (opcional): NVIDIA con CUDA para aceleración

## Instalación

### Linux / macOS

```bash
chmod +x instalar.sh
./instalar.sh
```

### Windows

```batch
instalar.bat
```

## Configuración de Telegram

### 1. Crear el bot

1. Abre Telegram y busca `@BotFather`
2. Envía `/newbot`
3. Sigue las instrucciones (nombre y username)
4. Copia el token que te da

### 2. Obtener tu Chat ID

1. Añade el token en tu archivo `.env`
2. Envía `/start` a tu nuevo bot en Telegram
3. Ejecuta:
```bash
python obtener_chat_id.py
```
4. Copia el Chat ID que aparece

### 3. Configurar .env

```bash
cp .env.ejemplo .env
```

Edita el archivo con tus datos:

```env
TELEGRAM_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID="123456789"
```

## Comandos del Bot

| Comando | Descripción |
|---------|-------------|
| `/activar` | Activar alarma (forzado) |
| `/desactivar` | Desactivar alarma (forzado) |
| `/auto` | Modo automático (usa horario) |
| `/estado` | Ver estado actual |
| `/horario` | Ver horario configurado |
| `/sethorario HH:MM HH:MM` | Cambiar horario |
| `/setdias 1,2,3,4,5` | Cambiar días activos |
| `/foto` | Obtener captura actual |
| `/help` | Ver comandos |

### Ejemplos de configuración por Telegram

```
/sethorario 08:00 22:00      → Alertas de 8:00 a 22:00
/sethorario 22:00 06:00      → Horario nocturno
/setdias 1,2,3,4,5           → Lunes a Viernes
/setdias 6,7                 → Solo fines de semana
```

Los días son: 1=Lun, 2=Mar, 3=Mié, 4=Jue, 5=Vie, 6=Sáb, 7=Dom

## Configuración de Horarios

En el archivo `.env`:

```env
HORARIO_INICIO="08:00"
HORARIO_FIN="22:00"
DIAS_ACTIVOS="1,2,3,4,5,6,7"
```

Dejar vacío `HORARIO_INICIO` y `HORARIO_FIN` para alertas 24/7.

## Configuración de Email (Opcional)

1. Activa verificación en 2 pasos en Gmail
2. Crea una contraseña de aplicación en https://myaccount.google.com/apppasswords
3. Configura en `.env`:

```env
GMAIL_CUENTA="tu_correo@gmail.com"
GMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
```

## Archivos del Proyecto

```
proyecto/
├── detector_movimiento_yolo.py  # Programa principal
├── obtener_chat_id.py           # Utilidad para Telegram
├── instalar.sh                  # Instalador Linux/macOS
├── instalar.bat                 # Instalador Windows
├── requirements.txt             # Dependencias
├── .env.ejemplo                 # Plantilla de configuración
├── .env                         # Tu configuración (crear)
└── yolo_model/                  # (creado automáticamente)
    ├── yolov8n.onnx
    └── coco.names
```

## Uso

### Configurar fuente de video

Edita `detector_movimiento_yolo.py` y cambia la variable `URL`:

```python
# Archivo local:
URL = "mi_video.mp4"

# Streaming:
URL = "https://url-del-streaming/video"

# Webcam:
URL = 0
```

### Ejecutar

```bash
python detector_movimiento_yolo.py
```

## Controles de Teclado

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir |
| `P` | Pausar / Reanudar |
| `R` | Reiniciar video |
| `S/W` | Ajustar sensibilidad YOLO |
| `A/D` | Ajustar sensibilidad movimiento |
| `Q/E` | Ajustar frecuencia YOLO |
| `M` | Ciclar alarma: ON → OFF → AUTO |

## Detecciones

| Objeto | Color en pantalla |
|--------|-------------------|
| Persona | Verde |
| Perro | Naranja |
| Gato | Magenta |

## Añadir más clases

Edita el diccionario `CLASES_DETECTAR` en el código:

```python
CLASES_DETECTAR = {
    "person": {"nombre": "PERSONA", "color": (0, 255, 0)},
    "dog": {"nombre": "PERRO", "color": (0, 165, 255)},
    "cat": {"nombre": "GATO", "color": (255, 0, 255)},
    "car": {"nombre": "COCHE", "color": (255, 0, 0)},
}

IDS_DETECTAR = {0: "person", 15: "cat", 16: "dog", 2: "car"}
```

### IDs de clases COCO

| ID | Clase | ID | Clase |
|----|-------|----|-------|
| 0 | person | 2 | car |
| 1 | bicycle | 3 | motorcycle |
| 5 | bus | 7 | truck |
| 14 | bird | 15 | cat |
| 16 | dog | 17 | horse |

## Solución de Problemas

**No se detectó GPU**
```bash
nvidia-smi                              # Verificar drivers
pip install onnxruntime-gpu --force     # Reinstalar
```

**No module named cv2**
```bash
pip install opencv-python
```

**No module named onnxruntime**
```bash
pip install onnxruntime-gpu    # Con GPU
pip install onnxruntime        # Sin GPU
```

**Video entrecortado**: Aumenta el intervalo de YOLO con la tecla `E`.

## Rendimiento

| Hardware | Tiempo/frame | FPS |
|----------|-------------|-----|
| CPU (i7) | ~80ms | ~12 |
| GTX 1060 | ~15ms | ~66 |
| RTX 3090 | ~5ms | ~200 |

## Licencia

GNU GPLv3

## Créditos

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)