# Detector de Movimiento + YOLOv8

Sistema de detección de movimiento e identificación de objetos (personas, perros, gatos) usando YOLOv8 con soporte para GPU y alertas por Telegram/Email.

## Descripción General

Este sistema combina detección de movimiento tradicional (OpenCV BackgroundSubtractor) con inteligencia artificial (YOLOv8) para crear un sistema de vigilancia inteligente que puede:

- **Detectar movimiento** en tiempo real usando algoritmos de sustracción de fondo
- **Identificar objetos específicos** (personas, perros, gatos) usando YOLOv8 con ONNX Runtime
- **Enviar alertas automáticas** por Telegram y/o Email cuando detecta personas
- **Control remoto** mediante bot de Telegram para activar/desactivar alarma
- **Horarios programables** para activar el sistema solo en ciertos momentos
- **Optimizado para rendimiento** con soporte GPU automático (NVIDIA CUDA)

## Cómo Funciona

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      FUENTE DE VIDEO                        │
│           (Webcam / Archivo / Stream RTSP/HTTP)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              DETECCIÓN DE MOVIMIENTO (OpenCV)               │
│   • BackgroundSubtractorMOG2                                │
│   • Filtrado morfológico                                     │
│   • Análisis de contornos                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ ¿Movimiento?     │
              └────┬─────────┬───┘
                   │ NO      │ SÍ
                   │         ▼
                   │    ┌─────────────────────────────────┐
                   │    │   DETECCIÓN YOLOv8 (ONNX)       │
                   │    │ • Inferencia cada N frames      │
                   │    │ • GPU automática (CUDA)         │
                   │    │ • Filtrado de clases            │
                   │    │ • NMS (Non-Maximum Suppression) │
                   │    └────────┬────────────────────────┘
                   │             │
                   │             ▼
                   │    ┌──────────────────┐
                   │    │ ¿Persona?        │
                   │    └────┬─────────┬───┘
                   │         │ NO      │ SÍ
                   │         │         ▼
                   │         │    ┌──────────────────────┐
                   │         │    │ VERIFICAR HORARIO    │
                   │         │    │ VERIFICAR COOLDOWN   │
                   │         │    └────┬─────────────────┘
                   │         │         │
                   │         │         ▼
                   │         │    ┌──────────────────────┐
                   │         │    │ ENVIAR ALERTAS       │
                   │         │    │ • Telegram (foto)    │
                   │         │    │ • Email (adjunta)    │
                   │         │    └──────────────────────┘
                   │         │
                   └─────────┴────────────────────────────────┐
                                                              │
                                                              ▼
                                             ┌─────────────────────────────┐
                                             │  VISUALIZACIÓN EN PANTALLA  │
                                             │ • Bounding boxes            │
                                             │ • Panel de información      │
                                             │ • Estado de alarma          │
                                             └─────────────────────────────┘
```

### Flujo de Procesamiento

#### 1. **Captura de Video**
```python
# El sistema lee frames de diferentes fuentes:
URL = 0                    # Webcam
URL = "video.mp4"          # Archivo local
URL = "https://..."        # Stream HTTP/RTSP
```

#### 2. **Detección de Movimiento (Optimización)**
- Usa `BackgroundSubtractorMOG2` para crear una máscara de píxeles en movimiento
- Aplica operaciones morfológicas (apertura, dilatación) para reducir ruido
- Calcula contornos y filtra por área mínima
- **Propósito**: Evitar ejecutar YOLOv8 en frames estáticos (ahorro de CPU/GPU)

```python
# Solo si hay movimiento significativo → ejecutar YOLOv8
if movimiento_detectado and frame_count % YOLO_INTERVAL == 0:
    detectar_yolov8(frame)
```

#### 3. **Detección con YOLOv8**
- **Preprocesamiento**: 
  - Redimensiona frame a 640x640 (letterbox con padding)
  - Normaliza valores a 0-1
  - Convierte formato HWC → CHW (Height, Width, Channels)
  
- **Inferencia ONNX**:
  - Ejecuta modelo YOLOv8n.onnx usando ONNX Runtime
  - Utiliza GPU automáticamente si está disponible (CUDA)
  - Salida: tensor [1, 84, 8400] con 8400 detecciones posibles

- **Postprocesamiento**:
  - Filtra detecciones por confianza (threshold ajustable)
  - Aplica NMS (Non-Maximum Suppression) para eliminar duplicados
  - Solo conserva clases configuradas (person, dog, cat)

```python
# Detecciones filtradas
CLASES_DETECTAR = {
    "person": {"nombre": "PERSONA", "color": (0, 255, 0)},
    "dog": {"nombre": "PERRO", "color": (0, 165, 255)},
    "cat": {"nombre": "GATO", "color": (255, 0, 255)}
}
```

#### 4. **Sistema de Alertas**
- **Verificación de condiciones**:
  - ¿Se detectó una PERSONA?
  - ¿La alarma está activa? (horario + día + estado forzado)
  - ¿Ha pasado el tiempo de cooldown? (evita spam)

- **Envío multihilo**:
  - Captura el frame actual
  - Lanza thread separado para no bloquear el video
  - Envía por Telegram (foto con caption)
  - Envía por Email (foto adjunta en MIME)

```python
# Cooldown de 60 segundos entre alertas
if (ahora - ultima_alerta).total_seconds() >= COOLDOWN_ALERTAS:
    enviar_alertas(frame, objetos_detectados)
```

#### 5. **Bot de Telegram (Control Remoto)**
- Thread daemon escuchando comandos 24/7
- Usa long polling (`getUpdates` con timeout=30s)
- Comandos procesados:
  - `/activar` → Fuerza alarma ON
  - `/desactivar` → Fuerza alarma OFF
  - `/auto` → Usa horario programado
  - `/foto` → Captura instantánea
  - `/sethorario HH:MM HH:MM` → Cambia horario
  - `/setdias 1,2,3,4,5` → Cambia días activos

```python
# El bot modifica variables globales
ALARMA_FORZADA = True   # Siempre activa
ALARMA_FORZADA = False  # Siempre inactiva
ALARMA_FORZADA = None   # Usar horario
```

### Características Técnicas

#### Optimizaciones de Rendimiento

1. **Procesamiento condicional**:
   - YOLOv8 solo se ejecuta cuando hay movimiento
   - Intervalo ajustable (`YOLO_INTERVAL`): GPU=1 frame, CPU=5 frames

2. **GPU automática**:
   - Detecta NVIDIA CUDA automáticamente
   - Fallback a CPU si no hay GPU
   - ONNX Runtime optimizado para inferencia

3. **Streaming optimizado**:
   - Buffer de 1 frame para streams en vivo
   - Descarta frames antiguos (`grab()` x2)
   - Reduce latencia en streams RTSP/HTTP

4. **Multithreading**:
   - Alertas enviadas en threads separados
   - Bot Telegram en thread daemon
   - No bloquea el procesamiento de video

#### Sistema de Horarios

```python
# Ejemplo: Alertas solo de lunes a viernes, 8:00 a 22:00
HORARIO_INICIO = "08:00"
HORARIO_FIN = "22:00"
DIAS_ACTIVOS = "1,2,3,4,5"  # 1=Lun, 7=Dom

# Horario nocturno (ej: 22:00 a 06:00)
HORARIO_INICIO = "22:00"
HORARIO_FIN = "06:00"
```

**Lógica de horario**:
- Verifica día de la semana (`isoweekday()`)
- Maneja horarios nocturnos (cruzan medianoche)
- Modo forzado anula horario programado

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
├── main.py                          # Programa principal
├── chat_id.py                       # Utilidad para obtener Chat ID Telegram
├── install.sh                       # Instalador Linux/macOS
├── install.bat                      # Instalador Windows
├── requirements.txt                 # Dependencias Python
├── env.example                      # Plantilla de configuración
├── .env                             # Tu configuración (crear manualmente)
├── README.md                        # Este archivo
└── yolo_model/                      # Modelo YOLOv8 (creado automáticamente)
    ├── yolov8n.onnx                 # Modelo convertido a ONNX
    └── coco.names                   # Nombres de las 80 clases COCO
```

## Uso

### Configurar fuente de video

Edita `main.py` y cambia la variable `URL`:

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
python main.py
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

**Bot de Telegram no responde**: Verifica que el `TELEGRAM_TOKEN` y `TELEGRAM_CHAT_ID` sean correctos en `.env`.

**Alertas no llegan**: Verifica el horario configurado con `/horario` en Telegram.

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