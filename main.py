import cv2
import numpy as np
import os

# =====================================================
# CONFIGURACIÓN DE YOLOv8 ONNX
# =====================================================
YOLO_MODEL = "yolo_model/yolov8n.onnx"
YOLO_NAMES = "yolo_model/coco.names"

if not os.path.exists(YOLO_MODEL):
    print("=" * 55)
    print("ERROR: No se encontró el modelo YOLOv8")
    print("Ejecuta primero: bash descargar_yolo.sh")
    print("=" * 55)
    exit()

# Cargar clases COCO
with open(YOLO_NAMES, "r") as f:
    clases = [line.strip() for line in f.readlines()]

# =====================================================
# FILTRO: Solo detectar estas clases
# =====================================================
CLASES_DETECTAR = {
    "person": {"nombre": "PERSONA", "color": (0, 255, 0)},  # Verde
    "dog": {"nombre": "PERRO", "color": (0, 165, 255)},  # Naranja
    "cat": {"nombre": "GATO", "color": (255, 0, 255)}  # Magenta
}

# IDs de las clases en COCO (person=0, cat=15, dog=16)
IDS_DETECTAR = {0: "person", 15: "cat", 16: "dog"}

# =====================================================
# CARGAR YOLOv8 CON ONNX RUNTIME (GPU/CPU automático)
# =====================================================
try:
    import onnxruntime as ort

    print("ONNX Runtime encontrado")
except ImportError:
    print("=" * 55)
    print("ERROR: ONNX Runtime no instalado")
    print("Instala con: pip install onnxruntime-gpu")
    print("=" * 55)
    exit()

# Verificar providers disponibles
providers = ort.get_available_providers()
print(f"Providers disponibles: {providers}")

# Intentar usar CUDA, si no usar CPU
USE_GPU = False
if 'CUDAExecutionProvider' in providers:
    try:
        session = ort.InferenceSession(
            YOLO_MODEL,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        USE_GPU = True
        print("✓ GPU CUDA activada")
    except Exception as e:
        print(f"✗ Error con CUDA: {e}")
        session = ort.InferenceSession(YOLO_MODEL, providers=['CPUExecutionProvider'])
        print("Usando CPU como fallback")
else:
    session = ort.InferenceSession(YOLO_MODEL, providers=['CPUExecutionProvider'])
    print("Usando CPU (instala onnxruntime-gpu para usar GPU)")

# Obtener info del modelo
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Modelo cargado: {YOLO_MODEL}")
print(f"Input: {input_name} {input_shape}")

INPUT_SIZE = 640

# =====================================================
# CONFIGURACIÓN DE VIDEO
# =====================================================
URL = "IntrusoCasa.mp4"
# URL = "https://plataforma.caceres.es/streaming/ayuntamiento"

ES_STREAMING = URL.startswith("http")

cap = cv2.VideoCapture(URL)

if ES_STREAMING:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: No se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
delay = int(1000 / fps)

print(f"Video cargado - FPS: {fps:.1f}")

# Detector de movimiento
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# =====================================================
# PARÁMETROS AJUSTABLES
# =====================================================
AREA_MINIMA = 500
conf_threshold = 0.25  # 75% sensibilidad
NMS_THRESHOLD = 0.45
# Con GPU podemos procesar cada frame, con CPU cada 5
YOLO_INTERVAL = 1 if USE_GPU else 5

print("=" * 55)
print(f"DETECTOR YOLOv8n [{'GPU' if USE_GPU else 'CPU'}]")
print("=" * 55)
print("Controles:")
print("  ESC = Salir | P = Pausar | R = Reiniciar")
print("  S/W = Sensibilidad YOLO | A/D = Sensibilidad Mov")
print("  Q/E = Frecuencia YOLO")
print("=" * 55)


def detectar_yolov8(frame, conf_thresh):
    """Procesa un frame con YOLOv8 ONNX Runtime"""
    height, width = frame.shape[:2]

    # Preparar imagen (letterbox a 640x640)
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    # Crear imagen con padding
    input_img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    input_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Preprocesar para ONNX Runtime
    input_img = input_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
    input_img = np.expand_dims(input_img, axis=0)  # Añadir batch dimension

    # Inferencia
    outputs = session.run(None, {input_name: input_img})[0]

    # YOLOv8 output: [1, 84, 8400] -> [8400, 84]
    outputs = outputs[0].T

    boxes = []
    confidences = []
    class_ids = []

    for detection in outputs:
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_thresh and class_id in IDS_DETECTAR:
            cx, cy, w, h = detection[:4]

            # Quitar padding y escalar
            cx = (cx - pad_x) / scale
            cy = (cy - pad_y) / scale
            w = w / scale
            h = h / scale

            x = int(cx - w / 2)
            y = int(cy - h / 2)
            w = int(w)
            h = int(h)

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)

            if w > 0 and h > 0:
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS
    detecciones = []
    objetos = {}

    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, NMS_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                conf = confidences[i]

                nombre_en = IDS_DETECTAR[class_id]
                info = CLASES_DETECTAR[nombre_en]
                nombre = info["nombre"]
                color = info["color"]

                detecciones.append((x, y, w, h, nombre, conf, color))
                objetos[nombre] = objetos.get(nombre, 0) + 1

    return detecciones, objetos


# =====================================================
# LOOP PRINCIPAL
# =====================================================
frame_count = 0
ultimo_detecciones = []
ultimo_objetos = {}
paused = False

# Para medir FPS de inferencia
import time

inference_times = []

while True:
    if not paused:
        if ES_STREAMING:
            for _ in range(2):
                cap.grab()

        ret, frame = cap.read()
        if not ret:
            if ES_STREAMING:
                cap.release()
                cap = cv2.VideoCapture(URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        frame_count += 1
        height, width = frame.shape[:2]

        # =====================================================
        # DETECCIÓN DE MOVIMIENTO
        # =====================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movimiento_detectado = any(cv2.contourArea(c) > AREA_MINIMA for c in cnts)

        # =====================================================
        # DETECCIÓN YOLOv8
        # =====================================================
        if frame_count % YOLO_INTERVAL == 0:
            t1 = time.perf_counter()
            ultimo_detecciones, ultimo_objetos = detectar_yolov8(frame, conf_threshold)
            t2 = time.perf_counter()

            inference_times.append((t2 - t1) * 1000)
            if len(inference_times) > 30:
                inference_times.pop(0)

        # =====================================================
        # DIBUJAR DETECCIONES
        # =====================================================
        for (x, y, w, h, nombre, conf, color) in ultimo_detecciones:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            label = f"{nombre}: {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # =====================================================
        # PANEL INFO
        # =====================================================
        cv2.rectangle(frame, (0, 0), (width, 90), (30, 30, 30), -1)

        if movimiento_detectado:
            cv2.putText(frame, "! MOVIMIENTO DETECTADO", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Sin movimiento", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        x_pos = 10
        y_pos = 50
        if ultimo_objetos:
            for nombre, cantidad in ultimo_objetos.items():
                for k, v in CLASES_DETECTAR.items():
                    if v["nombre"] == nombre:
                        color = v["color"]
                        break
                texto = f"{nombre}: {cantidad}"
                cv2.putText(frame, texto, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                x_pos += len(texto) * 12 + 20
        else:
            cv2.putText(frame, "Buscando...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Configuración y rendimiento
        avg_ms = np.mean(inference_times) if inference_times else 0
        backend = "GPU" if USE_GPU else "CPU"
        config_text = f"YOLOv8n [{backend}] | {avg_ms:.1f}ms | Sens: {int((1 - conf_threshold) * 100)}% | Int: {YOLO_INTERVAL}f"
        cv2.putText(frame, config_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Detector YOLOv8n", frame)

    # =====================================================
    # CONTROLES
    # =====================================================
    k = cv2.waitKey(delay if not paused else 100) & 0xFF

    if k == 27:
        break
    elif k == ord('p') or k == ord('P'):
        paused = not paused
        print("PAUSADO" if paused else "REANUDADO")
    elif k == ord('r') or k == ord('R'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        ultimo_detecciones = []
        ultimo_objetos = {}
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        print("VIDEO REINICIADO")
    elif k == ord('s') or k == ord('S'):
        conf_threshold = max(0.1, conf_threshold - 0.05)
        print(f"Sensibilidad YOLO: {int((1 - conf_threshold) * 100)}%")
    elif k == ord('w') or k == ord('W'):
        conf_threshold = min(0.9, conf_threshold + 0.05)
        print(f"Sensibilidad YOLO: {int((1 - conf_threshold) * 100)}%")
    elif k == ord('a') or k == ord('A'):
        AREA_MINIMA = max(100, AREA_MINIMA - 100)
        print(f"Sensibilidad movimiento: {AREA_MINIMA}px")
    elif k == ord('d') or k == ord('D'):
        AREA_MINIMA += 100
        print(f"Sensibilidad movimiento: {AREA_MINIMA}px")
    elif k == ord('q') or k == ord('Q'):
        YOLO_INTERVAL = max(1, YOLO_INTERVAL - 1)
        print(f"YOLO cada {YOLO_INTERVAL} frames")
    elif k == ord('e') or k == ord('E'):
        YOLO_INTERVAL += 1
        print(f"YOLO cada {YOLO_INTERVAL} frames")

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado")