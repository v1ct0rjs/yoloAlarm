import cv2
import numpy as np
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
import threading
from dotenv import load_dotenv
import requests
import io

# Cargar variables de entorno
load_dotenv()

# =====================================================
# CONFIGURACIÓN DE ALERTAS POR EMAIL
# =====================================================
ALERTA_EMAIL_ACTIVADA = True

# Configuración de Gmail (desde archivo .env)
GMAIL_CUENTA = os.getenv("GMAIL_CUENTA")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")

# Verificar configuración email
if ALERTA_EMAIL_ACTIVADA and (not GMAIL_CUENTA or not GMAIL_PASSWORD):
    print(" Email no configurado - alertas por email desactivadas")
    ALERTA_EMAIL_ACTIVADA = False

# =====================================================
# CONFIGURACIÓN DE TELEGRAM
# =====================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERTA_TELEGRAM_ACTIVADA = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

if not ALERTA_TELEGRAM_ACTIVADA:
    print(" Telegram no configurado - alertas por Telegram desactivadas")
else:
    print(f"[OK] Telegram configurado (Chat ID: {TELEGRAM_CHAT_ID})")

# =====================================================
# CONFIGURACIÓN DE HORARIOS
# =====================================================
HORARIO_INICIO = os.getenv("HORARIO_INICIO", "")
HORARIO_FIN = os.getenv("HORARIO_FIN", "")
DIAS_ACTIVOS = os.getenv("DIAS_ACTIVOS", "1,2,3,4,5,6,7")

# Parsear días activos
try:
    DIAS_ACTIVOS_LIST = [int(d.strip()) for d in DIAS_ACTIVOS.split(",") if d.strip()]
except:
    DIAS_ACTIVOS_LIST = [1, 2, 3, 4, 5, 6, 7]

# Estado global de alarma (puede cambiarse por Telegram)
ALARMA_ACTIVADA = True
ALARMA_FORZADA = None  # None=usar horario, True=siempre on, False=siempre off

# Tiempo mínimo entre alertas (evita spam)
COOLDOWN_ALERTAS = 60  # segundos

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
# SISTEMA DE ALERTAS POR EMAIL
# =====================================================
ultima_alerta = None
alerta_en_proceso = False


def enviar_alerta_email(frame, objetos_detectados):
    """Envía alerta por email con imagen adjunta"""
    global alerta_en_proceso

    if alerta_en_proceso:
        return

    alerta_en_proceso = True

    try:
        # Guardar frame como imagen temporal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"/tmp/alerta_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)

        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = GMAIL_CUENTA
        msg['To'] = GMAIL_CUENTA
        msg['Subject'] = f"ALERTA: Intruso detectado - {datetime.now().strftime('%H:%M:%S')}"

        # Cuerpo del mensaje
        cuerpo = f"""
        ALERTA DE SEGURIDAD 

        Se ha detectado movimiento sospechoso.

        Fecha: {datetime.now().strftime('%d/%m/%Y')}
        Hora: {datetime.now().strftime('%H:%M:%S')}

        👁️ Objetos detectados:
        """

        for nombre, cantidad in objetos_detectados.items():
            cuerpo += f"\n        • {nombre}: {cantidad}"

        cuerpo += """

        Se adjunta imagen de la detección.

        ---
        Sistema de Vigilancia YOLOv8
        """

        msg.attach(MIMEText(cuerpo, 'plain'))

        # Adjuntar imagen
        with open(img_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data, name=f"captura_{timestamp}.jpg")
            msg.attach(image)

        # Enviar email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(GMAIL_CUENTA, GMAIL_PASSWORD)
            server.sendmail(GMAIL_CUENTA, GMAIL_CUENTA, msg.as_string())

        print(f"Alerta enviada a {GMAIL_CUENTA}")

        # Limpiar imagen temporal
        if os.path.exists(img_path):
            os.remove(img_path)

    except Exception as e:
        print(f"[ERROR] Error enviando alerta: {e}")
    finally:
        alerta_en_proceso = False


def verificar_y_enviar_alerta(frame, objetos_detectados):
    """Verifica cooldown y envía alerta en hilo separado"""
    global ultima_alerta

    if not ALERTA_EMAIL_ACTIVADA and not ALERTA_TELEGRAM_ACTIVADA:
        return

    # Verificar si la alarma está activa
    if not verificar_alarma_activa():
        return

    # Verificar si hay personas detectadas
    if "PERSONA" not in objetos_detectados:
        return

    # Verificar cooldown
    ahora = datetime.now()
    if ultima_alerta is not None:
        tiempo_desde_ultima = (ahora - ultima_alerta).total_seconds()
        if tiempo_desde_ultima < COOLDOWN_ALERTAS:
            return

    # Actualizar tiempo de última alerta
    ultima_alerta = ahora

    # Enviar en hilo separado para no bloquear el video
    thread = threading.Thread(
        target=enviar_alertas,
        args=(frame.copy(), objetos_detectados.copy())
    )
    thread.daemon = True
    thread.start()


def enviar_alertas(frame, objetos_detectados):
    """Envía alertas por todos los canales configurados"""
    if ALERTA_EMAIL_ACTIVADA:
        enviar_alerta_email(frame, objetos_detectados)
    if ALERTA_TELEGRAM_ACTIVADA:
        enviar_alerta_telegram(frame, objetos_detectados)


# =====================================================
# FUNCIONES DE TELEGRAM
# =====================================================

def verificar_alarma_activa():
    """Verifica si la alarma debe estar activa según horario y estado"""
    global ALARMA_FORZADA

    # Si está forzada, usar ese valor
    if ALARMA_FORZADA is not None:
        return ALARMA_FORZADA

    ahora = datetime.now()

    # Verificar día de la semana (1=Lunes, 7=Domingo)
    dia_semana = ahora.isoweekday()
    if dia_semana not in DIAS_ACTIVOS_LIST:
        return False

    # Si no hay horario configurado, siempre activo
    if not HORARIO_INICIO or not HORARIO_FIN:
        return True

    # Verificar hora
    try:
        hora_inicio = datetime.strptime(HORARIO_INICIO, "%H:%M").time()
        hora_fin = datetime.strptime(HORARIO_FIN, "%H:%M").time()
        hora_actual = ahora.time()

        if hora_inicio <= hora_fin:
            return hora_inicio <= hora_actual <= hora_fin
        else:
            # Horario nocturno (ej: 22:00 a 06:00)
            return hora_actual >= hora_inicio or hora_actual <= hora_fin
    except:
        return True


def enviar_alerta_telegram(frame, objetos_detectados):
    """Envía alerta con foto por Telegram"""
    try:
        # Convertir frame a bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        img_bytes.name = 'alerta.jpg'

        # Crear mensaje
        mensaje = "*ALERTA: Intruso detectado*\n\n"
        mensaje += f"Fecha: {datetime.now().strftime('%d/%m/%Y')}\n"
        mensaje += f"Hora: {datetime.now().strftime('%H:%M:%S')}\n\n"
        mensaje += "👁️ *Detectado:*\n"

        for nombre, cantidad in objetos_detectados.items():
            mensaje += f"  • {nombre}: {cantidad}\n"

        # Enviar foto con caption
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        response = requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': mensaje,
            'parse_mode': 'Markdown'
        }, files={
            'photo': img_bytes
        })

        if response.status_code == 200:
            print("Alerta enviada por Telegram")
        else:
            print(f"[ERROR] Error Telegram: {response.text}")

    except Exception as e:
        print(f"[ERROR] Error enviando Telegram: {e}")


def enviar_mensaje_telegram(mensaje):
    """Envía un mensaje de texto por Telegram"""
    if not ALERTA_TELEGRAM_ACTIVADA:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        response = requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': mensaje,
            'parse_mode': 'Markdown'
        })
        return response.status_code == 200
    except:
        return False


def iniciar_bot_telegram():
    """Inicia el bot de Telegram en segundo plano para recibir comandos"""
    if not ALERTA_TELEGRAM_ACTIVADA:
        return

    def escuchar_comandos():
        global ALARMA_FORZADA, COOLDOWN_ALERTAS, HORARIO_INICIO, HORARIO_FIN, DIAS_ACTIVOS_LIST, SOLICITAR_FOTO

        last_update_id = 0
        url_base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

        # Enviar mensaje de inicio
        enviar_mensaje_telegram(
            "*Sistema de vigilancia iniciado*\n\nComandos disponibles:\n/activar - Activar alarma\n/desactivar - Desactivar alarma\n/estado - Ver estado\n/foto - Captura actual\n/horario - Ver horario")

        while True:
            try:
                response = requests.get(
                    f"{url_base}/getUpdates",
                    params={'offset': last_update_id + 1, 'timeout': 30},
                    timeout=35
                )

                if response.status_code != 200:
                    continue

                updates = response.json().get('result', [])

                for update in updates:
                    last_update_id = update['update_id']

                    if 'message' not in update:
                        continue

                    message = update['message']
                    chat_id = str(message['chat']['id'])

                    # Solo responder al chat autorizado
                    if chat_id != TELEGRAM_CHAT_ID:
                        continue

                    text = message.get('text', '').lower().strip()

                    if text == '/activar':
                        ALARMA_FORZADA = True
                        enviar_mensaje_telegram("*Alarma ACTIVADA*\n\nRecibirás alertas cuando se detecte una persona.")

                    elif text == '/desactivar':
                        ALARMA_FORZADA = False
                        enviar_mensaje_telegram("*Alarma DESACTIVADA*\n\nNo recibirás alertas hasta que la actives.")

                    elif text == '/auto':
                        ALARMA_FORZADA = None
                        enviar_mensaje_telegram("*Modo AUTOMÁTICO*\n\nLa alarma seguirá el horario configurado.")

                    elif text == '/estado':
                        if ALARMA_FORZADA is True:
                            estado = "ACTIVADA (forzada)"
                        elif ALARMA_FORZADA is False:
                            estado = "DESACTIVADA (forzada)"
                        else:
                            activa = verificar_alarma_activa()
                            estado = f"AUTO ({'activa' if activa else 'inactiva'} ahora)"

                        msg = f"*Estado del sistema*\n\n"
                        msg += f"Alarma: {estado}\n"
                        msg += f"Cooldown: {COOLDOWN_ALERTAS}s\n"
                        if HORARIO_INICIO and HORARIO_FIN:
                            msg += f"Horario: {HORARIO_INICIO} - {HORARIO_FIN}\n"
                        msg += f"Días: {DIAS_ACTIVOS}"
                        enviar_mensaje_telegram(msg)

                    elif text == '/horario':
                        nombres_dias = {1: "Lun", 2: "Mar", 3: "Mie", 4: "Jue", 5: "Vie", 6: "Sab", 7: "Dom"}
                        dias_texto = ", ".join([nombres_dias[d] for d in sorted(DIAS_ACTIVOS_LIST)])
                        if HORARIO_INICIO and HORARIO_FIN:
                            msg = f"*Horario configurado*\n\n"
                            msg += f"Inicio: {HORARIO_INICIO}\n"
                            msg += f"Fin: {HORARIO_FIN}\n"
                            msg += f"Dias: {dias_texto}\n\n"
                            msg += "Comandos:\n/sethorario HH:MM HH:MM\n/setdias 1,2,3,4,5"
                        else:
                            msg = "*Sin horario configurado*\n\nAlertas activas 24/7\n\n"
                            msg += f"Dias: {dias_texto}\n\n"
                            msg += "Comandos:\n/sethorario HH:MM HH:MM\n/setdias 1,2,3,4,5"
                        enviar_mensaje_telegram(msg)

                    elif text == '/foto':
                        enviar_mensaje_telegram("Capturando...")
                        # La foto se enviará desde el loop principal
                        global SOLICITAR_FOTO
                        SOLICITAR_FOTO = True

                    elif text == '/start' or text == '/help':
                        msg = "*Sistema de Vigilancia YOLOv8*\n\n"
                        msg += "*Comandos:*\n"
                        msg += "/activar - Activar alarma\n"
                        msg += "/desactivar - Desactivar alarma\n"
                        msg += "/auto - Modo automatico (horario)\n"
                        msg += "/estado - Ver estado actual\n"
                        msg += "/horario - Ver horario configurado\n"
                        msg += "/sethorario HH:MM HH:MM - Cambiar horario\n"
                        msg += "/setdias 1,2,3,4,5 - Cambiar dias\n"
                        msg += "/foto - Obtener captura actual"
                        enviar_mensaje_telegram(msg)

                    elif text.startswith('/sethorario'):
                        partes = text.split()
                        if len(partes) == 3:
                            try:
                                # Validar formato
                                datetime.strptime(partes[1], "%H:%M")
                                datetime.strptime(partes[2], "%H:%M")
                                HORARIO_INICIO = partes[1]
                                HORARIO_FIN = partes[2]
                                enviar_mensaje_telegram(
                                    f"*Horario actualizado*\n\nInicio: {HORARIO_INICIO}\nFin: {HORARIO_FIN}")
                            except ValueError:
                                enviar_mensaje_telegram(
                                    "Formato incorrecto. Usa: /sethorario HH:MM HH:MM\n\nEjemplo: /sethorario 08:00 22:00")
                        else:
                            enviar_mensaje_telegram(
                                "Uso: /sethorario HH:MM HH:MM\n\nEjemplo: /sethorario 08:00 22:00\nPara horario nocturno: /sethorario 22:00 06:00")

                    elif text.startswith('/setdias'):
                        partes = text.split()
                        if len(partes) == 2:
                            try:
                                dias = [int(d.strip()) for d in partes[1].split(",")]
                                # Validar que son dias validos (1-7)
                                if all(1 <= d <= 7 for d in dias):
                                    DIAS_ACTIVOS_LIST = dias
                                    nombres_dias = {1: "Lun", 2: "Mar", 3: "Mie", 4: "Jue", 5: "Vie", 6: "Sab",
                                                    7: "Dom"}
                                    dias_texto = ", ".join([nombres_dias[d] for d in sorted(dias)])
                                    enviar_mensaje_telegram(f"*Dias actualizados*\n\nActivos: {dias_texto}")
                                else:
                                    enviar_mensaje_telegram(
                                        "Los dias deben ser numeros del 1 al 7\n\n1=Lunes, 7=Domingo")
                            except ValueError:
                                enviar_mensaje_telegram(
                                    "Formato incorrecto. Usa numeros separados por comas.\n\nEjemplo: /setdias 1,2,3,4,5")
                        else:
                            enviar_mensaje_telegram(
                                "Uso: /setdias 1,2,3,4,5\n\n1=Lunes, 2=Martes, 3=Miercoles\n4=Jueves, 5=Viernes, 6=Sabado, 7=Domingo\n\nEjemplos:\n/setdias 1,2,3,4,5 (Lun-Vie)\n/setdias 6,7 (fines de semana)\n/setdias 1,2,3,4,5,6,7 (todos)")

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                print(f"Error bot Telegram: {e}")
                import time
                time.sleep(5)

    thread = threading.Thread(target=escuchar_comandos, daemon=True)
    thread.start()
    print("[OK] Bot de Telegram iniciado")


# Variable global para solicitar foto desde Telegram
SOLICITAR_FOTO = False


def enviar_foto_telegram(frame):
    """Envía una foto bajo demanda"""
    global SOLICITAR_FOTO

    if not SOLICITAR_FOTO:
        return

    SOLICITAR_FOTO = False

    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = io.BytesIO(img_encoded.tobytes())
        img_bytes.name = 'captura.jpg'

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        response = requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': f"📸 Captura: {datetime.now().strftime('%H:%M:%S')}"
        }, files={
            'photo': img_bytes
        })
    except Exception as e:
        print(f"Error enviando foto: {e}")


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
        print("[OK] GPU CUDA activada")
    except Exception as e:
        print(f"[X] Error con CUDA: {e}")
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
#URL = "IntrusoCasa.mp4"
URL = "https://plataforma.caceres.es/streaming/ayuntamiento"

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
print("  Q/E = Frecuencia YOLO | M = Activar/Desactivar alertas")
print("=" * 55)

# Iniciar bot de Telegram
iniciar_bot_telegram()


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

            # Verificar si hay que enviar alerta
            if ultimo_objetos:
                verificar_y_enviar_alerta(frame, ultimo_objetos)

        # =====================================================
        # DIBUJAR DETECCIONES
        # =====================================================
        for (x, y, w, h, nombre, conf, color) in ultimo_detecciones:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            label = f"{nombre}: {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Enviar foto si se solicitó desde Telegram
        enviar_foto_telegram(frame)

        # =====================================================
        # PANEL INFO
        # =====================================================
        cv2.rectangle(frame, (0, 0), (width, 110), (30, 30, 30), -1)

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

        # Estado de alertas
        alarma_activa = verificar_alarma_activa()
        if ALERTA_TELEGRAM_ACTIVADA or ALERTA_EMAIL_ACTIVADA:
            if ALARMA_FORZADA is True:
                alerta_text = "Alarma: ON (forzada)"
                color_alerta = (0, 255, 0)
            elif ALARMA_FORZADA is False:
                alerta_text = "Alarma: OFF (forzada)"
                color_alerta = (0, 0, 255)
            elif alarma_activa:
                if ultima_alerta is not None:
                    tiempo_desde = (datetime.now() - ultima_alerta).total_seconds()
                    cooldown_restante = max(0, COOLDOWN_ALERTAS - tiempo_desde)
                    if cooldown_restante > 0:
                        alerta_text = f"Alarma: ON (espera: {int(cooldown_restante)}s)"
                        color_alerta = (0, 255, 255)
                    else:
                        alerta_text = "Alarma: ON (lista)"
                        color_alerta = (0, 255, 0)
                else:
                    alerta_text = "Alarma: ON (lista)"
                    color_alerta = (0, 255, 0)
            else:
                alerta_text = "Alarma: OFF (fuera horario)"
                color_alerta = (100, 100, 100)
            cv2.putText(frame, alerta_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_alerta, 1)
        else:
            cv2.putText(frame, "Alertas: NO CONFIGURADAS", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100),
                        1)

        # Redimensionar a la mitad para mostrar
        frame_mostrar = cv2.resize(frame, (width // 2, height // 2))
        cv2.imshow("Detector YOLOv8n", frame_mostrar)

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
    elif k == ord('m') or k == ord('M'):
        # Ciclar entre: ON -> OFF -> AUTO
        if ALARMA_FORZADA is None:
            ALARMA_FORZADA = True
            print("Alarma: ACTIVADA (forzada)")
        elif ALARMA_FORZADA is True:
            ALARMA_FORZADA = False
            print("Alarma: DESACTIVADA (forzada)")
        else:
            ALARMA_FORZADA = None
            print("Alarma: AUTOMÁTICA (según horario)")

cap.release()
cv2.destroyAllWindows()
print("Programa finalizado")