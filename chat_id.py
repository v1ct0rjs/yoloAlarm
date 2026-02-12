#!/usr/bin/env python3
"""
Script para obtener tu Chat ID de Telegram
Ejecuta después de crear tu bot y enviarle /start
"""

import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    print("=" * 55)
    print("ERROR: Falta TELEGRAM_TOKEN en el archivo .env")
    print("")
    print("Pasos para crear tu bot:")
    print("1. Abre Telegram y busca @BotFather")
    print("2. Envía /newbot")
    print("3. Sigue las instrucciones (nombre y username)")
    print("4. Copia el token y pégalo en .env")
    print("5. Ejecuta este script de nuevo")
    print("=" * 55)
    exit(1)

try:
    import requests
except ImportError:
    print("Instalando requests...")
    import subprocess
    subprocess.run(["pip", "install", "requests", "-q"])
    import requests

print("Obteniendo actualizaciones del bot...")
print("(Asegúrate de haber enviado /start a tu bot)")
print("")

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
response = requests.get(url)
data = response.json()

if not data.get("ok"):
    print(f"Error: {data}")
    exit(1)

if not data.get("result"):
    print("=" * 55)
    print("No se encontraron mensajes.")
    print("")
    print("Asegúrate de:")
    print("1. Abrir Telegram")
    print("2. Buscar tu bot por su username")
    print("3. Enviar /start al bot")
    print("4. Ejecutar este script de nuevo")
    print("=" * 55)
    exit(1)

print("=" * 55)
print("USUARIOS QUE HAN HABLADO CON TU BOT:")
print("=" * 55)

chat_ids = set()
for update in data["result"]:
    if "message" in update:
        chat = update["message"]["chat"]
        chat_id = chat["id"]
        if chat_id not in chat_ids:
            chat_ids.add(chat_id)
            nombre = chat.get("first_name", "") + " " + chat.get("last_name", "")
            username = chat.get("username", "sin username")
            print(f"  Nombre: {nombre.strip()}")
            print(f"  Username: @{username}")
            print(f"  Chat ID: {chat_id}")
            print("-" * 55)

print("")
print("Copia el CHAT_ID y pégalo en tu archivo .env:")
print(f"TELEGRAM_CHAT_ID={list(chat_ids)[0] if chat_ids else 'TU_CHAT_ID'}")
print("=" * 55)