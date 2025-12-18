import sys
import requests
import os

def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("Error: Faltan las variables de entorno TELEGRAM_TOKEN o TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Mensaje enviado a Telegram")
        else:
            print(f"Error enviando mensaje: {response.text}")
    except Exception as e:
        print(f"Excepción al enviar a Telegram: {e}")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = sys.argv[1]
        send_telegram_message(msg)
    else:
        print("Uso: python telegram_notify.py 'Mensaje entre comillas'")