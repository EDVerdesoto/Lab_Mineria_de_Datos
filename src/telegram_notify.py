import requests
import os
import io

def send_telegram_report(chat_id: str, summary_text: str, file_content: str = None, filename: str = "report.md"):
    token = os.environ.get('TELEGRAM_TOKEN')
    
    if not token or not chat_id:
        print("Error: Credenciales de Telegram faltantes.")
        return

    # 1. Enviar el Resumen (Mensaje de texto)
    url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
    payload_msg = {
        "chat_id": chat_id,
        "text": summary_text,
        "parse_mode": "Markdown"
    }
    
    try:
        requests.post(url_msg, json=payload_msg)
    except Exception as e:
        print(f"Error enviando resumen: {e}")

    # 2. Si hay contenido detallado, enviar como archivo adjunto
    if file_content:
        url_doc = f"https://api.telegram.org/bot{token}/sendDocument"
        
        # Convertimos el string a un archivo en memoria (bytes)
        # Esto evita tener que guardar un archivo físico en el servidor
        file_buffer = io.BytesIO(file_content.encode('utf-8'))
        file_buffer.name = filename

        files = {
            'document': (filename, file_buffer)
        }
        data = {
            'chat_id': chat_id,
            'caption': "📄 *Reporte detallado adjunto*",
            'parse_mode': 'Markdown'
        }

        try:
            r = requests.post(url_doc, data=data, files=files)
            if r.status_code != 200:
                print(f"Error subiendo archivo: {r.text}")
        except Exception as e:
            print(f"Error enviando archivo: {e}")