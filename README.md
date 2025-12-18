# 🛡️ Synckuro POS - Secure CI/CD Pipeline con IA

> **Proyecto Integrador II - Desarrollo de Software Seguro** > Universidad de las Fuerzas Armadas ESPE

Este proyecto implementa una estrategia de **DevSecOps** y **Shift-Left Security** para la aplicación *Synckuro POS*. Se utiliza un pipeline de CI/CD automatizado que integra un modelo de Inteligencia Artificial (CodeBERT) capaz de detectar vulnerabilidades en el código fuente antes de que llegue a producción.

---

## 🚀 Enlaces en Producción

| Componente | Estado | Enlace |
|------------|--------|--------|
| **Aplicación Web** | 🟢 Online | [🔗 Abrir Synckuro POS](https://synckuropos.onrender.com/) |
| **Bot de Alertas** | 🤖 Activo | [🔗 @SwSeguro_bot](https://t.me/SwSeguro_bot) |

---

## 🧠 Entrenamiento del Modelo de IA

El núcleo de seguridad es un modelo **CodeBERT (Microsoft)** sometido a *fine-tuning* para clasificación de vulnerabilidades.

* **Arquitectura:** Transformer (BERT-based) pre-entrenado para lenguajes de programación.
* **Dataset:** Entrenado con un conjunto de datos masivo (~20GB) de funciones C/C++/Python etiquetadas como seguras o vulnerables (CWE-89, CWE-79, etc.).
* **Notebook de Entrenamiento:** El código fuente del entrenamiento y la validación del modelo se encuentra disponible en este repositorio:
    * 📄 [**Ver Notebook de Entrenamiento (04_codebert_ft.ipynb)**](./notebooks/04_codebert_ft.ipynb)
* **Guía de uso:** Se encuentra una guía de uso para el modelo en: 
    * 📄 [**Ver Notebook de guía (00_guide.ipynb)**](./notebooks/00_guide.ipynb)
> **Nota:** El modelo entrenado se despliega como un microservicio (API FastAPI) independiente para optimizar los recursos del pipeline.

---

## ⚙️ Instrucciones de Setup del Pipeline

Para replicar este pipeline en otro repositorio, se deben configurar los siguientes **GitHub Secrets** en la ruta `Settings > Secrets and variables > Actions`:

### 1. Variables de Entorno Requeridas

| Nombre del Secreto | Descripción |
|--------------------|-------------|
| `TELEGRAM_TOKEN` | Token de acceso del BotFather para el bot de notificaciones. |
| `TELEGRAM_CHAT_ID` | ID numérico del chat (grupo o usuario) donde llegarán las alertas. |
| `RENDER_DEPLOY_HOOK` | URL del Webhook de Render para disparar el despliegue automático del Frontend. |

### 2. Flujo de Trabajo (Workflow)

El pipeline está definido en `.github/workflows/pipeline_seguro.yml` y consta de tres etapas:

1.  **Security Gate (IA):** Se ejecuta al hacer Pull Request hacia la rama `test`. Envía los archivos modificados a la API de IA. Si detecta vulnerabilidades, bloquea el merge.
2.  **Testing:** Si el código es seguro, se ejecutan las pruebas unitarias (Jest/Pytest).
3.  **Deploy:** Al hacer merge a `main`, se despliega automáticamente en Render.

---

## 🤖 Evidencias del Bot de Telegram

El sistema notifica en tiempo real sobre el estado del análisis, fallos de seguridad y despliegues exitosos.

### Notificación de Bloqueo por Vulnerabilidad
*(El modelo detecta código inseguro y rechaza el PR)*

![Captura de Alerta de Vulnerabilidad](./img/captura_bot_fallo.png)
### Notificación de Despliegue Exitoso
*(El código pasa todas las pruebas y se actualiza la web)*

![Captura de Exito](./img/captura_bot_exito.png)
---

## 📸 Capturas de la Aplicación

**Vista Principal (Deploy en Render)**

![Captura Synckuro POS](./img/synckuro_deploy.png)
---

## 👥 Autores
* **Edison Verdesoto**
* **Joan [Apellido]**
* **Rubén [Apellido]**

---
*Generado para la asignatura de Desarrollo de Software Seguro - 2025*