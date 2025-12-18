# Usamos una imagen base ligera de Python 3.10
FROM python:3.10-slim

# Variables de entorno para optimizar Python en Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Evita que pip avise de actualizaciones
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Establecemos el directorio de trabajo
WORKDIR /app

# Instalamos dependencias del sistema necesarias
# (curl es útil para healthchecks, build-essential a veces se requiere para compilar deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copiamos PRIMERO el requirements.txt (para aprovechar la caché de Docker)
COPY requirements.txt .

# 2. Instalamos las dependencias
# El flag --no-cache-dir es vital para no guardar los whl descargados (ahorra ~500MB con PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copiamos el resto del código
COPY . .

# Creamos un usuario no-root por seguridad
RUN adduser --disabled-password --gecos "" appuser
# Cambiamos el dueño de la carpeta /app
RUN chown -R appuser:appuser /app

# Cambiamos al usuario limitado
USER appuser

# Exponemos el puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "analysis_api:app", "--host", "0.0.0.0", "--port", "8000"]