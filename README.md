![1766033617076](image/README/1766033617076.png)# Lab Minería de Datos - Sistema de Detección de Vulnerabilidades

## Descripción
Sistema de análisis de código estático para detección de vulnerabilidades de seguridad utilizando **CodeBERT** fine-tuned en el dataset CVEfixes. El modelo clasifica fragmentos de código en diferentes categorías CWE (Common Weakness Enumeration) o como código seguro.

## 📋 Requisitos

- Python 3.10+
- Docker y Docker Compose (para despliegue)
- CUDA 11.3+ (para entrenamiento con GPU)
- 16GB+ RAM (32GB recomendado para entrenamiento)
- 8GB+ VRAM (para entrenamiento)

## 🚀 Despliegue del API

### Opción 1: Docker (Recomendado)

1. **Navegar al directorio del servicio:**
```bash
cd src/
```

2. **Configurar variables de entorno (Opcional):**
Editar `docker-compose.yml` para configurar las credenciales de Telegram:
```yaml
environment:
  - TELEGRAM_TOKEN=tu_token_aqui
  - TELEGRAM_CHAT_ID=tu_chat_id_aqui
```

3. **Construir y levantar el contenedor:**
```bash
docker-compose up --build -d
```

4. **Verificar que el servicio está corriendo:**
```bash
curl http://localhost:8000/
```

Respuesta esperada:
```json
{
  "status": "online",
  "mode": "function-level-analysis + CodeBERT",
  "model_loaded": true
}
```

### Opción 2: Ejecución Local

1. **Instalar dependencias:**
```bash
cd src/
pip install -r requirements.txt
```

2. **Iniciar el servidor:**
```bash
python analysis_api.py
```

El servidor estará disponible en `http://localhost:8000`

## 📡 Uso del API

### Endpoint: `POST /analyze`

Analiza un lote de archivos de código y retorna vulnerabilidades detectadas.

**Request Body:**
```json
{
  "pr_title": "Feature: Nueva funcionalidad de login",
  "telegram_chat_id": "7690881680",
  "use_codebert": true,
  "files": [
    {
      "filename": "auth.py",
      "programming_language": "python",
      "code": "def login(user, password):\n    query = f\"SELECT * FROM users WHERE name='{user}' AND pass='{password}'\"\n    return db.execute(query)"
    }
  ]
}
```

**Response:**
```json
{
  "total_files_processed": 1,
  "results": [
    {
      "filename": "auth.py",
      "total_functions": 1,
      "functions": [
        {
          "function_name": "login",
          "start_line": 1,
          "end_line": 3,
          "risk_score": 85.3,
          "codebert_prediction": {
            "label": "CWE-89",
            "confidence": 0.92,
            "probabilities": {
              "CWE-89": 0.92,
              "CWE-79": 0.05,
              "Safe": 0.03
            }
          },
          "features": {
            "complexity": 3,
            "has_sql": true,
            "has_string_format": true
          },
          "findings": [
            {
              "type": "SQL_INJECTION",
              "severity": "HIGH",
              "message": "Possible SQL injection via string formatting"
            }
          ],
          "tags": ["sql", "injection", "high-risk"]
        }
      ]
    }
  ]
}
```

### Endpoint: `POST /predict`

Predice vulnerabilidad en un fragmento de código específico.

**Request:**
```json
{
  "code": "eval(user_input)"
}
```

**Response:**
```json
{
  "label": "CWE-94",
  "confidence": 0.87,
  "probabilities": {
    "CWE-94": 0.87,
    "CWE-95": 0.08,
    "Safe": 0.05
  }
}
```

### Endpoint: `POST /admin/reload-model`

Recarga el modelo sin reiniciar el servidor (útil después de reentrenar).

```bash
curl -X POST http://localhost:8000/admin/reload-model
```

## 🧪 Testing

Pruebas básicas con el script incluido:

```bash
python test_api.py
```

## 📊 Entrenamiento del Modelo

### 1. Preprocesamiento (Cacheo de Datos)

```bash
cd notebooks/
python pre_process.py
```

Esto genera dos directorios:
- `train_cache/`: Dataset de entrenamiento tokenizado
- `val_cache/`: Dataset de validación tokenizado

### 2. Fine-tuning de CodeBERT

Ejecutar el notebook `04_codebert_ft.ipynb` que:
- Carga datos desde cache (optimizado para RAM)
- Entrena con Focal Loss y WeightedSampler
- Aplica Sliding Window para código largo
- Usa Mixed Precision Training (FP16)
- Guarda el mejor modelo en `models/codebert_vuln/best_model.bin`

### 3. Recargar Modelo en API

Después del entrenamiento:
```bash
curl -X POST http://localhost:8000/admin/reload-model
```

## 📁 Estructura del Proyecto

```
├── data/
│   ├── processed/          # Datasets procesados
│   └── temp_validation/    # Datos de validación temporal
├── docs/
│   └── model_selection.md  # Justificación del modelo
├── models/
│   └── codebert_vuln/      # Modelos entrenados
├── notebooks/
│   ├── 00_guide.ipynb      # Guía de uso
│   ├── 02_exploratory_analysis.ipynb
│   ├── 04_codebert_ft.ipynb  # Entrenamiento principal
│   └── pre_process.py      # Script de preprocesamiento
└── src/
    ├── analysis_api.py     # API principal
    ├── security_service.py # Análisis de características
    ├── telegram_notify.py  # Notificaciones
    ├── Dockerfile
    ├── docker-compose.yml
    └── v3/
        ├── model.py        # Arquitectura CodeBERT
        ├── dataset.py      # Dataset con sliding window
        ├── losses.py       # Focal Loss
        └── predictor.py    # Inferencia
```

## 🔧 Configuración Avanzada

### Memoria Limitada

Si tienes problemas de memoria durante el entrenamiento, ajusta en `04_codebert_ft.ipynb`:

```python
BATCH_SIZE = 4              # Reducir de 8 a 4
ACCUMULATION_STEPS = 8      # Aumentar para mantener batch efectivo
MAX_WINDOWS = 4             # Reducir ventanas por muestra
```

### Personalizar Notificaciones Telegram

Editar `src/telegram_notify.py` para cambiar el formato de los reportes.

## 📖 Documentación Adicional

- **Análisis Exploratorio:** `notebooks/02_exploratory_analysis.ipynb`
- **Justificación del Modelo:** `docs/model_selection.md`
- **Guía Completa:** `notebooks/00_guide.ipynb`

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
- Reducir `BATCH_SIZE` y `MAX_WINDOWS`
- Habilitar `GRADIENT_CHECKPOINTING = True`

### Error: "Model not found"
- Verificar que existe `src/models/codebert_vuln/best_model.bin`
- Entrenar el modelo con el notebook `04_codebert_ft.ipynb`

### API no responde
- Verificar logs: `docker-compose logs -f`
- Revisar puerto 8000 disponible: `netstat -ano | findstr :8000`

## 📝 Licencia

Proyecto académico - Lab Minería de Datos