# Vulnerability Detection v1

Modelo híbrido **CodeBERT + Features manuales** para detección de vulnerabilidades en código fuente.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    CÓDIGO FUENTE                            │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    CodeBERT     │ │    Lizard       │ │    Patterns     │
│  (semántica)    │ │  (complejidad)  │ │  (regex + AST)  │
│  768 dims       │ │  5 dims         │ │  5 + 2 dims     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                      Fusion Layer
                              │
                              ▼
                      Clasificador
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Predicción          Localización
           (Safe/Vulnerable)      (líneas exactas)
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Para GPU NVIDIA (recomendado)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Dataset

El CSV debe tener las siguientes columnas:

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `code` | Código fuente | `"def login(user): ..."` |
| `programming_language` | Lenguaje | `"python"` |
| `safety` | Safe o Vulnerable | `"vulnerable"` |
| `cwe_id` | ID del CWE (solo si vulnerable) | `"CWE-89"` |

## Uso

### Entrenamiento

```bash
# Clasificación binaria (Safe vs Vulnerable)
python train.py --csv data/dataset.csv --mode binary --epochs 5

# Clasificación multi-clase (Safe + CWEs específicos + Other)
python train.py --csv data/dataset.csv --mode multiclass --epochs 10

# Con límite de muestras (para testing rápido)
python train.py --csv data/dataset.csv --max-samples 10000
```

### Inferencia

```bash
# Analizar un archivo
python predict.py --file vulnerable_code.py

# Analizar código directo
python predict.py --code "query = f'SELECT * FROM users WHERE id={user_id}'"

# Analizar directorio completo
python predict.py --dir ./src --ext py

# Generar reporte JSON
python predict.py --file code.py --format json
```

### Uso programático

```python
from inference import VulnerabilityPredictor

# Cargar modelo
predictor = VulnerabilityPredictor('checkpoints/best_model.pt')

# Analizar código
code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id={user_id}"
    return db.execute(query)
'''

result = predictor.predict(code, language='python')

print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence']:.1%}")
print(f"Líneas vulnerables: {result['vulnerable_lines']}")

# Generar reporte
report = predictor.generate_report(result)
print(report)
```

## Estructura del proyecto

```
v1/
├── config.py                 # Configuración central
├── train.py                  # Script de entrenamiento
├── predict.py                # Script de inferencia
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   └── hybrid_classifier.py  # Modelo CodeBERT + Features
│
├── extractors/
│   ├── __init__.py
│   ├── base.py               # Clase base
│   ├── complexity_extractor.py  # Métricas Lizard
│   ├── pattern_extractor.py     # Patrones regex
│   └── ast_extractor.py         # Análisis AST
│
├── training/
│   ├── __init__.py
│   ├── dataset.py            # Datasets para entrenamiento
│   └── trainer.py            # Trainer con optimizaciones
│
├── inference/
│   ├── __init__.py
│   └── predictor.py          # Predictor con localización
│
├── checkpoints/              # Modelos guardados
└── data/                     # Datasets
```

## CWEs soportados (modo multiclass)

- CWE-79: Cross-site Scripting (XSS)
- CWE-89: SQL Injection
- CWE-119: Buffer Overflow
- CWE-125: Out-of-bounds Read
- CWE-200: Information Exposure
- CWE-264: Permissions Issues
- CWE-287: Authentication Issues
- CWE-352: CSRF
- CWE-416: Use After Free
- CWE-476: NULL Pointer Dereference

## Optimizaciones incluidas

- **Mixed Precision (FP16)**: Acelera entrenamiento en GPU
- **Gradient Accumulation**: Permite batches grandes virtuales
- **Class Weights**: Maneja desbalance de clases
- **Early Stopping**: Evita overfitting
- **Checkpointing**: Guarda progreso automáticamente
- **Streaming Dataset**: Soporta archivos de 50GB+
