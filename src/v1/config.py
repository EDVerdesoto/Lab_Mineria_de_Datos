"""
Configuración central del proyecto.
"""

import os
from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_DIR = BASE_DIR / "models" / "saved"

# Crear directorios si no existen
for dir_path in [DATA_DIR, CHECKPOINT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_PATH = DATA_DIR / "dataset_limpio_v2.csv"
CHUNK_SIZE = 1000  # Registros por chunk para streaming

# Modelo
NUM_CLASSES = 2  # Binario: Safe vs Vulnerable (cambiar a N para multi-clase CWE)
MAX_SEQ_LENGTH = 512  # Longitud máxima de tokens para CodeBERT
CODEBERT_MODEL = "microsoft/codebert-base"

# Entrenamiento
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
ACCUMULATION_STEPS = 4  # Batch efectivo = BATCH_SIZE * ACCUMULATION_STEPS
TEST_SIZE = 0.2
WARMUP_RATIO = 0.1

# Extractores
MAX_CODE_LENGTH = 50_000  # Límite para evitar regex catastrophic backtracking
HASHING_FEATURES = 4096  # Para TextExtractor (no usado en híbrido)

# Clases CWE más comunes (para clasificación multi-clase)
TOP_CWES = [
    "CWE-79",   # XSS
    "CWE-89",   # SQL Injection
    "CWE-119",  # Buffer Overflow
    "CWE-125",  # Out-of-bounds Read
    "CWE-200",  # Information Exposure
    "CWE-264",  # Permissions
    "CWE-287",  # Authentication
    "CWE-352",  # CSRF
    "CWE-416",  # Use After Free
    "CWE-476",  # NULL Pointer Dereference
]

# Mapeo de clases
CLASS_NAMES_BINARY = ["Vulnerable", "Safe"]
CLASS_NAMES_MULTICLASS = ["Safe"] + TOP_CWES + ["Other"]

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
