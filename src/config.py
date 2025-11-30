import os

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cvefixes_grande.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "vulnerability_xgb.json")

# Parámetros del Modelo
CHUNK_SIZE = 5000
HASHING_FEATURES = 2**12
TEST_SIZE = 0.2

# Mapeo de Clases
CWE_MAP = {
    'Safe': 0,
    'CWE-119': 1,
    'CWE-89': 2,
    'CWE-79': 3,
    'Other': 4
}

ID_TO_CWE = {v: k for k, v in CWE_MAP.items()}
NUM_CLASSES = len(CWE_MAP)  # Agregar esto para evitar hardcodear