# Mapa de las CWEs más comunes a IDs de etiquetas
TOP_CWES = [
    'CWE-79',   # XSS
    'CWE-89',   # SQL Injection
    'CWE-78',   # OS Command Injection
    'CWE-22',   # Path Traversal
    'CWE-434',  # Unrestricted File Upload
    'CWE-352',  # CSRF
]

LABEL_NAMES = ['Safe'] + TOP_CWES + ['Other']
NUM_LABELS = len(LABEL_NAMES)

# Mapa CWE -> ID
LABEL_MAP = {cwe: i + 1 for i, cwe in enumerate(TOP_CWES)}
OTHER_LABEL = len(TOP_CWES) + 1

def get_label_id(cwe_id: str) -> int:
    """Convierte CWE ID string a label numérico."""
    if cwe_id == 'Safe':
        return 0
    return LABEL_MAP.get(cwe_id, OTHER_LABEL)