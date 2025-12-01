"""
Feature Extraction Pipeline (Multilanguage Support).

Este módulo procesa código fuente en múltiples lenguajes (Python, C++, Java, JS, PHP, C#, TS)
para extraer métricas de calidad y seguridad para modelos de Machine Learning.

Soporta:
- Métricas de Estilo: LOC, Complejidad Ciclomática (via Lizard).
- Métricas Estructurales: Profundidad de Anidamiento (Heurística).
- Métricas de Seguridad: Detección de funciones peligrosas (Regex).
"""

import os
import sys
import re
import lizard
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_CSV = os.path.join(PROCESSED_DATA_PATH, "features.csv")
OUTPUT_NORMALIZED_CSV = os.path.join(PROCESSED_DATA_PATH, "features_normalized.csv")

# --- CONFIGURACIÓN DE LENGUAJES ---
# Define extensiones y patrones de riesgo específicos por lenguaje
# --- CONFIGURACIÓN DE LENGUAJES ---
LANGUAGE_CONFIG = {
    # C/C++: Buscamos llamadas a función exactas con limite de palabra
    '.c':   {'style': 'braces', 'dangerous': [r'\bsystem\(', r'\bexecl\(', r'\bpopen\(', r'\bstrcpy\(']},
    '.cpp': {'style': 'braces', 'dangerous': [r'\bsystem\(', r'\bexecl\(', r'\bpopen\(', r'\bstrcpy\(']},
    '.h':   {'style': 'braces', 'dangerous': [r'\bsystem\(', r'\bstrcpy\(']},
    
    # Java: Ajustado para evitar coincidir con comentarios o declaraciones de variables
    # Buscamos 'Runtime.getRuntime().exec(' o 'new ProcessBuilder'
    '.java':{'style': 'braces', 'dangerous': [r'\.exec\(', r'new ProcessBuilder', r'\bnative ']},
    
    '.cs':  {'style': 'braces', 'dangerous': [r'Process\.Start\(', r'wscript\.shell']},
    
    # JS/TS: Eval y ejecución de strings
    '.js':  {'style': 'braces', 'dangerous': [r'\beval\(', r'\bsetTimeout\(.*[\'"]', r'\bexec\(']},
    '.ts':  {'style': 'braces', 'dangerous': [r'\beval\(', r'\bsetTimeout\(.*[\'"]', r'\bexec\(']},
    
    # PHP: Limites de palabra estrictos
    '.php': {'style': 'braces', 'dangerous': [r'\beval\(', r'\bexec\(', r'\bsystem\(', r'\bshell_exec\(']},
    
    # Python
    '.py':  {'style': 'indent', 'dangerous': [r'\beval\(', r'\bexec\(', r'os\.system\(', r'subprocess\.']}
}

class StaticAnalyzer:
    """Clase utilitaria para análisis estático ligero basado en texto."""

    @staticmethod
    def calculate_nesting_depth(content: str, style: str) -> int:
        """
        Calcula la profundidad máxima de anidamiento.
        
        Args:
            content (str): Código fuente.
            style (str): 'braces' (C/Java/JS) o 'indent' (Python).
            
        Returns:
            int: Profundidad máxima detectada.
        """
        max_depth = 0
        current_depth = 0
        
        lines = content.splitlines()

        if style == 'braces':
            # Heurística: Contar llaves { y } ignorando comentarios (simplificado)
            for line in lines:
                # Limpieza básica
                clean_line = line.strip()
                # Incremento por apertura
                current_depth += clean_line.count('{')
                # Decremento por cierre
                current_depth -= clean_line.count('}')
                
                # En C/Java, es común cerrar y abrir en la misma linea: } else {
                # El max debe registrarse antes de cerrar si aplica
                if current_depth > max_depth:
                    max_depth = current_depth
            
            return max(0, max_depth) # Evitar negativos por errores de sintaxis

        elif style == 'indent':
            # Heurística para Python: Contar espacios al inicio
            # Asumimos 4 espacios por nivel (PEP8 estándar)
            for line in lines:
                stripped = line.lstrip()
                if not stripped or stripped.startswith('#'):
                    continue # Ignorar vacías o comentarios
                
                # Calcular espacios
                spaces = len(line) - len(stripped)
                indent_level = spaces // 4
                
                # Ajuste: 'def' y 'class' son nivel 0 para nosotros, 
                # pero el contenido estará identado.
                # Aquí tomamos el nivel crudo de indentación como proxy de anidamiento.
                if indent_level > max_depth:
                    max_depth = indent_level
            
            return max(0, max_depth)
        
        return 0

    @staticmethod
    def count_dangerous_functions(content: str, patterns: List[str]) -> int:
        """
        Cuenta ocurrencias de funciones peligrosas usando Regex.
        
        Args:
            content (str): Código fuente.
            patterns (List[str]): Lista de regex patterns para el lenguaje.
        """
        count = 0
        for pattern in patterns:
            # Re.IGNORECASE puede ser util, pero en C/Python es case sensitive.
            # Usamos findall para contar todas las apariciones
            matches = re.findall(pattern, content)
            count += len(matches)
        return count


def analyze_file(filepath: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Analiza un archivo individual soportando múltiples lenguajes.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # Si la extensión no está soportada, ignoramos el archivo
    if ext not in LANGUAGE_CONFIG:
        return None

    config = LANGUAGE_CONFIG[ext]

    # 1. Análisis Universal (Lizard)
    try:
        liz_data = lizard.analyze_file(filepath)
        loc = liz_data.nloc
        num_funcs = len(liz_data.function_list)
        
        if num_funcs > 0:
            avg_complexity = liz_data.average_cyclomatic_complexity
        else:
            # Script plano o clase sin métodos detectados
            avg_complexity = 1.0
            
    except Exception as e:
        # print(f"[WARN] Lizard falló en {filename}: {e}")
        return None

    # 2. Análisis Específico (Lectura de texto)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Profundidad de Anidamiento
            nesting = StaticAnalyzer.calculate_nesting_depth(content, config['style'])
            
            # Funciones Peligrosas
            dangerous_count = StaticAnalyzer.count_dangerous_functions(content, config['dangerous'])
            
    except Exception as e:
        print(f"[ERROR] Lectura fallida {filename}: {e}")
        return None

    # 3. Etiquetado (Labeling)
    # Heurística basada en nombres comunes del dataset SARD
    is_vulnerable = 0
    fname_lower = filename.lower()
    if any(x in fname_lower for x in ['bad', 'vuln', 'unsafe']):
        is_vulnerable = 1
    elif any(x in fname_lower for x in ['good', 'fixed', 'safe']):
        is_vulnerable = 0
    
    return {
        "file_id": filename,
        "loc": loc,
        "complexity": avg_complexity,
        "nesting_depth": nesting,
        "num_functions": num_funcs,
        "uses_dangerous_funcs": dangerous_count,
        "is_vulnerable": is_vulnerable
    }

def process_dataset() -> None:
    """Función principal del pipeline ETL."""
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] Directorio no encontrado: {RAW_DATA_PATH}")
        sys.exit(1)
        
    data_rows = []
    print(f"[*] Escaneando {RAW_DATA_PATH} para lenguajes: {list(LANGUAGE_CONFIG.keys())}...")

    # Recorrido recursivo
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            fullpath = os.path.join(root, file)
            # Solo intentamos analizar si tiene extensión conocida
            row = analyze_file(fullpath, file)
            if row:
                data_rows.append(row)

    if not data_rows:
        print("[ERROR] No se generaron datos. Verifica la ruta o las extensiones.")
        sys.exit(1)

    df = pd.DataFrame(data_rows)
    print(f"[*] Archivos procesados: {len(df)}")
    print(f"[*] Distribución de vulnerables:\n{df['is_vulnerable'].value_counts()}")

    # Crear directorios
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # 1. Guardar RAW
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Raw features: {OUTPUT_CSV}")

    # 2. Normalizar (StandardScaler)
    features_to_scale = ['loc', 'complexity', 'nesting_depth', 'num_functions', 'uses_dangerous_funcs']
    
    # Validar que las columnas existan y tengan datos numéricos
    for col in features_to_scale:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    scaler = StandardScaler()
    df_normalized = df.copy()
    
    # Transformación
    df_normalized[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Guardar NORMALIZED
    df_normalized.to_csv(OUTPUT_NORMALIZED_CSV, index=False)
    print(f"[OK] Normalized features: {OUTPUT_NORMALIZED_CSV}")
    print("\n--- Vista Previa (Normalizada) ---")
    print(df_normalized.head())

if __name__ == "__main__":
    process_dataset()