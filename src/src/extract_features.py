"""
Módulo de Extracción de Características (Feature Engineering Pipeline).

Este script procesa un conjunto de datos de código fuente (SARD Dataset), realiza análisis estático
para extraer métricas de complejidad y seguridad, y genera un dataset normalizado listo
para el entrenamiento de modelos de Machine Learning.

Attributes:
    RAW_DATA_PATH (str): Ruta al directorio que contiene los archivos .py crudos.
    PROCESSED_DATA_PATH (str): Ruta de salida para los datasets procesados.
    OUTPUT_CSV (str): Ruta del archivo CSV con métricas crudas.
    OUTPUT_NORMALIZED_CSV (str): Ruta del archivo CSV con métricas normalizadas (StandardScaler).
"""

import os
import sys
import ast
import lizard
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Set

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DATA_PATH = "data/raw/sard"
PROCESSED_DATA_PATH = "data/processed"
OUTPUT_CSV = os.path.join(PROCESSED_DATA_PATH, "features.csv")
OUTPUT_NORMALIZED_CSV = os.path.join(PROCESSED_DATA_PATH, "features_normalized.csv")


class SecurityVisitor(ast.NodeVisitor):
    """
    Visitante de AST (Abstract Syntax Tree) diseñado para extraer métricas de seguridad y estructura.

    Hereda de `ast.NodeVisitor` para recorrer el árbol sintáctico de Python.
    Calcula la profundidad de anidamiento real (ignorando definiciones de funciones)
    y detecta el uso de funciones potencialmente peligrosas.

    Attributes:
        max_depth (int): La profundidad máxima de anidamiento encontrada en el código.
        current_depth (int): Profundidad actual durante el recorrido recursivo.
        dangerous_count (int): Contador de llamadas a funciones peligrosas detectadas.
        dangerous_funcs (Set[str]): Conjunto de nombres de funciones consideradas peligrosas (ej: 'eval').
    """

    def __init__(self):
        """Inicializa el visitante con contadores a cero y define la lista negra de funciones."""
        self.max_depth: int = 0
        self.current_depth: int = 0
        self.dangerous_count: int = 0
        self.dangerous_funcs: Set[str] = {'eval', 'exec'}

    def generic_visit(self, node: ast.AST) -> None:
        """
        Visita genérica para calcular la profundidad de anidamiento.

        Incrementa la profundidad solo si el nodo es una estructura de control de flujo
        (If, For, While, Try, With).

        Args:
            node (ast.AST): El nodo actual del árbol sintáctico.
        """
        # Nodos que incrementan la complejidad de lectura/anidamiento
        nesting_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With)

        if isinstance(node, nesting_nodes):
            self.current_depth += 1
            if self.current_depth > self.max_depth:
                self.max_depth = self.current_depth
            super().generic_visit(node)
            self.current_depth -= 1
        else:
            super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visita nodos de llamada a función para detectar vulnerabilidades de inyección.

        Verifica si la función llamada está en la lista negra (`dangerous_funcs`) o
        si es una llamada crítica de sistema como `os.system`.

        Args:
            node (ast.Call): Nodo de llamada a función.
        """
        # Caso 1: Funciones built-in directas (ej: eval("..."))
        if isinstance(node.func, ast.Name):
            if node.func.id in self.dangerous_funcs:
                self.dangerous_count += 1
        
        # Caso 2: Métodos de atributos (ej: os.system("..."))
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == 'system':
                self.dangerous_count += 1
        
        self.generic_visit(node)


def analyze_single_file(filepath: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Ejecuta el análisis estático completo sobre un único archivo Python.

    Combina Lizard (para métricas estándar) y AST personalizado (para métricas de seguridad).

    Args:
        filepath (str): Ruta absoluta o relativa al archivo.
        filename (str): Nombre del archivo (usado para etiquetado heurístico).

    Returns:
        Optional[Dict[str, Any]]: Un diccionario con las características extraídas:
            - file_id: Nombre del archivo.
            - loc: Líneas de código (Lizard).
            - complexity: Complejidad ciclomática promedio (Lizard).
            - nesting_depth: Profundidad máxima (AST).
            - num_functions: Cantidad de funciones definidas.
            - uses_dangerous_funcs: Conteo de funciones peligrosas.
            - is_vulnerable: Etiqueta binaria (0 o 1).
            
            Retorna `None` si el archivo no puede ser parseado o leído.
    """
    # 1. Análisis con LIZARD (Métricas de Industria)
    try:
        liz_data = lizard.analyze_file(filepath)
        loc = liz_data.nloc
        num_funcs = len(liz_data.function_list)
        avg_complexity = liz_data.average_cyclomatic_complexity
    except Exception:
        # Falla silenciosa controlada para archivos corruptos en el dataset
        return None

    # 2. Análisis con AST (Métricas de Seguridad y Estructura)
    max_nesting = 0
    dangerous_uses = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            tree = ast.parse(content)
            visitor = SecurityVisitor()
            visitor.visit(tree)
            max_nesting = visitor.max_depth
            dangerous_uses = visitor.dangerous_count
    except SyntaxError:
        # El archivo tiene errores de sintaxis Python, se asignan valores por defecto
        pass
    except Exception as e:
        print(f"[WARN] Error procesando AST de {filename}: {e}")
        return None

    # 3. Etiquetado (Labeling) Heurístico
    # TODO: Ajustar esta lógica según el manifiesto XML oficial de SARD si es necesario.
    is_vulnerable = 0
    name_lower = filename.lower()
    if "bad" in name_lower or "vuln" in name_lower:
        is_vulnerable = 1
    elif "good" in name_lower or "fixed" in name_lower:
        is_vulnerable = 0
    
    return {
        "file_id": filename,
        "loc": loc,
        "complexity": avg_complexity,
        "nesting_depth": max_nesting,
        "num_functions": num_funcs,
        "uses_dangerous_funcs": dangerous_uses,
        "is_vulnerable": is_vulnerable
    }


def process_dataset() -> None:
    """
    Orquesta el pipeline ETL (Extract, Transform, Load) completo.

    Pasos:
    1. Recorre recursivamente `RAW_DATA_PATH`.
    2. Extrae características de cada archivo `.py`.
    3. Construye un DataFrame de Pandas.
    4. Guarda el dataset crudo.
    5. Normaliza las características numéricas usando `StandardScaler` (Z-score).
    6. Guarda el dataset normalizado listo para ML.

    Raises:
        FileNotFoundError: Si el directorio de datos crudos no existe.
        SystemExit: Si no se encuentran archivos para procesar.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] No existe el directorio de origen: {RAW_DATA_PATH}")
        sys.exit(1)
        
    data_rows = []
    print(f"[*] Iniciando procesamiento masivo en: {RAW_DATA_PATH}...")

    # --- FASE 1: EXTRACCIÓN ---
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith(".py"):
                fullpath = os.path.join(root, file)
                row = analyze_single_file(fullpath, file)
                if row:
                    data_rows.append(row)

    if not data_rows:
        print("[ERROR] No se encontraron archivos .py validos para procesar.")
        sys.exit(1)

    df = pd.DataFrame(data_rows)
    print(f"[*] Procesamiento finalizado. Archivos analizados: {len(df)}")

    # Asegurar existencia del directorio de salida
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # --- FASE 2: CARGA (RAW) ---
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Dataset crudo guardado en: {OUTPUT_CSV}")

    # --- FASE 3: TRANSFORMACIÓN (NORMALIZACIÓN) ---
    # Seleccionamos solo las columnas numéricas que influyen en el modelo.
    # NO normalizamos 'is_vulnerable' (target) ni 'file_id' (identificador).
    features_to_scale = ['loc', 'complexity', 'nesting_depth', 'num_functions', 'uses_dangerous_funcs']
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    
    # Aplicamos Z-score normalization: z = (x - u) / s
    df_normalized[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    df_normalized.to_csv(OUTPUT_NORMALIZED_CSV, index=False)
    print(f"[OK] Dataset normalizado guardado en: {OUTPUT_NORMALIZED_CSV}")
    
    print("\n--- Vista Previa del Dataset Normalizado ---")
    print(df_normalized.head())


if __name__ == "__main__":
    process_dataset()