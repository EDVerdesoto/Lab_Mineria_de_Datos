import pandas as pd
import numpy as np
from embedder import CodeEmbedder
from maps import LABEL_MAP, OTHER_LABEL, LANG_MAP, NUM_LANGS

def get_label(row):
    cwe = row['cwe_id']
    if cwe == 'Safe':
        return 0
    elif cwe in LABEL_MAP:
        return LABEL_MAP[cwe]
    return OTHER_LABEL

def get_language_vector(lang_val):
    """
    Genera un vector One-Hot basado en el lenguaje.
    Ej: Si es 'C' -> [0, 1, 0, 0, ... 0]
    Si es 'unknown' o 'None' -> [0, 0, ... 1]
    """
    vec = np.zeros(NUM_LANGS)
    
    # Limpieza: convertir a string y quitar espacios
    s_lang = str(lang_val).strip()
    
    # Verificamos si es uno de los casos "nulos"
    if s_lang in ['None', 'unknown', 'nan', '']:
        idx = NUM_LANGS - 1 # Última posición (Desconocido)
    else:
        # Buscamos en el mapa. Si aparece algo raro no listado, va a Desconocido.
        idx = LANG_MAP.get(s_lang, NUM_LANGS - 1)
    
    vec[idx] = 1.0
    return vec

def data_generator(csv_path, chunk_size=5000):
    """
    Generador que lee el CSV por partes y devuelve (X, y) listos para entrenar.
    """

    # Instanciamos el embedder una sola vez
    embedder = CodeEmbedder(batch_size=64)
    
    # Columnas numéricas adicionales al embedder de CodeBERT
    numeric_cols = ['nloc', 'complexity', 'token_count', 'top_nesting_level', 'parameters_count']
    
    print(f"[INFO] Iniciando lectura de {csv_path} en chunks de {chunk_size}...")
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # 1. Limpieza básica
        chunk = chunk.dropna(subset=['code'])
        # Aseguramos que las métricas sean numéricas
        chunk[numeric_cols] = chunk[numeric_cols].fillna(0).astype(float)
        
        if chunk.empty:
            continue

        # 2. Generar Target (y)
        y = chunk.apply(get_label, axis=1).values
        
        # 3. Generar Embeddings (X parte 1)
        embeddings = embedder.get_embeddings(chunk['code'].tolist())

        # 4. Generar Vector de Lenguaje de Programación y concatenar (X parte 2)
        lang_vectors = np.vstack(chunk['programming_language'].apply(get_language_vector).values)      

        # 5. Obtener Métricas Manuales (X parte 3)
        metrics = chunk[numeric_cols].values
        
        # 6. Concatenar (X final)
        # X shape: [n_samples, 5 + 768 + NUM_LANGS]
        X = np.hstack((metrics, embeddings, lang_vectors))
        
        yield X, y