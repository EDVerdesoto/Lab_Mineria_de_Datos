import pandas as pd
from config import DATA_PATH, CHUNK_SIZE, CWE_MAP

def get_data_chunks():
    """
    Generador que lee el CSV preprocesado por chunks.
    """
    try:
        for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
            required_cols = ['code', 'language', 'safety', 'cwe_id']
            if not all(col in chunk.columns for col in required_cols):
                raise ValueError(f"El CSV debe tener estas columnas: {required_cols}")
            
            chunk['target'] = chunk['cwe_id'].map(CWE_MAP).fillna(CWE_MAP['Other'])
            chunk.loc[chunk['cwe_id'].isna(), 'target'] = CWE_MAP['Safe']
            chunk['target'] = chunk['target'].astype(int)
            
            yield chunk
            
    except FileNotFoundError:
        print(f"[ERROR] No se encuentra el dataset en {DATA_PATH}")
        raise