import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v3.dataset import VulnDataset
from src.v3.map_cwe import get_label_id

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, "data", "processed", "dataset_ml_ready.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dataset_cached.pt")
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
STRIDE = 256
MAX_WINDOWS = 8
USE_AUGMENTATION = False  # ¿Quieres guardar datos aumentados?
AUGMENT_PROB = 0.3      # Si es True, 30% de probabilidad por muestra
MASK_PROB = 0.10

# Función auxiliar para procesar un chunk de datos en un proceso separado
def process_chunk(args):
    """
    Esta función se ejecuta en paralelo en cada núcleo.
    Recibe un subconjunto de datos y devuelve la lista procesada.
    """
    df_chunk, tokenizer_path = args
    
    # Re-instanciar tokenizer dentro del proceso (necesario para multiprocessing en Windows)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Usamos tu clase original para aprovechar su lógica
    # Nota: Desactivamos el 'augment' del dataset para controlarlo manualmente si quisiéramos
    # pero aquí lo dejamos activado según la config global.
    dataset = VulnDataset(
        codes=df_chunk['code'].values,
        labels=df_chunk['label_id'].values,
        tokenizer=tokenizer,
        training=USE_AUGMENTATION, # ¡OJO! Si es True, quemará la augmentación en el archivo
        use_augmentation=USE_AUGMENTATION,
        use_sliding_window=True,
        augment_prob=AUGMENT_PROB,
        mask_prob=MASK_PROB,
        max_len=MAX_LEN,
        stride=STRIDE,
        max_windows=MAX_WINDOWS
    )
    
    processed_samples = []
    # Iteramos manualmente para extraer los datos ya procesados (__getitem__)
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            processed_samples.append(sample)
        except Exception as e:
            # Capturar errores para no tumbar todo el proceso
            print(f"Error en muestra {i}: {e}")
            continue
            
    return processed_samples

def main():
    print(f"Cargando {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, usecols=['code', 'cwe_id']).dropna(subset=['code'])
    df['label_id'] = df['cwe_id'].apply(get_label_id)
    
    # Solo usaremos TRAIN para esto (Validation no debe tener augmentation)
    # Aquí puedes filtrar si quieres solo procesar train o todo
    print(f"Total muestras: {len(df)}")

    # Configurar Multiprocessing
    num_cores = 4
    print(f"Iniciando procesamiento paralelo con {num_cores} núcleos...")
    
    # Dividir el dataframe en chunks
    df_chunks = np.array_split(df, num_cores * 4) # 4 tareas por núcleo para mejor balanceo
    
    # Preparar argumentos (pasamos el path del tokenizer para cargarlo en cada proceso)
    tasks = [(chunk, MODEL_NAME) for chunk in df_chunks]
    
    all_data = []
    
    # Ejecutar Pool
    with Pool(processes=num_cores) as pool:
        # Usamos tqdm para barra de progreso
        results = list(tqdm(pool.imap(process_chunk, tasks), total=len(tasks), unit="chunk"))
    
    # Aplanar lista de listas
    print("Uniendo resultados...")
    for res in results:
        all_data.extend(res)
        
    print(f"Guardando {len(all_data)} muestras procesadas en {OUTPUT_FILE}...")
    torch.save(all_data, OUTPUT_FILE)

if __name__ == '__main__':
    main()