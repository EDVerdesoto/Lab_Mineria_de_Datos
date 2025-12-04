import pandas as pd
import torch
import numpy as np
import gc
import shutil
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import sys
import os

# Ajusta esto a tu estructura de carpetas real
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v3.dataset import VulnDataset
from src.v3.map_cwe import get_label_id

# Estrategia para evitar límite de archivos abiertos
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except:
    pass

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, "data", "processed", "dataset_ml_ready.csv")

# CAMBIO: Usamos directorios en lugar de archivos únicos
OUTPUT_DIR_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_cache")
OUTPUT_DIR_VAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_cache")

MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
STRIDE = 256
MAX_WINDOWS = 8
AUGMENT_PROB = 0.3
MASK_PROB = 0.10
TEST_SIZE = 0.1
SEED = 42

def process_chunk(args):
    """Procesa un chunk y retorna una lista de diccionarios (tensores)"""
    df_chunk, tokenizer_path, use_augmentation = args
    
    # Re-instanciar tokenizer en cada proceso
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    dataset = VulnDataset(
        codes=df_chunk['code'].values,
        labels=df_chunk['label_id'].values,
        tokenizer=tokenizer,
        training=use_augmentation,
        use_augmentation=use_augmentation,
        use_sliding_window=True,
        augment_prob=AUGMENT_PROB if use_augmentation else 0,
        mask_prob=MASK_PROB if use_augmentation else 0,
        max_len=MAX_LEN,
        stride=STRIDE,
        max_windows=MAX_WINDOWS
    )
    
    processed_samples = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            processed_samples.append(sample)
        except Exception as e:
            print(f"Error ignorable en muestra {i}: {e}")
            continue
            
    return processed_samples

def setup_output_dir(directory):
    """Crea directorio limpio para guardar partes"""
    if os.path.exists(directory):
        print(f"Limpiando directorio existente: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory)

def process_and_save(df, output_dir, use_augment, num_cores):
    """Lógica genérica para procesar y guardar incrementalmente"""
    print(f"Procesando {len(df)} muestras -> {output_dir}")
    
    # Dividir en muchos chunks pequeños (ej. 40 chunks) para liberar RAM rápido
    # Cuantos más chunks, menos RAM usa, pero más archivos genera.
    num_chunks = num_cores * 10 
    df_chunks = np.array_split(df, num_chunks)
    
    tasks = [(chunk, MODEL_NAME, use_augment) for chunk in df_chunks]
    
    with Pool(processes=num_cores) as pool:
        # Usamos imap para recibir resultados apenas estén listos
        for i, result in tqdm(enumerate(pool.imap(process_chunk, tasks)), total=len(tasks)):
            if not result: continue # Saltar chunks vacíos si los hay
            
            # GUARDAR INMEDIATAMENTE
            save_path = os.path.join(output_dir, f"part_{i}.pt")
            torch.save(result, save_path)
            
            # LIBERAR MEMORIA
            del result
            gc.collect()

def main():
    # Fix para multiprocessing en algunos Linux
    try:
        set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    print(f"Cargando {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, usecols=['code', 'cwe_id']).dropna(subset=['code'])
    df['label_id'] = df['cwe_id'].apply(get_label_id)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['label_id'], random_state=SEED
    )
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}")

    # Preparar carpetas
    setup_output_dir(OUTPUT_DIR_TRAIN)
    setup_output_dir(OUTPUT_DIR_VAL)

    # Configurar cores
    num_cores = 4
    
    # 1. PROCESAR TRAIN
    print(f"\n{'='*40}\nIniciando TRAIN (Augmentation=True)\n{'='*40}")
    process_and_save(train_df, OUTPUT_DIR_TRAIN, use_augment=True, num_cores=num_cores)
    
    # 2. PROCESAR VAL
    print(f"\n{'='*40}\nIniciando VAL (Augmentation=False)\n{'='*40}")
    process_and_save(val_df, OUTPUT_DIR_VAL, use_augment=False, num_cores=num_cores)

    print("\n[SUCCESS] Todo procesado y guardado en carpetas.")

if __name__ == '__main__':
    main()