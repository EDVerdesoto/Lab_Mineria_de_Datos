"""
Datasets para entrenamiento con streaming de archivos grandes.
Optimizado con pre-extracción de features y procesamiento paralelo.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from typing import List, Optional, Iterator, Tuple
from transformers import PreTrainedTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from extractors import ComplexityExtractor, PatternExtractor, ASTExtractor
from config import MAX_SEQ_LENGTH, CHUNK_SIZE, TOP_CWES, FEATURE_EXTRACTION_WORKERS


def _extract_all_features(args: Tuple) -> Tuple[int, List[float], List[float], List[float]]:
    """
    Worker function para extracción paralela.
    Retorna (idx, complexity_feats, pattern_feats, ast_feats)
    """
    idx, code, lang, complexity_ext, pattern_ext, ast_ext = args
    return (
        idx,
        complexity_ext.extract(code, lang),
        pattern_ext.extract(code, lang),
        ast_ext.extract(code, lang)
    )


class HybridCodeDataset(Dataset):
    """
    Dataset para batches en memoria.
    Extrae features de CodeBERT + Lizard + Patterns + AST.
    
    OPTIMIZACIÓN: Pre-extrae todas las features al inicializar
    para evitar llamar Lizard en cada __getitem__.
    """
    
    def __init__(
        self, 
        codes: List[str],
        labels: List[int],
        languages: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_SEQ_LENGTH,
        precompute_features: bool = True,
        num_workers: int = FEATURE_EXTRACTION_WORKERS
    ):
        self.codes = codes
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Inicializar extractores (compartidos)
        self.complexity_extractor = ComplexityExtractor()
        self.pattern_extractor = PatternExtractor()
        self.ast_extractor = ASTExtractor()
        
        # Pre-computar features para evitar Lizard en cada __getitem__
        self._precomputed = None
        if precompute_features:
            self._precompute_all_features(num_workers)
    
    def _precompute_all_features(self, num_workers: int = 4):
        """
        Pre-extrae todas las features usando múltiples threads.
        Lizard libera el GIL, así que threading funciona bien.
        """
        print(f"[*] Pre-extrayendo features para {len(self.codes)} muestras...")
        
        n = len(self.codes)
        self._precomputed = {
            'complexity': [None] * n,
            'pattern': [None] * n,
            'ast': [None] * n
        }
        
        # Preparar argumentos
        args_list = [
            (i, str(self.codes[i]), 
             self.languages[i] if self.languages else None,
             self.complexity_extractor,
             self.pattern_extractor, 
             self.ast_extractor)
            for i in range(n)
        ]
        
        # Extracción paralela con ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_extract_all_features, args): args[0] 
                      for args in args_list}
            
            for future in tqdm(as_completed(futures), total=n, desc="Extrayendo features"):
                try:
                    idx, complexity, pattern, ast_feats = future.result()
                    self._precomputed['complexity'][idx] = complexity
                    self._precomputed['pattern'][idx] = pattern
                    self._precomputed['ast'][idx] = ast_feats
                except Exception as e:
                    idx = futures[future]
                    # Fallback a valores por defecto
                    self._precomputed['complexity'][idx] = [0.0] * 5
                    self._precomputed['pattern'][idx] = [0.0] * 5
                    self._precomputed['ast'][idx] = [0.0] * 2
        
        # Convertir a numpy arrays para acceso más rápido
        self._precomputed['complexity'] = np.array(self._precomputed['complexity'], dtype=np.float32)
        self._precomputed['pattern'] = np.array(self._precomputed['pattern'], dtype=np.float32)
        self._precomputed['ast'] = np.array(self._precomputed['ast'], dtype=np.float32)
        
        print(f"[OK] Features pre-extraídas. Cache stats:")
        print(f"     Complexity: {self.complexity_extractor.get_cache_stats()}")
        print(f"     Pattern: {self.pattern_extractor.get_cache_stats()}")
    
    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int) -> dict:
        code = str(self.codes[idx])
        
        # Tokenización para CodeBERT
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Usar features pre-computadas si están disponibles
        if self._precomputed is not None:
            complexity_feats = self._precomputed['complexity'][idx]
            pattern_feats = self._precomputed['pattern'][idx]
            ast_feats = self._precomputed['ast'][idx]
        else:
            # Fallback: extraer en tiempo real (lento)
            lang = self.languages[idx] if self.languages else None
            complexity_feats = self.complexity_extractor.extract(code, lang)
            pattern_feats = self.pattern_extractor.extract(code, lang)
            ast_feats = self.ast_extractor.extract(code, lang)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'complexity_features': torch.tensor(complexity_feats, dtype=torch.float32),
            'pattern_features': torch.tensor(pattern_feats, dtype=torch.float32),
            'ast_features': torch.tensor(ast_feats, dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class StreamingCodeDataset(IterableDataset):
    """
    Dataset para streaming de archivos grandes (50GB+).
    Lee en chunks para no cargar todo en RAM.
    
    OPTIMIZACIÓN: Procesa features por batch dentro de cada chunk
    usando ThreadPoolExecutor.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        mode: str = 'binary',  # 'binary' o 'multiclass'
        chunk_size: int = CHUNK_SIZE,
        max_length: int = MAX_SEQ_LENGTH,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
        num_workers: int = FEATURE_EXTRACTION_WORKERS
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.skip_rows = skip_rows
        self.max_rows = max_rows
        self.num_workers = num_workers
        
        # Extractores (compartidos entre workers)
        self.complexity_extractor = ComplexityExtractor()
        self.pattern_extractor = PatternExtractor()
        self.ast_extractor = ASTExtractor()
        
        # Mapeo de CWEs para modo multiclass
        self.cwe_to_idx = {cwe: i + 1 for i, cwe in enumerate(TOP_CWES)}
        self.cwe_to_idx['Safe'] = 0
        self.cwe_to_idx['Other'] = len(TOP_CWES) + 1
    
    def _get_label(self, row: pd.Series) -> int:
        """Obtiene el label según el modo de clasificación."""
        safety = str(row.get('safety', '')).lower().strip()
        
        if self.mode == 'binary':
            # 0 = Vulnerable, 1 = Safe
            return 1 if safety == 'safe' else 0
        
        else:  # multiclass
            if safety == 'safe':
                return self.cwe_to_idx['Safe']
            
            cwe_id = str(row.get('cwe_id', '')).strip()
            
            if cwe_id in self.cwe_to_idx:
                return self.cwe_to_idx[cwe_id]
            else:
                return self.cwe_to_idx['Other']
    
    def _process_chunk_parallel(self, chunk: pd.DataFrame) -> List[dict]:
        """Procesa un chunk completo en paralelo."""
        chunk = chunk.dropna(subset=['code', 'safety'])
        if len(chunk) == 0:
            return []
        
        codes = chunk['code'].astype(str).tolist()
        langs = chunk.get('programming_language', pd.Series([None] * len(chunk))).tolist()
        labels = [self._get_label(row) for _, row in chunk.iterrows()]
        
        n = len(codes)
        
        # Preparar argumentos para extracción paralela
        args_list = [
            (i, codes[i], langs[i],
             self.complexity_extractor,
             self.pattern_extractor,
             self.ast_extractor)
            for i in range(n)
        ]
        
        # Extraer features en paralelo
        results = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_extract_all_features, args): args[0] 
                      for args in args_list}
            
            for future in as_completed(futures):
                try:
                    idx, complexity, pattern, ast_feats = future.result()
                    results[idx] = (complexity, pattern, ast_feats)
                except Exception:
                    idx = futures[future]
                    results[idx] = ([0.0]*5, [0.0]*5, [0.0]*2)
        
        # Construir samples
        samples = []
        for i in range(n):
            code = codes[i]
            complexity_feats, pattern_feats, ast_feats = results[i]
            
            encoding = self.tokenizer(
                code,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            samples.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'complexity_features': torch.tensor(complexity_feats, dtype=torch.float32),
                'pattern_features': torch.tensor(pattern_feats, dtype=torch.float32),
                'ast_features': torch.tensor(ast_feats, dtype=torch.float32),
                'label': torch.tensor(labels[i], dtype=torch.long)
            })
        
        return samples
    
    def __iter__(self) -> Iterator[dict]:
        rows_processed = 0
        
        reader = pd.read_csv(
            self.file_path, 
            chunksize=self.chunk_size,
            skiprows=range(1, self.skip_rows + 1) if self.skip_rows > 0 else None
        )
        
        for chunk in reader:
            # Procesar chunk completo en paralelo
            samples = self._process_chunk_parallel(chunk)
            
            for sample in samples:
                if self.max_rows and rows_processed >= self.max_rows:
                    return
                
                rows_processed += 1
                yield sample


def create_datasets_from_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    mode: str = 'binary',
    test_size: float = 0.2
) -> tuple:
    """
    Crea datasets de entrenamiento y validación desde un DataFrame.
    
    Args:
        df: DataFrame con columnas code, safety, programming_language, cwe_id
        tokenizer: Tokenizer de CodeBERT
        mode: 'binary' o 'multiclass'
        test_size: Proporción para validación
    
    Returns:
        (train_dataset, val_dataset)
    """
    from sklearn.model_selection import train_test_split
    
    df = df.dropna(subset=['code', 'safety'])
    
    # Preparar labels
    if mode == 'binary':
        df['label'] = df['safety'].apply(lambda x: 1 if str(x).lower() == 'safe' else 0)
    else:
        cwe_to_idx = {cwe: i + 1 for i, cwe in enumerate(TOP_CWES)}
        
        def get_multiclass_label(row):
            if str(row['safety']).lower() == 'safe':
                return 0
            cwe = str(row.get('cwe_id', '')).strip()
            return cwe_to_idx.get(cwe, len(TOP_CWES) + 1)
        
        df['label'] = df.apply(get_multiclass_label, axis=1)
    
    # Split
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'],
        random_state=42
    )
    
    # Crear datasets
    train_dataset = HybridCodeDataset(
        codes=train_df['code'].tolist(),
        labels=train_df['label'].tolist(),
        languages=train_df.get('programming_language', pd.Series([None] * len(train_df))).tolist(),
        tokenizer=tokenizer
    )
    
    val_dataset = HybridCodeDataset(
        codes=val_df['code'].tolist(),
        labels=val_df['label'].tolist(),
        languages=val_df.get('programming_language', pd.Series([None] * len(val_df))).tolist(),
        tokenizer=tokenizer
    )
    
    return train_dataset, val_dataset
