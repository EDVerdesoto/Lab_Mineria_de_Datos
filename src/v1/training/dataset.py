"""
Datasets para entrenamiento con streaming de archivos grandes.
"""

import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from typing import List, Optional, Iterator
from transformers import PreTrainedTokenizer

from extractors import ComplexityExtractor, PatternExtractor, ASTExtractor
from config import MAX_SEQ_LENGTH, CHUNK_SIZE, TOP_CWES


class HybridCodeDataset(Dataset):
    """
    Dataset para batches en memoria.
    Extrae features de CodeBERT + Lizard + Patterns + AST.
    """
    
    def __init__(
        self, 
        codes: List[str],
        labels: List[int],
        languages: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_SEQ_LENGTH
    ):
        self.codes = codes
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Inicializar extractores
        self.complexity_extractor = ComplexityExtractor()
        self.pattern_extractor = PatternExtractor()
        self.ast_extractor = ASTExtractor()
    
    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int) -> dict:
        code = str(self.codes[idx])
        lang = self.languages[idx] if self.languages else None
        
        # Tokenización para CodeBERT
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Features manuales
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
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        mode: str = 'binary',  # 'binary' o 'multiclass'
        chunk_size: int = CHUNK_SIZE,
        max_length: int = MAX_SEQ_LENGTH,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.skip_rows = skip_rows
        self.max_rows = max_rows
        
        # Extractores
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
    
    def __iter__(self) -> Iterator[dict]:
        rows_processed = 0
        
        reader = pd.read_csv(
            self.file_path, 
            chunksize=self.chunk_size,
            skiprows=range(1, self.skip_rows + 1) if self.skip_rows > 0 else None
        )
        
        for chunk in reader:
            chunk = chunk.dropna(subset=['code', 'safety'])
            
            for _, row in chunk.iterrows():
                if self.max_rows and rows_processed >= self.max_rows:
                    return
                
                code = str(row['code'])
                lang = row.get('programming_language', None)
                label = self._get_label(row)
                
                # Tokenización
                encoding = self.tokenizer(
                    code,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Features manuales
                complexity_feats = self.complexity_extractor.extract(code, lang)
                pattern_feats = self.pattern_extractor.extract(code, lang)
                ast_feats = self.ast_extractor.extract(code, lang)
                
                rows_processed += 1
                
                yield {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'complexity_features': torch.tensor(complexity_feats, dtype=torch.float32),
                    'pattern_features': torch.tensor(pattern_feats, dtype=torch.float32),
                    'ast_features': torch.tensor(ast_feats, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.long)
                }


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
