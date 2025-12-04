import re
import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict
import warnings

from .vulnerability_indicators import (
    INDICATORS_SET,
    RESERVED_WORDS_SET,
)

class VulnDataset(Dataset):
    """
    Dataset para detección de vulnerabilidades optimizado.
    """
    
    def __init__(
        self,
        codes: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        training: bool = True,
        use_augmentation: bool = True,
        use_sliding_window: bool = True,
        augment_prob: float = 0.3,
        mask_prob: float = 0.10,
        max_len: int = 512,
        stride: int = 256,
        max_windows: int = 8 
    ):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.training = training
        self.use_augmentation = use_augmentation
        self.augment_prob = augment_prob
        self.use_sliding_window = use_sliding_window
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.stride = stride
        self.max_windows = max_windows
    
    def __len__(self) -> int:
        return len(self.codes)
    
    def __getitem__(self, idx: int) -> Dict:
        code = str(self.codes[idx])
        label = self.labels[idx]
        
        # 1. Data Augmentation
        if self.training and self.use_augmentation:
            if np.random.random() < self.augment_prob:
                code = self._augment(code)

        # 2. Tokenización
        if self.use_sliding_window:
            windows = self._create_sliding_windows(code)
        else:
            windows = [self._tokenize(code)]
        
        return {
            'windows': windows,
            'num_windows': len(windows),
            'label': label
        }
    
    # --- DATA AUGMENTATION (Sin cambios) ---
    def _augment(self, code: str) -> str:
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
        rename_map = {}
        if np.random.random() < 0.3:
            rename_map = self._create_rename_map(tokens)

        result = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in INDICATORS_SET:
                result.append(token)
            elif token_lower in RESERVED_WORDS_SET:
                result.append(token)
            elif len(token) == 1 and not token.isalnum():
                result.append(token)
            elif token in rename_map:
                result.append(rename_map[token])
            elif np.random.random() < self.mask_prob:
                result.append(self.tokenizer.mask_token)
            else:
                result.append(token)
        return ' '.join(result)

    def _create_rename_map(self, tokens: List[str]) -> Dict[str, str]:
        candidates = {}
        for token in tokens:
            token_lower = token.lower()
            if (len(token) >= 3 and
                token.replace('_', '').isalnum() and
                token_lower not in INDICATORS_SET and
                token_lower not in RESERVED_WORDS_SET):
                candidates[token] = candidates.get(token, 0) + 1
        sorted_vars = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
        rename_map = {}
        for i, (var_name, _) in enumerate(sorted_vars):
            if np.random.random() < 0.5:
                rename_map[var_name] = f"var_{i}"
        return rename_map
    

    # --- SLIDING WINDOW OPTIMIZADO ---
    def _create_sliding_windows(self, code: str) -> List[Dict[str, torch.Tensor]]:
        """
        Divide código en ventanas. 
        OPTIMIZACIÓN: Si el código es enorme, muestrea TEXTO antes de tokenizar.
        """
        # UMBRAL DE SEGURIDAD: 50,000 caracteres (~12k tokens)
        # Si pasamos esto, tokenizar todo de golpe es lento e ineficiente.
        if len(code) > 50000:
            return self._sample_windows_from_raw_text(code)

        # --- Lógica Estándar (Para archivos normales) ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
            tokens = self.tokenizer.encode(code, add_special_tokens=False, truncation=False)

        effective_len = self.max_len - 2

        if len(tokens) <= effective_len:
            return [self._tokenize(code)]

        # Si hay demasiados tokens (pero < 50k chars), usamos sampling de tokens
        if len(tokens) > effective_len * self.max_windows:
            return self._sample_windows_strategically(tokens, effective_len)

        # Sliding window normal
        windows = []
        start = 0
        while start < len(tokens) and len(windows) < self.max_windows:
            end = min(start + effective_len, len(tokens))
            window_tokens = tokens[start:end]
            windows.append(self._build_window(window_tokens))
            start += self.stride
            if len(tokens) - start < effective_len // 4:
                break
        return windows[:self.max_windows]

    def _sample_windows_from_raw_text(self, code: str) -> List[Dict[str, torch.Tensor]]:
        """
        ESTRATEGIA PARA ARCHIVOS GIGANTES:
        En lugar de tokenizar 500k caracteres, calculamos los offsets en el STRING,
        cortamos pedazos de texto y SOLO tokenizamos esos pedazos.
        """
        windows = []
        total_chars = len(code)
        # Aproximamos cuántos caracteres necesitamos para llenar 512 tokens.
        # 1 token ~= 4 chars promedio. Usamos 3000 chars para asegurar que llenamos la ventana.
        chunk_chars = 3000 
        
        # Generar posiciones distribuidas (Inicio, ..., Medio, ..., Fin)
        # positions = [0, 10000, 20000, ...]
        positions = np.linspace(0, total_chars - chunk_chars, self.max_windows, dtype=int)
        
        for start_char in positions:
            # Cortar el texto crudo
            end_char = min(start_char + chunk_chars, total_chars)
            text_chunk = code[start_char:end_char]
            
            # Tokenizar SOLO este pedazo
            tokens = self.tokenizer.encode(
                text_chunk, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_len - 2 # Aseguramos no pasarnos
            )
            
            windows.append(self._build_window(tokens))
            
        return windows

    def _sample_windows_strategically(self, tokens: List[int], effective_len: int) -> List[Dict[str, torch.Tensor]]:
        """Estrategia para archivos medianos (tokenizados completos)"""
        windows = []
        total_len = len(tokens)
        positions = np.linspace(0, total_len - effective_len, self.max_windows, dtype=int)

        for start in positions:
            end = min(start + effective_len, total_len)
            window_tokens = tokens[start:end]
            windows.append(self._build_window(window_tokens))
        return windows

    def _build_window(self, tokens: List[int]) -> Dict[str, torch.Tensor]:
        """Helper para construir el tensor final con padding"""
        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        
        # Truncar si por alguna razón nos pasamos
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len-1] + [self.tokenizer.sep_token_id]

        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def _tokenize(self, code: str) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer.encode_plus(
            code, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Collate function y CachedDataset se mantienen igual...
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []
    window_counts = []
    labels = []
    for item in batch:
        window_counts.append(item['num_windows'])
        labels.append(item['label'])
        for window in item['windows']:
            all_input_ids.append(window['input_ids'])
            all_attention_masks.append(window['attention_mask'])
    return {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_attention_masks),
        'window_counts': torch.tensor(window_counts, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.data = []
        print(f"Buscando partes en: {cache_dir}")
        
        if not os.path.isdir(cache_dir):
            if os.path.isfile(cache_dir):
                print(f"Aviso: Cargando archivo único {cache_dir}")
                self.data = torch.load(cache_dir, weights_only=False)
                return
            raise ValueError(f"'{cache_dir}' no es un directorio válido.")
            
        files = sorted(glob.glob(os.path.join(cache_dir, "part_*.pt")))
        if not files:
            raise ValueError(f"No se encontraron archivos .pt en {cache_dir}")
            
        print(f"Cargando {len(files)} partes en memoria RAM...")
        for f in files:
            part_data = torch.load(f, weights_only=False)
            self.data.extend(part_data)
            
        print(f"Dataset cargado. Total muestras: {len(self.data)}")
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)