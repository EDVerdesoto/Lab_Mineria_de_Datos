import re
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict

from src.v3.vulnerability_indicators import (
    INDICATORS_SET,
    RESERVED_WORDS_SET,
)


class VulnDataset(Dataset):
    """
    Dataset para detección de vulnerabilidades.
    
    Características:
    - Sliding Window: Divide código largo en ventanas solapadas
    - Data Augmentation: Token masking on-the-fly
    - Preservación: No modifica indicadores de vulnerabilidad
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
        max_windows: int = 8  # ⚠️ NUEVO: Límite de ventanas por muestra
    ):
        """
        Args:
            codes: Array de strings de código
            labels: Array de labels numéricos
            tokenizer: Tokenizer de HuggingFace
            training: Si True, aplica augmentation
            use_augmentation: Si True, aplica token masking y variable renaming
            use_sliding_window: Si True, divide código largo en ventanas
            augment_prob: Probabilidad de augmentar una muestra (0.3 = 30%)
            mask_prob: Probabilidad de enmascarar un token (0.10 = 10%)
            max_len: Longitud máxima de secuencia
            stride: Desplazamiento entre ventanas (overlap = max_len - stride)
            max_windows: Máximo de ventanas por muestra (evita OOM con código muy largo)
        """
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
        
        # 1. Data Augmentation (solo en training)
        if self.training and self.use_augmentation:
            if np.random.random() < self.augment_prob:
                code = self._augment(code)

        # 2. Tokenización (con o sin sliding window)
        if self.use_sliding_window:
            windows = self._create_sliding_windows(code)
        else:
            windows = [self._tokenize(code)]
        
        return {
            'windows': windows,
            'num_windows': len(windows),
            'label': label
        }
    
    # DATA AUGMENTATION
    def _augment(self, code: str) -> str:
        """
        Aplica data augmentation al código:

        1. Variable Renaming (30% probabilidad):
           - Renombra variables de usuario a nombres genéricos (var_1, var_2, ...)
           - NUNCA renombra indicadores de vulnerabilidad
           - NUNCA renombra palabras reservadas
           - Renombrado consistente en todo el código

        2. Token Masking:
           - Enmascara tokens aleatorios con <mask>
           - NUNCA enmascara indicadores de vulnerabilidad
           - NUNCA enmascara palabras reservadas
        """
        # Tokenizar a nivel de palabras
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)

        # PASO 1: Variable Renaming (conservador)
        # Solo aplicar 30% del tiempo para no ser agresivo
        rename_map = {}
        if np.random.random() < 0.3:
            rename_map = self._create_rename_map(tokens)

        # PASO 2: Aplicar transformaciones
        result = []
        for token in tokens:
            token_lower = token.lower()

            # Preservar indicadores de vulnerabilidad
            if token_lower in INDICATORS_SET:
                result.append(token)
            # Preservar palabras reservadas
            elif token_lower in RESERVED_WORDS_SET:
                result.append(token)
            # Preservar símbolos y operadores
            elif len(token) == 1 and not token.isalnum():
                result.append(token)
            # Aplicar renaming si existe
            elif token in rename_map:
                result.append(rename_map[token])
            # Enmascarar probabilísticamente
            elif np.random.random() < self.mask_prob:
                result.append(self.tokenizer.mask_token)
            else:
                result.append(token)

        return ' '.join(result)

    def _create_rename_map(self, tokens: List[str]) -> Dict[str, str]:
        """
        Crea un mapa de renombrado consistente para variables de usuario.

        Solo renombra:
        - Identificadores que parecen variables de usuario
        - Que tienen longitud >= 3
        - Que NO son indicadores ni palabras reservadas
        - Máximo 10 variables por muestra (las más frecuentes)
        """
        # Identificar candidatos a renombrar
        candidates = {}
        for token in tokens:
            token_lower = token.lower()
            # Solo considerar identificadores válidos
            if (len(token) >= 3 and
                token.replace('_', '').isalnum() and
                token_lower not in INDICATORS_SET and
                token_lower not in RESERVED_WORDS_SET):
                candidates[token] = candidates.get(token, 0) + 1

        # Ordenar por frecuencia y tomar las top 10
        sorted_vars = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]

        # Crear mapa de renombrado
        rename_map = {}
        for i, (var_name, _) in enumerate(sorted_vars):
            # 50% probabilidad de renombrar cada variable
            if np.random.random() < 0.5:
                rename_map[var_name] = f"var_{i}"

        return rename_map
    

    # SLIDING WINDOW
    def _create_sliding_windows(self, code: str) -> List[Dict[str, torch.Tensor]]:
        """
        Divide código largo en ventanas solapadas.

        Si el código cabe en una ventana, retorna solo una.
        Si es más largo, crea múltiples ventanas con overlap.
        ⚠️ LÍMITE: max_windows para evitar OOM con código extremadamente largo.
        """
        # Tokenizar sin truncar
        # Suprimir warning - es esperado para código largo, lo manejamos con sliding window
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length is longer")
            tokens = self.tokenizer.encode(
                code,
                add_special_tokens=False,
                truncation=False
            )

        # Longitud efectiva (restando CLS y SEP)
        effective_len = self.max_len - 2

        # Si cabe en una ventana, retornar directamente
        if len(tokens) <= effective_len:
            return [self._tokenize(code)]

        # ⚠️ PROTECCIÓN: Si el código es MUY largo, usar sampling estratégico
        if len(tokens) > effective_len * self.max_windows:
            # Código extremadamente largo: muestrear ventanas uniformemente
            return self._sample_windows_strategically(tokens, effective_len)

        # Crear ventanas solapadas (normal)
        windows = []
        start = 0

        while start < len(tokens) and len(windows) < self.max_windows:
            end = min(start + effective_len, len(tokens))
            window_tokens = tokens[start:end]

            # Construir input con tokens especiales
            input_ids = (
                [self.tokenizer.cls_token_id] +
                window_tokens +
                [self.tokenizer.sep_token_id]
            )

            # Padding
            padding_len = self.max_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len

            windows.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            })

            # Avanzar por stride
            start += self.stride

            # Evitar ventana final muy pequeña
            if len(tokens) - start < effective_len // 4:
                break

        return windows[:self.max_windows]  # Asegurar límite

    def _sample_windows_strategically(
        self,
        tokens: List[int],
        effective_len: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Para código MUY largo (>max_windows ventanas), muestrea estratégicamente:
        - Inicio (primeras líneas suelen tener imports/setup)
        - Medio (lógica principal)
        - Final (return/cleanup)
        """
        windows = []
        total_len = len(tokens)

        # Distribuir ventanas uniformemente
        positions = np.linspace(0, total_len - effective_len, self.max_windows, dtype=int)

        for start in positions:
            end = min(start + effective_len, total_len)
            window_tokens = tokens[start:end]

            # Construir input con tokens especiales
            input_ids = (
                [self.tokenizer.cls_token_id] +
                window_tokens +
                [self.tokenizer.sep_token_id]
            )

            # Padding
            padding_len = self.max_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len

            windows.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            })

        return windows
    
    def _tokenize(self, code: str) -> Dict[str, torch.Tensor]:
        """Tokeniza código corto (sin sliding window)."""
        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function para el DataLoader.
    
    Agrupa todas las ventanas de todos los samples en tensores,
    y mantiene track de cuántas ventanas tiene cada sample.
    """
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
    def __init__(self, pt_file):
        self.data = torch.load(pt_file)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)