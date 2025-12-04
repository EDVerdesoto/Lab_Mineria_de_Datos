import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

# Importar tus módulos del proyecto
from .model import VulnClassifier
from .map_cwe import LABEL_NAMES, NUM_LABELS

class VulnerabilityPredictor:
    def __init__(self, model_path, model_name="microsoft/codebert-base", device=None):
        """
        Inicializa el predictor cargando el modelo y el tokenizador.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")

        # 1. Cargar Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2. Configurar Modelo (Debe coincidir con la config de entrenamiento)
        # Asumimos num_labels=8 y aggregation='max' (tu configuración por defecto)
        self.model = VulnClassifier(
            model_name=model_name,
            num_labels=NUM_LABELS,
            aggregation='max' 
        )

        # 3. Cargar Pesos Entrenados
        print(f"Cargando pesos desde {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval() # Modo evaluación (desactiva dropout)

        # Parámetros de Sliding Window (Iguales al entrenamiento)
        self.max_len = 512
        self.stride = 256
        self.max_windows = 8

    def _create_sliding_windows(self, code):
        """
        Replica la lógica de sliding window del dataset para una sola muestra.
        """
        tokens = self.tokenizer.encode(code, add_special_tokens=False, truncation=False)
        
        effective_len = self.max_len - 2
        windows = []

        # Si el código es corto
        if len(tokens) <= effective_len:
            windows.append(self._build_window(tokens))
            return windows

        # Si es largo, crear ventanas
        start = 0
        while start < len(tokens) and len(windows) < self.max_windows:
            end = min(start + effective_len, len(tokens))
            window_tokens = tokens[start:end]
            windows.append(self._build_window(window_tokens))
            
            start += self.stride
            if len(tokens) - start < effective_len // 4:
                break
        
        return windows

    def _build_window(self, tokens):
        """Construye el tensor con [CLS], [SEP] y Padding."""
        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        
        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def predict(self, code_snippet):
        """
        Realiza la predicción para un string de código.
        """
        # 1. Pre-procesamiento (Sliding Window)
        windows = self._create_sliding_windows(code_snippet)
        
        # 2. Preparar Batch
        # Stackear ventanas
        input_ids = torch.stack([w['input_ids'] for w in windows]).to(self.device)
        attention_mask = torch.stack([w['attention_mask'] for w in windows]).to(self.device)
        
        # Window counts (necesario para tu modelo)
        window_counts = torch.tensor([len(windows)], dtype=torch.long).to(self.device)

        # 3. Inferencia
        with torch.no_grad():
            # Pasamos todo al modelo
            logits = self.model(input_ids, attention_mask, window_counts)
            
            # Aplicar Softmax para obtener probabilidades
            probs = F.softmax(logits, dim=1)
            
            # Obtener la clase ganadora
            conf, pred_id = torch.max(probs, dim=1)

        # 4. Formatear resultado
        pred_idx = pred_id.item()
        result = {
            "label": LABEL_NAMES[pred_idx],
            "confidence": conf.item(),
            "probabilities": {name: p.item() for name, p in zip(LABEL_NAMES, probs[0])}
        }
        return result