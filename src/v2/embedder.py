import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class CodeEmbedder:
    def __init__(self, model_name="microsoft/codebert-base", batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- CodeBERT cargado en: {self.device} ---")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size

    def get_embeddings(self, code_list):
        """
        Recibe una lista de strings de código y devuelve una matriz numpy (N, 768).
        Procesa internamente en mini-batches para no saturar la VRAM.
        """
        all_embeddings = []
        
        # Modo evaluación para no calcular gradientes porque no será el clasificador
        self.model.eval()

        # Procesamos en lotes pequeños (ej. 64) dentro del chunk grande
        for i in range(0, len(code_list), self.batch_size):
            batch_texts = code_list[i : i + self.batch_size]
            
            # Tokenizar
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extraer [CLS] token (primer token) y mover a CPU
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
            # Limpieza opcional de caché CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Concatenar todos los mini-batches
        return np.vstack(all_embeddings)