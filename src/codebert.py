import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import MODEL_PATH, NUM_CLASSES

# Ruta del modelo CodeBERT
CODEBERT_PATH = MODEL_PATH.replace('.json', '_codebert.pt')

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.codes = list(codes)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.codes[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


class CodeBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('microsoft/codebert-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(self.dropout(pooled))


class CodeBERTTrainer:
    def __init__(self, num_classes=NUM_CLASSES, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = CodeBERTClassifier(num_classes).to(self.device)
        self.num_classes = num_classes
        
        # Cargar modelo existente si existe (entrenamiento incremental)
        if os.path.exists(CODEBERT_PATH):
            self.load(CODEBERT_PATH)
            print(f"[INFO] Modelo CodeBERT cargado desde {CODEBERT_PATH}")
        else:
            print(f"[INFO] Nuevo modelo CodeBERT creado")
            os.makedirs(os.path.dirname(CODEBERT_PATH), exist_ok=True)
        
        print(f"[INFO] Dispositivo: {self.device}")
        print(f"[INFO] Clases: {num_classes}")
    
    def train_on_chunk(self, chunk, epochs=3, batch_size=8, lr=2e-5, test_size=0.2):
        """
        Entrena incrementalmente sobre un chunk del dataset.
        Compatible con el pipeline de data_loader.get_data_chunks()
        """
        from sklearn.model_selection import train_test_split
        
        # Preparar datos
        chunk = chunk.dropna(subset=['code', 'target'])
        if chunk.empty:
            return None
        
        codes = chunk['code'].astype(str).tolist()
        labels = chunk['target'].astype(int).tolist()
        
        # Split estratificado
        unique_classes = len(set(labels))
        stratify = labels if unique_classes > 1 else None
        
        train_codes, val_codes, train_labels, val_labels = train_test_split(
            codes, labels, test_size=test_size, stratify=stratify
        )
        
        train_dataset = CodeDataset(train_codes, train_labels, self.tokenizer)
        val_dataset = CodeDataset(val_codes, val_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            f1 = self._evaluate(val_loader)
            
            if f1 > best_f1:
                best_f1 = f1
                self.save(CODEBERT_PATH)
        
        return best_f1
    
    def _evaluate(self, data_loader):
        """Evalúa el modelo y retorna F1-score."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].numpy())
        
        return f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    def predict(self, codes, batch_size=16):
        """Predice clases para una lista de códigos."""
        self.model.eval()
        
        # Crear dataset sin labels (dummy labels)
        dummy_labels = [0] * len(codes)
        dataset = CodeDataset(codes, dummy_labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_preds, all_probs
    
    def save(self, path=CODEBERT_PATH):
        """Guarda el modelo completo."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"[INFO] Modelo guardado en {path}")
    
    def load(self, path=CODEBERT_PATH):
        """Carga el modelo desde archivo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)


def train_codebert():
    """
    Pipeline de entrenamiento incremental para CodeBERT.
    Comparable con train_pipeline.py de XGBoost.
    """
    import numpy as np
    from data_loader import get_data_chunks
    
    trainer = CodeBERTTrainer(num_classes=NUM_CLASSES)
    
    print("[INFO] Iniciando Pipeline de Entrenamiento CodeBERT")
    
    chunks = get_data_chunks()
    metrics_log = []
    total_samples = 0
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.dropna(subset=['code', 'target'])
        if chunk.empty:
            continue
        
        f1 = trainer.train_on_chunk(chunk, epochs=2, batch_size=8)
        
        if f1 is not None:
            metrics_log.append(f1)
            total_samples += len(chunk)
            print(f"[SUCCESS] Chunk {i+1} | Muestras: {total_samples:,} | F1: {f1:.4f}")
    
    if metrics_log:
        print(f"\n[INFO] Entrenamiento Finalizado.")
        print(f"[INFO] Total muestras procesadas: {total_samples:,}")
        print(f"[INFO] F1 Promedio: {np.mean(metrics_log):.4f} (±{np.std(metrics_log):.4f})")
        print(f"[INFO] Modelo guardado en: {CODEBERT_PATH}")
    else:
        print("[WARN] No se procesaron chunks.")


if __name__ == "__main__":
    train_codebert()