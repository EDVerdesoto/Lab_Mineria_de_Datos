"""
Trainer optimizado para datasets grandes.
Incluye: gradient accumulation, mixed precision (FP16), checkpointing.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, List
import json
from datetime import datetime

from models import HybridCodeClassifier
from config import (
    DEVICE, CODEBERT_MODEL, CHECKPOINT_DIR,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, 
    ACCUMULATION_STEPS, WARMUP_RATIO,
    CLASS_NAMES_BINARY, CLASS_NAMES_MULTICLASS
)


class HybridTrainer:
    """
    Trainer para el modelo híbrido CodeBERT + Features.
    
    Características:
    - Gradient accumulation para batches grandes virtuales
    - Mixed precision (FP16) para acelerar en GPU
    - Checkpointing automático
    - Early stopping
    - Class weights para desbalance
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        accumulation_steps: int = ACCUMULATION_STEPS,
        checkpoint_dir: str = CHECKPOINT_DIR,
        class_weights: Optional[torch.Tensor] = None,
        freeze_codebert_epochs: int = 0
    ):
        self.device = torch.device(DEVICE)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.freeze_codebert_epochs = freeze_codebert_epochs
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
        
        # Modelo
        self.model = HybridCodeClassifier(
            num_classes=num_classes,
            codebert_model=CODEBERT_MODEL,
            freeze_codebert=(freeze_codebert_epochs > 0)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Loss con class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler (se inicializa en train())
        self.scheduler = None
        
        # Métricas
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': []
        }
        
        # Mejor modelo
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # Class names
        self.class_names = CLASS_NAMES_BINARY if num_classes == 2 else CLASS_NAMES_MULTICLASS[:num_classes]
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"[INFO] Trainer inicializado")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Clases: {num_classes}")
        print(f"[INFO] Batch size efectivo: {batch_size * accumulation_steps}")
    
    def _unfreeze_codebert(self):
        """Descongela los parámetros de CodeBERT."""
        for param in self.model.codebert.parameters():
            param.requires_grad = True
        print("[INFO] CodeBERT descongelado")
    
    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Calcula pesos de clase para manejar desbalance."""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        
        for step, batch in enumerate(pbar):
            # Mover a device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            complexity = batch['complexity_features'].to(self.device)
            patterns = batch['pattern_features'].to(self.device)
            ast_feats = batch['ast_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward con mixed precision
            with autocast(device_type=self.device.type):
                logits = self.model(
                    input_ids, attention_mask,
                    complexity, patterns, ast_feats
                )
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict:
        """Evalúa el modelo en el conjunto de validación."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            complexity = batch['complexity_features'].to(self.device)
            patterns = batch['pattern_features'].to(self.device)
            ast_feats = batch['ast_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with autocast(device_type=self.device.type):
                logits = self.model(
                    input_ids, attention_mask,
                    complexity, patterns, ast_feats
                )
                loss = self.criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1
        
        # Métricas
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
        early_stopping_patience: int = 3
    ) -> Dict:
        """
        Entrena el modelo completo.
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número de épocas
            early_stopping_patience: Épocas sin mejora antes de parar
        
        Returns:
            Diccionario con historial de entrenamiento
        """
        # Scheduler con warmup
        total_steps = len(train_loader) * epochs // self.accumulation_steps
        warmup_steps = int(total_steps * WARMUP_RATIO)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"     ENTRENAMIENTO - {epochs} épocas")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_loader.dataset):,}")
        print(f"Val samples: {len(val_loader.dataset):,}")
        print(f"Total steps: {total_steps:,}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Descongelar CodeBERT después de N épocas
            if epoch == self.freeze_codebert_epochs and self.freeze_codebert_epochs > 0:
                self._unfreeze_codebert()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Guardar mejor modelo
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ Nuevo mejor modelo guardado (F1: {self.best_f1:.4f})")
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n[INFO] Early stopping en época {epoch + 1}")
                break
        
        # Resumen final
        print(f"\n{'='*60}")
        print(f"     ENTRENAMIENTO FINALIZADO")
        print(f"{'='*60}")
        print(f"Mejor F1: {self.best_f1:.4f} (época {self.best_epoch})")
        print(f"{'='*60}\n")
        
        # Classification report del mejor modelo
        self.load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))
        final_metrics = self.evaluate(val_loader)
        
        print("\nReporte de Clasificación:")
        print(classification_report(
            final_metrics['labels'],
            final_metrics['predictions'],
            target_names=self.class_names[:len(set(final_metrics['labels']))],
            zero_division=0
        ))
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Guarda checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'num_classes': self.num_classes,
            'best_f1': self.best_f1
        }
        
        # Guardar checkpoint regular
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(checkpoint, path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # Guardar historial como JSON
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint del modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f"[INFO] Checkpoint cargado: {path}")
        return checkpoint['epoch']


def train_from_csv(
    csv_path: str,
    mode: str = 'binary',
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_samples: Optional[int] = None
):
    """
    Función de conveniencia para entrenar desde un CSV.
    
    Args:
        csv_path: Ruta al archivo CSV
        mode: 'binary' o 'multiclass'
        epochs: Número de épocas
        batch_size: Tamaño de batch
        max_samples: Máximo de muestras (para testing)
    """
    import pandas as pd
    from training.dataset import create_datasets_from_dataframe
    
    print(f"[INFO] Cargando dataset: {csv_path}")
    
    # Cargar datos
    if max_samples:
        df = pd.read_csv(csv_path, nrows=max_samples)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"[INFO] Muestras cargadas: {len(df):,}")
    
    # Determinar número de clases
    num_classes = 2 if mode == 'binary' else len(CLASS_NAMES_MULTICLASS)
    
    # Inicializar trainer
    trainer = HybridTrainer(num_classes=num_classes)
    
    # Crear datasets
    train_dataset, val_dataset = create_datasets_from_dataframe(
        df, 
        trainer.tokenizer,
        mode=mode
    )
    
    # Calcular class weights
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_weights = trainer.compute_class_weights(train_labels)
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights.to(trainer.device))
    
    print(f"[INFO] Class weights: {class_weights}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    
    # Entrenar
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    return trainer, history
