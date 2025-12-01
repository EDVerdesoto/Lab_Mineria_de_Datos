"""
Clasificador híbrido: CodeBERT + Features manuales (Lizard + Patterns + AST)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridCodeClassifier(nn.Module):
    """
    Combina:
    - CodeBERT: representación semántica del código (768 dims)
    - Complexity: métricas de Lizard (5 dims)
    - Patterns: conteo de patrones vulnerables (5 dims)
    - AST: métricas de análisis estático (2 dims)
    
    Total: 768 + 5 + 5 + 2 = 780 features → Clasificador
    """
    
    def __init__(
        self, 
        num_classes: int,
        codebert_model: str = 'microsoft/codebert-base',
        freeze_codebert: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Dimensiones de cada componente
        self.codebert_dim = 768
        self.complexity_dim = 5
        self.pattern_dim = 5
        self.ast_dim = 2
        
        # CodeBERT encoder
        self.codebert = AutoModel.from_pretrained(codebert_model)
        
        if freeze_codebert:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        # Proyección de features manuales a espacio más rico
        self.complexity_proj = nn.Sequential(
            nn.Linear(self.complexity_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pattern_proj = nn.Sequential(
            nn.Linear(self.pattern_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ast_proj = nn.Sequential(
            nn.Linear(self.ast_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dimensión después de proyección: 768 + 32 + 32 + 16 = 848
        fusion_dim = self.codebert_dim + 32 + 32 + 16
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        complexity_features: torch.Tensor,
        pattern_features: torch.Tensor,
        ast_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass del modelo híbrido.
        
        Args:
            input_ids: Tokens del código [batch_size, seq_len]
            attention_mask: Máscara de atención [batch_size, seq_len]
            complexity_features: Features de Lizard [batch_size, 5]
            pattern_features: Features de patrones [batch_size, 5]
            ast_features: Features de AST [batch_size, 2]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # CodeBERT encoding - usar [CLS] token
        codebert_output = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = codebert_output.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Proyectar features manuales
        complexity_proj = self.complexity_proj(complexity_features)  # [batch, 32]
        pattern_proj = self.pattern_proj(pattern_features)  # [batch, 32]
        ast_proj = self.ast_proj(ast_features)  # [batch, 16]
        
        # Fusionar todas las representaciones
        fused = torch.cat([
            cls_embedding,
            complexity_proj,
            pattern_proj,
            ast_proj
        ], dim=1)  # [batch, 848]
        
        # Clasificación
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        complexity_features: torch.Tensor,
        pattern_features: torch.Tensor,
        ast_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Obtiene embeddings fusionados (útil para análisis/visualización).
        """
        codebert_output = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = codebert_output.last_hidden_state[:, 0, :]
        
        complexity_proj = self.complexity_proj(complexity_features)
        pattern_proj = self.pattern_proj(pattern_features)
        ast_proj = self.ast_proj(ast_features)
        
        fused = torch.cat([
            cls_embedding,
            complexity_proj,
            pattern_proj,
            ast_proj
        ], dim=1)
        
        return fused


class BinaryClassifier(HybridCodeClassifier):
    """Clasificador binario: Safe vs Vulnerable"""
    
    def __init__(self, **kwargs):
        super().__init__(num_classes=2, **kwargs)


class MultiClassClassifier(HybridCodeClassifier):
    """Clasificador multi-clase: Safe + CWEs específicos + Other"""
    
    def __init__(self, num_cwe_classes: int = 10, **kwargs):
        # Safe + N CWEs + Other
        num_classes = 1 + num_cwe_classes + 1
        super().__init__(num_classes=num_classes, **kwargs)
