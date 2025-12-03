"""
model.py - Clasificador de vulnerabilidades con Sliding Window.

Arquitectura:
1. CodeBERT encoder extrae embeddings de cada ventana
2. Agregación combina múltiples ventanas en un solo vector
3. Clasificador predice la vulnerabilidad

Métodos de agregación:
- MAX: Toma el máximo por dimensión (mejor para recall)
- MEAN: Promedio de todas las ventanas (balanceado)
- ATTENTION: Pesos aprendidos por ventana (más sofisticado)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class VulnClassifier(nn.Module):
    """
    Clasificador de vulnerabilidades basado en CodeBERT.

    Soporta código largo mediante Sliding Window:
    - Cada ventana se procesa por CodeBERT
    - Las ventanas se agregan en un solo vector
    - El clasificador predice sobre el vector agregado
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 8,
        aggregation: str = 'max',
        dropout: float = 0.1
    ):
        super().__init__()
        self.aggregation = aggregation

        # Encoder (CodeBERT)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Clasificador
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Attention para agregación (si se usa)
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        window_counts: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (total_windows, seq_len) tokens de todas las ventanas
            attention_mask: (total_windows, seq_len) máscaras de atención
            window_counts: (batch_size,) número de ventanas por muestra

        Returns:
            logits: (batch_size, num_labels) predicciones
        """
        # Obtener embeddings del encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Usar CLS token como representación de cada ventana
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Si no hay window_counts, cada ventana es una muestra independiente
        if window_counts is None:
            embeddings = self.dropout(embeddings)
            return self.classifier(embeddings)

        # Agregar ventanas por muestra
        aggregated = self._aggregate_windows(embeddings, window_counts)

        # Clasificar
        aggregated = self.dropout(aggregated)
        return self.classifier(aggregated)
    
    def _aggregate_windows(
        self,
        embeddings: torch.Tensor,
        window_counts: torch.Tensor
    ) -> torch.Tensor:
        """Agrega embeddings de múltiples ventanas en uno por muestra."""
        results = []
        start_idx = 0

        for count in window_counts:
            count = count.item()
            sample_emb = embeddings[start_idx:start_idx + count]

            if self.aggregation == 'max':
                agg, _ = torch.max(sample_emb, dim=0)
            elif self.aggregation == 'mean':
                agg = torch.mean(sample_emb, dim=0)
            elif self.aggregation == 'attention':
                scores = self.attention(sample_emb)
                weights = F.softmax(scores, dim=0)
                agg = torch.sum(weights * sample_emb, dim=0)
            else:
                raise ValueError(f"Aggregation '{self.aggregation}' no soportada")

            results.append(agg)
            start_idx += count

        return torch.stack(results)