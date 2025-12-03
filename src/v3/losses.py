import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación desbalanceada.
    
    Args:
        alpha: Tensor de pesos por clase (None = sin pesos)
        gamma: Factor de enfoque (2.0 recomendado)
               - gamma=0: equivalente a CrossEntropy
               - gamma=2: enfoca en ejemplos difíciles
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()