"""
Módulo de inferencia.
"""

from .predictor import VulnerabilityPredictor, load_predictor

__all__ = [
    'VulnerabilityPredictor',
    'load_predictor'
]
