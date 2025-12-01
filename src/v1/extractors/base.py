"""
Interfaz base para extractores de características.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseExtractor(ABC):
    """
    Clase base abstracta para todos los extractores de características.
    """
    
    @abstractmethod
    def extract(self, code: str, language: str = None) -> List[float]:
        """
        Extrae características de un fragmento de código.
        
        Args:
            code: String con el código fuente
            language: Lenguaje de programación (opcional)
        
        Returns:
            Lista de valores numéricos (features)
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Retorna los nombres de las características extraídas."""
        pass
    
    @property
    def num_features(self) -> int:
        """Número de características que genera este extractor."""
        return len(self.feature_names)
    
    def extract_batch(self, codes: List[str], languages: List[str] = None) -> np.ndarray:
        """
        Extrae características de múltiples fragmentos de código.
        """
        if languages is None:
            languages = [None] * len(codes)
        
        features = [self.extract(code, lang) for code, lang in zip(codes, languages)]
        return np.array(features)
