"""
Extractor de métricas de complejidad usando Lizard.
"""

from typing import List, Dict
import numpy as np
import lizard

from .base import BaseExtractor
from config import MAX_CODE_LENGTH


class ComplexityExtractor(BaseExtractor):
    """
    Extrae métricas de complejidad de código usando Lizard.
    
    Métricas:
    - NLOC: Líneas de código lógicas
    - Complejidad ciclomática: Caminos independientes
    - Token count: Número de tokens
    - Parámetros: Promedio de parámetros por función
    - Número de funciones
    """
    
    LANG_EXTENSIONS: Dict[str, str] = {
        'python': '.py', 'py': '.py', 'python3': '.py',
        'java': '.java',
        'cpp': '.cpp', 'c++': '.cpp',
        'c': '.c',
        'javascript': '.js', 'js': '.js',
        'csharp': '.cs', 'c#': '.cs',
        'php': '.php',
        'ruby': '.rb',
        'swift': '.swift',
        'go': '.go',
        'rust': '.rs',
        'kotlin': '.kt',
        'scala': '.scala',
        'typescript': '.ts', 'ts': '.ts',
    }
    
    def __init__(self):
        self._feature_names = [
            'nloc',
            'cyclomatic_complexity', 
            'token_count',
            'avg_params',
            'num_functions'
        ]
    
    def _get_extension(self, language: str) -> str:
        """Obtiene extensión de archivo según el lenguaje."""
        if not language:
            return '.txt'
        lang_key = str(language).lower().strip()
        return self.LANG_EXTENSIONS.get(lang_key, '.txt')
    
    def extract(self, code: str, language: str = None) -> List[float]:
        """Extrae métricas de complejidad del código."""
        try:
            code_str = str(code)
            
            # Limitar tamaño para evitar que Lizard se cuelgue
            if len(code_str) > MAX_CODE_LENGTH:
                code_str = code_str[:MAX_CODE_LENGTH]
            
            extension = self._get_extension(language)
            analysis = lizard.analyze_file.analyze_source_code(
                f"stream{extension}",
                code_str
            )
            
            if not analysis.function_list:
                lines = code_str.split('\n') if code_str else []
                tokens = code_str.split() if code_str else []
                return [
                    float(len(lines)),
                    1.0,
                    float(len(tokens)),
                    0.0,
                    0.0
                ]
            
            funcs = analysis.function_list
            return [
                float(np.mean([f.nloc for f in funcs])),
                float(np.mean([f.cyclomatic_complexity for f in funcs])),
                float(np.mean([f.token_count for f in funcs])),
                float(np.mean([len(f.parameters) for f in funcs])),
                float(len(funcs))
            ]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
