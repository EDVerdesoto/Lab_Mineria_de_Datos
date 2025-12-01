"""
Extractor de métricas de complejidad usando Lizard.
Optimizado para procesamiento batch con caché LRU.
"""

from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import hashlib
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
    
    Optimizaciones:
    - Caché LRU para evitar re-análisis de código duplicado
    - Cálculos inline sin numpy para reducir overhead
    - Fallback rápido para código sin funciones
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
    
    # Extensiones soportadas por Lizard (evita análisis innecesarios)
    LIZARD_SUPPORTED = {'.py', '.java', '.cpp', '.c', '.js', '.cs', '.php', 
                        '.rb', '.swift', '.go', '.rs', '.kt', '.scala', '.ts', '.h'}
    
    def __init__(self, cache_size: int = 10000):
        """
        Args:
            cache_size: Tamaño máximo del caché LRU (default 10000 entradas)
        """
        self._feature_names = [
            'nloc',
            'cyclomatic_complexity', 
            'token_count',
            'avg_params',
            'num_functions'
        ]
        self._cache_size = cache_size
        self._cache: Dict[str, Tuple[float, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_extension(self, language: Optional[str]) -> str:
        """Obtiene extensión de archivo según el lenguaje."""
        if not language:
            return '.txt'
        lang_key = str(language).lower().strip()
        return self.LANG_EXTENSIONS.get(lang_key, '.txt')
    
    def _compute_hash(self, code: str, extension: str) -> str:
        """Genera hash rápido para el caché."""
        # Usar solo primeros y últimos caracteres + longitud para hash rápido
        key = f"{extension}:{len(code)}:{code[:100]}:{code[-100:] if len(code) > 100 else ''}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
    
    def _quick_fallback(self, code_str: str) -> List[float]:
        """Fallback rápido sin usar Lizard para código simple."""
        lines = code_str.count('\n') + 1
        # Estimación rápida de tokens (espacios + operadores)
        tokens = len(code_str.split())
        return [float(lines), 1.0, float(tokens), 0.0, 0.0]
    
    def _analyze_with_lizard(self, code_str: str, extension: str) -> List[float]:
        """Análisis con Lizard (sin caché)."""
        try:
            analysis = lizard.analyze_file.analyze_source_code(
                f"stream{extension}",
                code_str
            )
            
            funcs = analysis.function_list
            if not funcs:
                return self._quick_fallback(code_str)
            
            # Cálculo inline sin numpy para mejor rendimiento
            n = len(funcs)
            total_nloc = 0
            total_cc = 0
            total_tokens = 0
            total_params = 0
            
            for f in funcs:
                total_nloc += f.nloc
                total_cc += f.cyclomatic_complexity
                total_tokens += f.token_count
                total_params += len(f.parameters)
            
            return [
                total_nloc / n,
                total_cc / n,
                total_tokens / n,
                total_params / n,
                float(n)
            ]
            
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def extract(self, code: str, language: str = None) -> List[float]:
        """
        Extrae métricas de complejidad del código.
        
        Usa caché LRU para evitar re-análisis de código duplicado.
        """
        code_str = str(code) if code else ""
        
        # Skip rápido para código vacío o muy corto
        if len(code_str) < 10:
            return [0.0, 1.0, 0.0, 0.0, 0.0]
        
        # Limitar tamaño
        if len(code_str) > MAX_CODE_LENGTH:
            code_str = code_str[:MAX_CODE_LENGTH]
        
        extension = self._get_extension(language)
        
        # Skip para extensiones no soportadas por Lizard
        if extension not in self.LIZARD_SUPPORTED:
            return self._quick_fallback(code_str)
        
        # Verificar caché
        cache_key = self._compute_hash(code_str, extension)
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return list(self._cache[cache_key])
        
        self._cache_misses += 1
        
        # Analizar con Lizard
        result = self._analyze_with_lizard(code_str, extension)
        
        # Guardar en caché (LRU simple: eliminar entrada más antigua si lleno)
        if len(self._cache) >= self._cache_size:
            # Eliminar primera entrada (FIFO aproximado)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = tuple(result)
        
        return result
    
    def clear_cache(self):
        """Limpia el caché y reinicia contadores."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Retorna estadísticas del caché."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
