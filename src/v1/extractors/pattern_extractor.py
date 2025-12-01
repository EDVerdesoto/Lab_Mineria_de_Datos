"""
Extractor de patrones de vulnerabilidad basado en regex.
Incluye localización de líneas vulnerables y caché para rendimiento.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional

from .base import BaseExtractor
from config import MAX_CODE_LENGTH


class PatternExtractor(BaseExtractor):
    """
    Extrae características basadas en patrones de vulnerabilidad.
    También puede localizar las líneas exactas donde se detectan.
    
    Optimizaciones:
    - Patrones pre-compilados
    - Caché LRU para códigos repetidos
    - Early exit para código vacío/corto
    """
    
    # Patrones optimizados (non-greedy para evitar backtracking)
    DEFAULT_PATTERNS: Dict[str, List[str]] = {
        'buffer_overflow': [
            r'\bstrcpy\s*\(', r'\bstrcat\s*\(', r'\bsprintf\s*\(',
            r'\bgets\s*\(', r'\bmemcpy\s*\(', r'\bscanf\s*\(',
            r'\bvsprintf\s*\(', r'\bsscanf\s*\('
        ],
        'sql_injection': [
            r'SELECT\s+.+?\s+FROM', r'INSERT\s+INTO', r'DELETE\s+FROM',
            r'UPDATE\s+.+?\s+SET', r'DROP\s+TABLE',
            r'"\s*\+\s*\w+', r"'\s*\+\s*\w+",
            r'f["\'][^"\']{0,200}SELECT', r'\.format\s*\([^)]{0,200}SELECT'
        ],
        'xss': [
            r'\.innerHTML\s*=', r'document\.write\s*\(',
            r'document\.writeln\s*\(', r'<script>',
            r'\$_GET\s*\[', r'\$_POST\s*\[', r'\$_REQUEST\s*\[',
            r'\.outerHTML\s*='
        ],
        'command_injection': [
            r'\bsystem\s*\(', r'\bpopen\s*\(', r'\bexecl?\s*\(',
            r'\bexecv\s*\(', r'\bshell_exec\s*\(', r'\bpassthru\s*\(',
            r'\bproc_open\s*\(', r'Runtime\.getRuntime\(\)\.exec'
        ],
        'dangerous_functions': [
            r'\beval\s*\(', r'\bexec\s*\(', r'\bcompile\s*\(',
            r'\bpickle\.loads?\s*\(', r'\byaml\.load\s*\(',
            r'\byaml\.unsafe_load\s*\(', r'\bmarshal\.loads?\s*\(',
            r'\b__import__\s*\('
        ]
    }
    
    # Mapeo de categorías a CWEs
    CATEGORY_TO_CWE: Dict[str, str] = {
        'buffer_overflow': 'CWE-119',
        'sql_injection': 'CWE-89',
        'xss': 'CWE-79',
        'command_injection': 'CWE-78',
        'dangerous_functions': 'CWE-94'
    }
    
    def __init__(self, patterns: Dict[str, List[str]] = None, cache_size: int = 10000):
        self.patterns = patterns if patterns is not None else self.DEFAULT_PATTERNS.copy()
        self._compile_patterns()
        self._update_feature_names()
        
        # Caché
        self._cache_size = cache_size
        self._cache: Dict[str, Tuple[float, ...]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _compile_patterns(self) -> None:
        """Pre-compila patrones regex para mejor rendimiento."""
        self._compiled = {}
        # Combinar patrones de cada categoría en una sola regex con grupos
        self._combined = {}
        
        for category, pattern_list in self.patterns.items():
            self._compiled[category] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
            # Regex combinada para conteo rápido
            combined_pattern = '|'.join(f'(?:{p})' for p in pattern_list)
            self._combined[category] = re.compile(combined_pattern, re.IGNORECASE)
    
    def _update_feature_names(self) -> None:
        """Actualiza nombres de features según patrones activos."""
        self._feature_names = [f'{cat}_count' for cat in self.patterns.keys()]
    
    def _compute_hash(self, code: str) -> str:
        """Genera hash rápido para el caché."""
        key = f"{len(code)}:{code[:100]}:{code[-100:] if len(code) > 100 else ''}"
        return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
    
    def extract(self, code: str, language: str = None) -> List[float]:
        """
        Cuenta ocurrencias de patrones de vulnerabilidad.
        
        Returns:
            Lista con conteo por cada categoría de patrón.
        """
        code_str = str(code) if code else ""
        
        # Early exit para código vacío o muy corto
        if len(code_str) < 5:
            return [0.0] * len(self.patterns)
        
        # Limitar tamaño para evitar regex catastrophic backtracking
        if len(code_str) > MAX_CODE_LENGTH:
            code_str = code_str[:MAX_CODE_LENGTH]
        
        # Verificar caché
        cache_key = self._compute_hash(code_str)
        if cache_key in self._cache:
            self._cache_hits += 1
            return list(self._cache[cache_key])
        
        self._cache_misses += 1
        
        # Usar regex combinada para conteo más rápido
        counts = []
        for category in self.patterns.keys():
            matches = self._combined[category].findall(code_str)
            counts.append(float(len(matches)))
        
        # Guardar en caché
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = tuple(counts)
        
        return counts
    
    def find_matches(self, code: str, category: str = None) -> Dict[str, List[dict]]:
        """
        Encuentra todas las coincidencias con información detallada.
        Útil para explicar predicciones y localizar vulnerabilidades.
        
        Args:
            code: Código a analizar
            category: Categoría específica (opcional, None = todas)
        
        Returns:
            Diccionario con matches por categoría, incluyendo línea y columna
        """
        results = {}
        code_str = str(code)
        lines = code_str.split('\n')
        
        categories_to_check = [category] if category else self.patterns.keys()
        
        for cat in categories_to_check:
            if cat not in self._compiled:
                continue
                
            results[cat] = []
            for line_num, line in enumerate(lines, 1):
                for pattern_idx, compiled_pattern in enumerate(self._compiled[cat]):
                    for match in compiled_pattern.finditer(line):
                        results[cat].append({
                            'line': line_num,
                            'column': match.start(),
                            'content': line.strip(),
                            'match': match.group(),
                            'pattern': self.patterns[cat][pattern_idx],
                            'cwe': self.CATEGORY_TO_CWE.get(cat, 'Unknown')
                        })
        
        return results
    
    def get_vulnerability_summary(self, code: str) -> Dict:
        """
        Genera un resumen de vulnerabilidades detectadas.
        
        Returns:
            Diccionario con resumen de vulnerabilidades
        """
        matches = self.find_matches(code)
        
        summary = {
            'total_issues': 0,
            'by_category': {},
            'by_cwe': {},
            'lines_affected': set()
        }
        
        for category, category_matches in matches.items():
            if category_matches:
                summary['by_category'][category] = len(category_matches)
                summary['total_issues'] += len(category_matches)
                
                cwe = self.CATEGORY_TO_CWE.get(category, 'Unknown')
                summary['by_cwe'][cwe] = summary['by_cwe'].get(cwe, 0) + len(category_matches)
                
                for m in category_matches:
                    summary['lines_affected'].add(m['line'])
        
        summary['lines_affected'] = sorted(summary['lines_affected'])
        return summary
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
    
    @property
    def categories(self) -> List[str]:
        return list(self.patterns.keys())
    
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
