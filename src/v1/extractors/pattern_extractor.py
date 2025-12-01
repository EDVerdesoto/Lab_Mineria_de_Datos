"""
Extractor de patrones de vulnerabilidad basado en regex.
Incluye localización de líneas vulnerables.
"""

import re
from typing import List, Dict

from .base import BaseExtractor
from config import MAX_CODE_LENGTH


class PatternExtractor(BaseExtractor):
    """
    Extrae características basadas en patrones de vulnerabilidad.
    También puede localizar las líneas exactas donde se detectan.
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
    
    def __init__(self, patterns: Dict[str, List[str]] = None):
        self.patterns = patterns if patterns is not None else self.DEFAULT_PATTERNS.copy()
        self._compile_patterns()
        self._update_feature_names()
    
    def _compile_patterns(self) -> None:
        """Pre-compila patrones regex para mejor rendimiento."""
        self._compiled = {}
        for category, pattern_list in self.patterns.items():
            self._compiled[category] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
    
    def _update_feature_names(self) -> None:
        """Actualiza nombres de features según patrones activos."""
        self._feature_names = [f'{cat}_count' for cat in self.patterns.keys()]
    
    def extract(self, code: str, language: str = None) -> List[float]:
        """
        Cuenta ocurrencias de patrones de vulnerabilidad.
        
        Returns:
            Lista con conteo por cada categoría de patrón.
        """
        counts = []
        code_str = str(code)
        
        # Limitar tamaño para evitar regex catastrophic backtracking
        if len(code_str) > MAX_CODE_LENGTH:
            code_str = code_str[:MAX_CODE_LENGTH]
        
        for category in self.patterns.keys():
            count = 0
            for compiled_pattern in self._compiled[category]:
                matches = compiled_pattern.findall(code_str)
                count += len(matches)
            counts.append(float(count))
        
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
