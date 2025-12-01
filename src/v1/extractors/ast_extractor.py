"""
Extractor de métricas basado en AST (Abstract Syntax Tree).
Solo funciona con código Python válido.
"""

import ast
from typing import List, Set

from .base import BaseExtractor


class SecurityVisitor(ast.NodeVisitor):
    """
    Visitante AST para extraer métricas de seguridad en código Python.
    """
    
    DANGEROUS_FUNCTIONS: Set[str] = {
        'eval', 'exec', 'compile', 'execfile',
        'system', 'popen', 'spawn',
        'input',  # Python 2
        '__import__'
    }
    
    DANGEROUS_ATTRIBUTES: Set[str] = {
        'system', 'popen', 'shell', 'subprocess',
        'call', 'run', 'Popen'
    }
    
    NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
    
    def __init__(self):
        self.max_depth: int = 0
        self.current_depth: int = 0
        self.dangerous_count: int = 0
    
    def generic_visit(self, node: ast.AST) -> None:
        """Calcula profundidad de anidamiento."""
        if isinstance(node, self.NESTING_NODES):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            super().generic_visit(node)
            self.current_depth -= 1
        else:
            super().generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Detecta llamadas a funciones peligrosas."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_FUNCTIONS:
                self.dangerous_count += 1
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.DANGEROUS_ATTRIBUTES:
                self.dangerous_count += 1
        
        self.generic_visit(node)


class ASTExtractor(BaseExtractor):
    """
    Extrae métricas de seguridad usando análisis AST.
    Solo funciona con código Python válido.
    """
    
    PYTHON_LANGUAGES = {'python', 'py', 'python3', 'python2'}
    
    def __init__(self):
        self._feature_names = [
            'nesting_depth',
            'dangerous_func_count'
        ]
    
    def _is_python(self, language: str) -> bool:
        """Verifica si el lenguaje es Python."""
        if not language:
            return False
        return language.lower().strip() in self.PYTHON_LANGUAGES
    
    def extract(self, code: str, language: str = None) -> List[float]:
        """Extrae métricas AST del código."""
        if not self._is_python(language):
            return [0.0, 0.0]
        
        try:
            tree = ast.parse(str(code))
            visitor = SecurityVisitor()
            visitor.visit(tree)
            return [
                float(visitor.max_depth),
                float(visitor.dangerous_count)
            ]
        except (SyntaxError, ValueError, TypeError):
            return [0.0, 0.0]
    
    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
