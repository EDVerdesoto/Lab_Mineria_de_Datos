import numpy as np
import lizard
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, csr_matrix
from config import HASHING_FEATURES
from concurrent.futures import ThreadPoolExecutor

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=HASHING_FEATURES, 
            alternate_sign=False,
            ngram_range=(1, 2)  # Captura patrones de 2 tokens
        )
        
        self.lang_extensions = {
            'python': '.py', 'java': '.java', 'cpp': '.cpp', 'c': '.c',
            'javascript': '.js', 'js': '.js', 'csharp': '.cs', 'c#': '.cs',
            'php': '.php', 'ruby': '.rb', 'swift': '.swift', 'go': '.go',
            'rust': '.rs', 'kotlin': '.kt', 'scala': '.scala', 
            'typescript': '.ts', 'perl': '.pl', 'other': '.txt'
        }

    def _get_lizard_metrics(self, code, language='c'):
        """Extrae métricas de ingeniería de software según el lenguaje."""
        try:
            lang_key = str(language).lower().strip() if language else 'other'
            extension = self.lang_extensions.get(lang_key, '.txt')
            
            analysis = lizard.analyze_file.analyze_source_code(
                f"stream{extension}", 
                str(code)
            )
            
            if not analysis.function_list:
                return [0, 0, 0, 0, len(str(code).split('\n'))]  # Agregar líneas totales
            
            funcs = analysis.function_list
            return [
                np.mean([f.nloc for f in funcs]),
                np.mean([f.cyclomatic_complexity for f in funcs]),
                np.mean([f.token_count for f in funcs]),
                np.mean([len(f.parameters) for f in funcs]),
                len(funcs)  # Número de funciones como feature adicional
            ]
        except Exception:
            return [0, 0, 0, 0, 0]

    def transform(self, raw_code_list, languages=None):
        """
        Convierte código en matriz híbrida con paralelización.
        """
        X_text = self.vectorizer.transform(raw_code_list)
        
        # Paralelizar extracción de métricas Lizard
        if languages is not None:
            pairs = list(zip(raw_code_list, languages))
            with ThreadPoolExecutor() as executor:
                lizard_data = list(executor.map(
                    lambda p: self._get_lizard_metrics(p[0], p[1]), 
                    pairs
                ))
        else:
            with ThreadPoolExecutor() as executor:
                lizard_data = list(executor.map(self._get_lizard_metrics, raw_code_list))
        
        X_lizard = np.array(lizard_data)
        return hstack([X_text, csr_matrix(X_lizard)]), X_lizard