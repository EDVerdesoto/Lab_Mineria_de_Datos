import sys
import re
import lizard
from typing import Dict, Any

# --- CONFIGURACIÓN DE EXTENSIONES Y LENGUAJES ---
# Mapeo para ayudar a Lizard/Fallback a entender qué buscar
LANG_MAPPING = {
    'python': 'python', 'py': 'python',
    'c': 'c', 'cpp': 'cpp', 'c++': 'cpp', 'h': 'cpp', 'hpp': 'cpp',
    'java': 'java',
    'javascript': 'javascript', 'js': 'javascript', 'typescript': 'typescript', 'ts': 'typescript',
    'php': 'php',
    'ruby': 'ruby',
    'go': 'go',
    'c#': 'csharp', 'cs': 'csharp',
    'scala': 'scala',
    'swift': 'swift',
    'lua': 'lua'
}

def clean_code_for_parsing(code: str, lang: str) -> str:
    """Limpieza básica para ayudar al parser"""
    code = str(code)
    # Fix PHP: Lizard necesita etiquetas
    if 'php' in lang and '<?' not in code:
        code = "<?php\n" + code + "\n?>"
    return code

def calculate_fallback_metrics(code: str) -> Dict[str, Any]:
    """
    NIVEL 2: Extracción Heurística (A prueba de balas).
    Se usa cuando el parser principal falla. Es una aproximación.
    """
    metrics = {
        "nloc": 0, "complexity": 0, "token_count": 0, 
        "top_nesting_level": 0, "parameters_count": 0,
        "extraction_method": "fallback" # Para que sepas que fue aproximado
    }
    
    try:
        lines = code.splitlines()
        # 1. NLOC: Líneas que no están vacías
        # (Simplificación: no filtramos comentarios multilínea complejos para ser rápidos)
        metrics["nloc"] = sum(1 for line in lines if line.strip())

        # 2. Token Count: Aproximación por palabras
        # Dividimos por cualquier cosa que no sea alfanumérica
        tokens = re.findall(r'\w+|[^\w\s]', code)
        metrics["token_count"] = len(tokens)

        # 3. Complejidad Ciclomática Estimada
        # Contamos palabras clave de control de flujo comunes en C/Java/Py/JS
        # Palabras: if, else, for, while, case, catch, ?, &&, ||
        control_flow_pattern = r'\b(if|for|while|case|catch|elif)\b|(\?)|(&&)|(\|\|)'
        complexity_points = len(re.findall(control_flow_pattern, code))
        metrics["complexity"] = complexity_points + 1 # Base es 1

        # 4. Parameters Count Estimado
        # Buscamos definiciones de funciones y contamos comas
        # Heurística muy básica: contar paréntesis de apertura seguidos de comas
        # Esto es muy difícil de hacer bien con regex, devolvemos un promedio seguro o 0
        metrics["parameters_count"] = 0 

        # 5. Nesting
        # Difícil con regex. Devolvemos 0 o -1 para indicar "no calculado"
        metrics["top_nesting_level"] = 0 

    except Exception:
        # Si incluso esto falla (rarísimo), devolvemos ceros
        pass
        
    return metrics

def extract_metrics_from_text(code_snippet, language, file_prefix="test"):
    """
    Función principal que intenta Nivel 1 y cae a Nivel 2.
    """
    metrics = {
        "nloc": 0, "complexity": 0, "token_count": 0, 
        "top_nesting_level": 0, "parameters_count": 0,
        "extraction_method": "lizard"
    }
    
    # 1. Normalizar lenguaje
    lang_key = str(language).lower().strip()
    lizard_lang = LANG_MAPPING.get(lang_key, None)
    
    # 2. Pre-procesado
    clean_code = clean_code_for_parsing(code_snippet, lang_key)

    # --- NIVEL 1: LIZARD DIRECTO ---
    try:
        # Analizamos el string directamente (sin escribir archivos a disco = MÁS RÁPIDO)
        analysis = lizard.analyze_file.analyze_source_code(
            f"dummy_file.{lang_key}", 
            clean_code
        )
        
        # Lizard devuelve métricas por función y un resumen global
        if analysis:
            metrics["nloc"] = analysis.nloc
            metrics["token_count"] = analysis.token_count
            
            # La complejidad ciclomática global en Lizard es el promedio o suma?
            # Para el dataset CVEfixes, usaban el promedio de funciones o el archivo total.
            # Usaremos la suma de complejidad de funciones (método estándar) o la media.
            # PERO: analysis.average_cyclomatic_complexity existe.
            # Ajustaremos para replicar comportamiento: max complexity encontrada o suma.
            
            total_complexity = 0
            max_nesting = 0
            total_params = 0
            
            # Lizard calcula complejidad por función. Si hay código fuera de funciones,
            # a veces no lo suma a la complejidad total de la misma manera.
            # Tomaremos la complejidad promedio calculada por Lizard para ser consistentes.
            
            if analysis.function_list:
                # Si detectó funciones
                metrics["complexity"] = sum(f.cyclomatic_complexity for f in analysis.function_list) / len(analysis.function_list) # Promedio
                
                for func in analysis.function_list:
                    # Tokens por función + tokens globales? Usamos el global de analysis
                    total_params += len(func.parameters)
                    # Lizard estándar a veces no trae nesting en analyze_source_code simple
                    # Verificamos si existe el atributo (depende versión)
                    if hasattr(func, 'top_nesting_level'):
                        if func.top_nesting_level > max_nesting:
                            max_nesting = func.top_nesting_level
            else:
                # Si no detectó funciones (script plano), asumimos complejidad 1 o lo que diga NLOC
                metrics["complexity"] = 1

            metrics["parameters_count"] = total_params
            metrics["top_nesting_level"] = max_nesting
            
            # Validación simple: Si NLOC es > 10 y complexity es 0, algo falló silenciosamente
            if metrics["nloc"] > 10 and metrics["complexity"] == 0:
                raise ValueError("Lizard devolvió métricas vacías")

            return metrics

    except Exception:
        # Si Lizard falla (RecursionError, ParseError, o validación nuestra)
        # Pasamos silenciosamente al fallback
        pass

    # --- NIVEL 2: FALLBACK ---
    # Si llegamos aquí, Lizard falló. Usamos el método sucio.
    return calculate_fallback_metrics(clean_code)