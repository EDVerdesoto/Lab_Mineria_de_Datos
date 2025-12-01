"""
Predictor para inferencia con localización de vulnerabilidades.
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
import json

from models import HybridCodeClassifier
from extractors import ComplexityExtractor, PatternExtractor, ASTExtractor
from config import (
    DEVICE, CODEBERT_MODEL, MAX_SEQ_LENGTH,
    CLASS_NAMES_BINARY, CLASS_NAMES_MULTICLASS, TOP_CWES
)


class VulnerabilityPredictor:
    """
    Predice vulnerabilidades en código y localiza las líneas problemáticas.
    
    Uso:
        predictor = VulnerabilityPredictor('checkpoints/best_model.pt')
        result = predictor.predict(code, language='python')
        print(result['prediction'])
        print(result['vulnerable_lines'])
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = DEVICE
    ):
        self.device = torch.device(device)
        
        # Cargar checkpoint para obtener num_classes
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint.get('num_classes', 2)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
        
        # Modelo
        self.model = HybridCodeClassifier(
            num_classes=num_classes,
            codebert_model=CODEBERT_MODEL
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Extractores
        self.complexity_extractor = ComplexityExtractor()
        self.pattern_extractor = PatternExtractor()
        self.ast_extractor = ASTExtractor()
        
        # Class names
        self.num_classes = num_classes
        if num_classes == 2:
            self.class_names = CLASS_NAMES_BINARY
        else:
            self.class_names = CLASS_NAMES_MULTICLASS[:num_classes]
        
        # CWE descriptions
        self.cwe_descriptions = {
            'CWE-79': 'Cross-site Scripting (XSS)',
            'CWE-89': 'SQL Injection',
            'CWE-119': 'Buffer Overflow',
            'CWE-125': 'Out-of-bounds Read',
            'CWE-200': 'Information Exposure',
            'CWE-264': 'Permissions Issues',
            'CWE-287': 'Authentication Issues',
            'CWE-352': 'Cross-Site Request Forgery (CSRF)',
            'CWE-416': 'Use After Free',
            'CWE-476': 'NULL Pointer Dereference'
        }
        
        print(f"[INFO] Predictor cargado desde {model_path}")
        print(f"[INFO] Clases: {num_classes} ({self.class_names})")
        print(f"[INFO] Device: {self.device}")
    
    @torch.no_grad()
    def predict(
        self,
        code: str,
        language: Optional[str] = None,
        return_embeddings: bool = False
    ) -> Dict:
        """
        Predice si el código es vulnerable y localiza las líneas problemáticas.
        
        Args:
            code: Código fuente a analizar
            language: Lenguaje de programación (opcional)
            return_embeddings: Si retornar los embeddings fusionados
        
        Returns:
            Diccionario con predicción, confianza, líneas vulnerables, etc.
        """
        # Tokenización
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )
        
        # Features manuales
        complexity_feats = torch.tensor(
            [self.complexity_extractor.extract(code, language)],
            dtype=torch.float32
        )
        pattern_feats = torch.tensor(
            [self.pattern_extractor.extract(code, language)],
            dtype=torch.float32
        )
        ast_feats = torch.tensor(
            [self.ast_extractor.extract(code, language)],
            dtype=torch.float32
        )
        
        # Mover a device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        complexity_feats = complexity_feats.to(self.device)
        pattern_feats = pattern_feats.to(self.device)
        ast_feats = ast_feats.to(self.device)
        
        # Predicción
        logits = self.model(
            input_ids, attention_mask,
            complexity_feats, pattern_feats, ast_feats
        )
        
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        # Localización de vulnerabilidades
        pattern_matches = self.pattern_extractor.find_matches(code)
        vulnerability_summary = self.pattern_extractor.get_vulnerability_summary(code)
        
        # Métricas de complejidad
        complexity_values = self.complexity_extractor.extract(code, language)
        complexity_names = self.complexity_extractor.feature_names
        
        # Construir resultado
        result = {
            'prediction': self.class_names[pred_class],
            'prediction_idx': pred_class,
            'confidence': confidence,
            'probabilities': {
                name: probs[0][i].item() 
                for i, name in enumerate(self.class_names)
            },
            'is_vulnerable': pred_class == 0 if self.num_classes == 2 else pred_class != 0,
            'vulnerable_lines': pattern_matches,
            'vulnerability_summary': vulnerability_summary,
            'complexity_metrics': {
                name: value 
                for name, value in zip(complexity_names, complexity_values)
            }
        }
        
        # Agregar descripción de CWE si aplica
        if self.num_classes > 2 and pred_class > 0:
            cwe = self.class_names[pred_class]
            if cwe in self.cwe_descriptions:
                result['cwe_description'] = self.cwe_descriptions[cwe]
        
        # Embeddings opcionales
        if return_embeddings:
            embeddings = self.model.get_embeddings(
                input_ids, attention_mask,
                complexity_feats, pattern_feats, ast_feats
            )
            result['embeddings'] = embeddings.cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None,
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Predice múltiples códigos en batch.
        
        Args:
            codes: Lista de códigos fuente
            languages: Lista de lenguajes (opcional)
            batch_size: Tamaño de batch
        
        Returns:
            Lista de diccionarios con predicciones
        """
        if languages is None:
            languages = [None] * len(codes)
        
        results = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            batch_langs = languages[i:i + batch_size]
            
            for code, lang in zip(batch_codes, batch_langs):
                result = self.predict(code, lang)
                results.append(result)
        
        return results
    
    def analyze_file(self, file_path: str, language: Optional[str] = None) -> Dict:
        """
        Analiza un archivo de código.
        
        Args:
            file_path: Ruta al archivo
            language: Lenguaje (se infiere de la extensión si no se especifica)
        
        Returns:
            Diccionario con análisis completo
        """
        # Leer archivo
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        # Inferir lenguaje de la extensión
        if language is None:
            ext = file_path.split('.')[-1].lower()
            ext_to_lang = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'c': 'c',
                'cpp': 'cpp',
                'cs': 'csharp',
                'php': 'php',
                'rb': 'ruby',
                'go': 'go',
                'rs': 'rust'
            }
            language = ext_to_lang.get(ext, None)
        
        result = self.predict(code, language)
        result['file_path'] = file_path
        result['language'] = language
        result['lines_of_code'] = len(code.split('\n'))
        
        return result
    
    def generate_report(self, result: Dict, format: str = 'text') -> str:
        """
        Genera un reporte legible del análisis.
        
        Args:
            result: Resultado de predict() o analyze_file()
            format: 'text' o 'json'
        
        Returns:
            Reporte formateado
        """
        if format == 'json':
            # Limpiar datos no serializables
            clean_result = {k: v for k, v in result.items() if k != 'embeddings'}
            return json.dumps(clean_result, indent=2)
        
        lines = []
        lines.append("=" * 60)
        lines.append("     ANÁLISIS DE VULNERABILIDADES")
        lines.append("=" * 60)
        
        if 'file_path' in result:
            lines.append(f"Archivo: {result['file_path']}")
            lines.append(f"Lenguaje: {result.get('language', 'Desconocido')}")
            lines.append(f"Líneas: {result.get('lines_of_code', 'N/A')}")
            lines.append("-" * 60)
        
        lines.append(f"Predicción: {result['prediction']}")
        lines.append(f"Confianza: {result['confidence']:.1%}")
        
        if result['is_vulnerable']:
            lines.append("")
            lines.append("⚠️  CÓDIGO VULNERABLE DETECTADO")
            
            if 'cwe_description' in result:
                lines.append(f"Tipo: {result['cwe_description']}")
        else:
            lines.append("")
            lines.append("✓ Código aparentemente seguro")
        
        # Probabilidades
        lines.append("")
        lines.append("Probabilidades:")
        for name, prob in result['probabilities'].items():
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            lines.append(f"  {name:15} {bar} {prob:.1%}")
        
        # Líneas vulnerables detectadas por patrones
        summary = result.get('vulnerability_summary', {})
        if summary.get('total_issues', 0) > 0:
            lines.append("")
            lines.append(f"Patrones detectados: {summary['total_issues']}")
            lines.append(f"Líneas afectadas: {summary['lines_affected']}")
            
            for category, count in summary.get('by_category', {}).items():
                lines.append(f"  - {category}: {count}")
            
            # Mostrar detalles de matches
            lines.append("")
            lines.append("Detalle de vulnerabilidades:")
            for category, matches in result.get('vulnerable_lines', {}).items():
                for match in matches[:5]:  # Máximo 5 por categoría
                    lines.append(f"  Línea {match['line']}: {match['content'][:50]}...")
                    lines.append(f"    → {match['cwe']} ({category})")
        
        # Métricas de complejidad
        lines.append("")
        lines.append("Métricas de complejidad:")
        for name, value in result.get('complexity_metrics', {}).items():
            lines.append(f"  {name}: {value:.2f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def load_predictor(model_path: str) -> VulnerabilityPredictor:
    """Función de conveniencia para cargar el predictor."""
    return VulnerabilityPredictor(model_path)
