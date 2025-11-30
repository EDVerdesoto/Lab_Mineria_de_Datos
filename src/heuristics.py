import re
from config import ID_TO_CWE

class HeuristicAnalyzer:
    def __init__(self):
        self.PATTERNS = {
            "CWE-119": [
                r'\bstrcpy\s*\(', r'\bstrcat\s*\(', r'\bsprintf\s*\(',
                r'\bgets\s*\(', r'\bmemcpy\s*\(', r'\bscanf\s*\('
            ],
            "CWE-89": [
                r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'DELETE\s+FROM',
                r'UPDATE\s+.*\s+SET', r'\bexecute\s*\(', r'"\s*\+\s*\w+\s*\+'
            ],
            "CWE-79": [
                r'\.innerHTML\s*=', r'document\.write\s*\(',
                r'<script>', r'eval\s*\(', r'\$_GET\[', r'\$_POST\['
            ],
            "Other": [
                r'\bsystem\s*\(', r'\bpopen\s*\(', r'\bexec\s*\('
            ]
        }

    def _is_comment(self, line):
        clean = line.strip()
        return (clean.startswith("//") or clean.startswith("/*") or 
                clean.startswith("*") or clean.startswith("#"))

    def find_suspicious_lines(self, code, cwe_id):
        """Devuelve líneas sospechosas basadas en la clase predicha."""
        cwe_name = ID_TO_CWE.get(cwe_id, "Other")
        key = next((k for k in self.PATTERNS if k in cwe_name), "Other")
        regex_list = self.PATTERNS.get(key, self.PATTERNS["Other"])
        
        findings = []
        for line_num, line in enumerate(code.split('\n'), 1):
            if self._is_comment(line):
                continue
            for pattern in regex_list:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        'line': line_num,
                        'content': line.strip(),
                        'pattern': pattern
                    })
        return findings

    def explain_prediction(self, code, cwe_id):
        """Genera explicación legible de la predicción."""
        cwe_name = ID_TO_CWE.get(cwe_id, "Unknown")
        findings = self.find_suspicious_lines(code, cwe_id)
        
        explanation = f"\n[Predicción: {cwe_name}]\n"
        if findings:
            explanation += f"Se encontraron {len(findings)} patrones sospechosos:\n"
            for f in findings[:5]:  # Máximo 5
                explanation += f"  Línea {f['line']}: {f['content'][:60]}...\n"
        else:
            explanation += "No se encontraron patrones heurísticos específicos.\n"
        
        return explanation