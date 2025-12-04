import re
from typing import List, Dict, Set, Any
from metrics_core import extract_metrics_from_text
import lizard

# ==============================================================================
# 1. BASE DE CONOCIMIENTO (Tus definiciones)
# ==============================================================================

# Lista de indicadores críticos (Tokens que el modelo debe "mirar")

VULNERABILITY_INDICATORS = {
    # =========================================================================
    # CWE-89: SQL Injection
    # =========================================================================
    'execute', 'executemany', 'executescript', 'executequery', 'executeupdate',
    'cursor', 'query', 'rawquery', 'raw', 'nativequery', 'createquery',
    'preparestatement', 'preparedstatement', 'statement',
    'select', 'insert', 'update', 'delete', 'drop', 'truncate', 'alter',
    'where', 'from', 'join', 'union', 'having', 'order', 'group',
    'sqlalchemy', 'hibernate', 'jdbc', 'mybatis', 'sequelize', 'knex',
    
    # =========================================================================
    # CWE-79: XSS (Cross-Site Scripting)
    # =========================================================================
    'innerhtml', 'outerhtml', 'innertext', 'textcontent',
    'documentwrite', 'writeln', 'createelement', 'appendchild',
    'insertadjacenthtml', 'setattribute',
    'jquery', 'html', 'append', 'prepend',
    'dangerouslysetinnerhtml',
    'bypasssecuritytrust',
    'safe', 'mark_safe', 'autoescape', 'noescape',
    
    # =========================================================================
    # CWE-78: OS Command Injection
    # =========================================================================
    'system', 'popen', 'subprocess', 'call', 'check_output', 'check_call',
    'spawn', 'fork', 'exec', 'execfile', 'execl', 'execv', 'execve',
    'shell_exec', 'passthru', 'proc_open',
    'runtime', 'getruntime', 'processbuilder',
    'child_process', 'execsync',
    'shell', 'cmd', 'command', 'bash', 'powershell',
    
    # =========================================================================
    # CWE-22: Path Traversal
    # =========================================================================
    'open', 'read', 'write', 'readfile', 'writefile',
    'fopen', 'fread', 'fwrite', 'file_get_contents', 'file_put_contents',
    'path', 'join', 'abspath', 'realpath', 'normpath',
    'dirname', 'basename', 'filepath', 'filename',
    'include', 'require', 'include_once', 'require_once',
    'sendfile', 'download', 'attachment',
    'listdir', 'scandir', 'glob', 'walk',
    'unlink', 'remove', 'delete', 'mkdir', 'rmdir',
    
    # =========================================================================
    # CWE-434: Unrestricted File Upload
    # =========================================================================
    'upload', 'fileupload', 'multipart', 'formdata',
    'saveas', 'move_uploaded_file', 'moveuploaded',
    'getoriginalfilename', 'getfilename', 'getcontenttype',
    'mimetype', 'extension', 'getsize',
    'tempfile', 'temporary',
    
    # =========================================================================
    # CWE-352: CSRF
    # =========================================================================
    'csrf', 'csrftoken', 'csrf_token', 'xsrf', 'xsrftoken',
    'antiforgery', 'validateantiforgerytoken', 'csrf_exempt',
    'samesite', 'origin', 'referer',
    
    # =========================================================================
    # CWE-502: Deserialization
    # =========================================================================
    'pickle', 'unpickle', 'loads', 'load', 'dumps', 'dump',
    'yaml', 'unsafe_load', 'marshal', 'shelve',
    'unserialize', 'serialize',
    'objectinputstream', 'readobject', 'xmldecoder',
    'deserialize', 'jsonpickle', 'dill',
    
    # =========================================================================
    # CWE-611: XXE
    # =========================================================================
    'xml', 'parse', 'parsestring', 'etree', 'minidom',
    'saxparser', 'xmlreader', 'documentbuilder',
    'loadxml', 'entity', 'dtd', 'doctype',
    
    # =========================================================================
    # CWE-918: SSRF
    # =========================================================================
    'request', 'requests', 'urllib', 'urlopen', 'httplib',
    'fetch', 'axios', 'http', 'https', 'curl', 'curl_exec',
    'httpclient', 'webclient', 'resttemplate',
    
    # =========================================================================
    # CWE-94: Code Injection
    # =========================================================================
    'eval', 'exec', 'compile', 'execfile',
    'function', 'settimeout', 'setinterval',
    'constructor', 'prototype',
    'reflection', 'invoke', 'getmethod', 'forname',
    'template', 'render', 'render_template',
    
    # =========================================================================
    # Authentication / Secrets
    # =========================================================================
    'password', 'passwd', 'secret', 'secretkey', 'apikey', 'api_key',
    'token', 'accesstoken', 'privatekey', 'credential',
    'auth', 'authenticate', 'login', 'session', 'cookie',
    'jwt', 'bearer', 'oauth',
    'hash', 'md5', 'sha1', 'sha256', 'bcrypt',
    
    # =========================================================================
    # Input Sources
    # =========================================================================
    'request', 'req', 'input', 'stdin',
    'params', 'query', 'body', 'form', 'args',
    'getparameter', 'getheader',
    'get', 'post', 'put', 'delete',
    'argv', 'environ', 'getenv',
    'userinput', 'userdata',
    
    # =========================================================================
    # Crypto (Weak)
    # =========================================================================
    'random', 'rand', 'mt_rand',
    'des', 'rc4', 'md5', 'sha1', 'ecb',
    'encrypt', 'decrypt', 'cipher',
    
    # =========================================================================
    # Memory (C/C++)
    # =========================================================================
    'strcpy', 'strcat', 'sprintf', 'gets', 'scanf',
    'memcpy', 'memmove', 'malloc', 'free',
    'buffer', 'overflow',
}


INDICATORS_SET = frozenset(VULNERABILITY_INDICATORS)

# Reglas estáticas de "Code Smell" o malas prácticas
STATIC_RULES = {
    'javascript': [
        {'pattern': r'\bvar\b', 'msg': 'Uso de "var" detectado. Prefiera "let" o "const".', 'severity': 'LOW'},
        {'pattern': r'eval\(', 'msg': 'Uso peligroso de "eval()". Riesgo de inyección.', 'severity': 'HIGH'},
        {'pattern': r'document\.write', 'msg': 'Uso de document.write. Riesgo de XSS.', 'severity': 'MEDIUM'},
    ],
    'python': [
        {'pattern': r'shell=True', 'msg': 'Uso de shell=True en subprocess.', 'severity': 'HIGH'},
        {'pattern': r'exec\(', 'msg': 'Uso de exec(). Riesgo de ejecución arbitraria.', 'severity': 'HIGH'},
        {'pattern': r'pickle\.load', 'msg': 'Deserialización insegura (pickle).', 'severity': 'HIGH'},
        {'pattern': r'md5\(', 'msg': 'Uso de algoritmo de hash débil (MD5).', 'severity': 'MEDIUM'}
    ],
    'c': [
        {'pattern': r'strcpy\(', 'msg': 'Uso inseguro de strcpy (Buffer Overflow). Use strncpy.', 'severity': 'HIGH'},
        {'pattern': r'gets\(', 'msg': 'Uso prohibido de gets (Buffer Overflow).', 'severity': 'CRITICAL'},
        {'pattern': r'sprintf\(', 'msg': 'Uso riesgoso de sprintf. Use snprintf.', 'severity': 'MEDIUM'}
    ],
    'php': [
        {'pattern': r'shell_exec\(', 'msg': 'Ejecución de comandos de sistema detectada.', 'severity': 'HIGH'},
        {'pattern': r'unserialize\(', 'msg': 'Deserialización insegura de objetos.', 'severity': 'HIGH'}
    ]
}

class SecurityFeatureExtractor:
    
    def analyze_file_functions(self, code: str, language: str, filename: str = "unknown") -> List[Dict[str, Any]]:
        lang_key = str(language).lower().strip()
        results = []
        
        # --- FIX: Mapa de extensiones para ayudar a Lizard ---
        EXTENSION_MAP = {
            'python': '.py', 'py': '.py', 'javascript': '.js', 'js': '.js', 
            'typescript': '.ts', 'c': '.c', 'cpp': '.cpp', 'java': '.java', 'php': '.php'
        }
        
        if filename == "unknown" or "." not in filename:
            valid_ext = EXTENSION_MAP.get(lang_key, ".txt")
            filename = f"virtual_file{valid_ext}"

        try:
            analysis = lizard.analyze_file.analyze_source_code(filename, code)
        except Exception as e:
            print(f"Error parsing: {e}")
            return []

        if not analysis.function_list:
             return [self._analyze_single_block(code, "main_script", 1, len(code.splitlines()), lang_key, filename)]

        lines = code.splitlines()
        
        for func in analysis.function_list:
            start = func.start_line - 1
            end = func.end_line
            func_code_str = "\n".join(lines[start:end])
            
            func_analysis = self._analyze_single_block(
                code_snippet=func_code_str,
                func_name=func.name,
                start_line=func.start_line,
                end_line=func.end_line,
                language=lang_key,
                filename=filename,
                pre_calc_metrics=func 
            )
            results.append(func_analysis)
            
        return results

    def _analyze_single_block(self, code_snippet, func_name, start_line, end_line, language, filename, pre_calc_metrics=None):
        
        # 1. Métricas Numéricas
        if pre_calc_metrics:
            metrics = {
                "nloc": pre_calc_metrics.nloc,
                "complexity": pre_calc_metrics.cyclomatic_complexity,
                "token_count": pre_calc_metrics.token_count,
                "parameters_count": len(pre_calc_metrics.parameters),
                "top_nesting_level": getattr(pre_calc_metrics, 'top_nesting_level', 0)
            }
        else:
            metrics = extract_metrics_from_text(code_snippet, language, filename)
            if not metrics: metrics = {"nloc": 0, "complexity": 0, "token_count": 0, "parameters": 0, "nesting": 0}

        # 2. Detección DE LÍNEA EXACTA (Mejora)
        # Ahora obtenemos el detalle de dónde están los tags y las reglas
        detailed_indicators = self._scan_indicators_detailed(code_snippet, offset=start_line)
        static_findings = self._check_static_rules(code_snippet, language, offset_line=start_line)

        # Recopilar todos los tags únicos encontrados para el vector de características
        unique_tags = {item['token'] for item in detailed_indicators}

        # 3. Cálculo de Riesgo
        suspicious_count = len(unique_tags)
        findings_count = len(static_findings)
        
        # Riesgo simple: si hay hallazgos estáticos es muy grave (0.8+), si solo hay tags es advertencia
        risk_score = min((suspicious_count * 0.15) + (findings_count * 0.4), 1.0)

        return {
            "function_name": func_name,
            "start_line": start_line,
            "end_line": end_line,
            "risk_score": risk_score,
            "features": {
                "nloc": metrics.get("nloc", 0),
                "complexity": metrics.get("complexity", 0),
                "token_count": metrics.get("token_count", 0),
                "parameters_count": metrics.get("parameters_count", 0),
                "top_nesting_level": metrics.get("top_nesting_level", 0),
                "suspicious_tags_count": suspicious_count
            },
            "context": {
                "suspicious_tags": list(unique_tags),
                "indicator_locations": detailed_indicators, # <--- AQUÍ ESTÁ EL CÓDIGO CULPABLE
                "static_findings": static_findings
            }
        }

    def _scan_indicators_detailed(self, code: str, offset: int) -> List[Dict]:
        """Escanea línea por línea para encontrar DÓNDE están los indicadores."""
        matches = []
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Tokenizamos la línea
            tokens_in_line = re.findall(r'\w+', line.lower())
            # Intersección con nuestra base de datos de peligro
            found = set(tokens_in_line).intersection(INDICATORS_SET)
            
            for token in found:
                matches.append({
                    'line': offset + i,
                    'token': token,
                    'content': line.strip()
                })
        return matches

    def _check_static_rules(self, code: str, lang: str, offset_line: int = 1) -> List[Dict]:
        findings = []
        lines = code.split('\n')
        
        if lang in ['js', 'ts', 'typescript']: lang = 'javascript'
        if lang in ['c++', 'cpp']: lang = 'c'
        
        rules = STATIC_RULES.get(lang, [])
        
        for i, line in enumerate(lines):
            for rule in rules:
                if re.search(rule['pattern'], line):
                    findings.append({
                        'line': offset_line + i,
                        'content': line.strip()[:80],
                        'message': rule['msg'],
                        'severity': rule['severity']
                    })
        return findings

# ==============================================================================
# 3. ZONA DE PRUEBAS VISUALES
# ==============================================================================
if __name__ == "__main__":
    extractor = SecurityFeatureExtractor()
    
    code_file = """import pickle

def safe_function(x):
    return x * 2
    
def vulnerable_function(data):
    # Esta función debe tener riesgo alto
    obj = pickle.loads(data)
    exec(obj)
    return True
"""
    
    print(f"{'='*60}")
    print(f"{'REPORTE DE SEGURIDAD POR FUNCIONES':^60}")
    print(f"{'='*60}")
    
    results = extractor.analyze_file_functions(code_file, "python")
    
    for res in results:
        icon = "🟢" if res['risk_score'] < 0.5 else "🔴"
        print(f"\n{icon} Función: {res['function_name']} (Líneas {res['start_line']}-{res['end_line']})")
        print(f"   Riesgo Calculado: {res['risk_score']:.2f}")
        
        # --- AQUÍ ESTÁN TUS MÉTRICAS ---
        f = res['features']
        print(f"   📊 Métricas (Vector para IA):")
        print(f"      [NLOC: {f['nloc']} | Complexity: {f['complexity']} | Tokens: {f['token_count']} | Params: {f['parameters_count']} | Suspicious: {f['suspicious_tags_count']}]")

        # 1. Mostrar Hallazgos de Reglas (Lo más grave)
        findings = res['context']['static_findings']
        if findings:
            print(f"   ⚠️  Violaciones de Reglas ({len(findings)}):")
            for find in findings:
                print(f"      [Línea {find['line']}] {find['message']}")
                print(f"      └── Código: {find['content']}")
        
        # 2. Mostrar Indicadores Sospechosos (Contexto para el modelo)
        locs = res['context']['indicator_locations']
        if locs:
            print(f"   👀 Indicadores Detectados ({len(locs)}):")
            for loc in locs:
                print(f"      [Línea {loc['line']}] Token: '{loc['token']}'")
                print(f"      └── Código: {loc['content']}")