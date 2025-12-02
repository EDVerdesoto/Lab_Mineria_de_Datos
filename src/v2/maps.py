# Etiquetas de las Top CWE más comunes
TOP_CWES = [
    'CWE-79', 
    'CWE-787',
    'CWE-89',
    'CWE-352', 
    'CWE-22',
    'CWE-125',
    'CWE-78',
    'CWE-416',
    'CWE-862',
    'CWE-434',
]

# Mapa: Safe=0, TopN=1..N, Other=N+1
LABEL_MAP = {cwe: i+1 for i, cwe in enumerate(TOP_CWES)}
OTHER_LABEL = len(TOP_CWES) + 1

# Lista de lenguajes de programación encontrados en el dataset
PROG_LANGUAGES = [
    'Batchfile', 'C', 'TeX', 'Shell', 'Perl', 'Lua', 'Go', 'Markdown',
    'C++', 'Python', 'R', 'PHP', 'Swift', 'Java', 'SQL', 'Matlab',
    'Ruby', 'JavaScript', 'HTML', 'CSS', 'CoffeeScript', 'C#', 'Scala',
    'PowerShell', 'Rust', 'Haskell', 'TypeScript', 'Erlang', 'Objective-C',
    'Jupyter Notebook'
]

# Creamos el mapa: {'Batchfile': 0, 'C': 1, ...}
LANG_MAP = {lang: i for i, lang in enumerate(PROG_LANGUAGES)}
NUM_LANGS = len(PROG_LANGUAGES) + 1