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
