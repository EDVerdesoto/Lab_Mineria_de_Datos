# Source Code

## Formato del Dataset CVEfixes Preprocesado

## Ubicación
Guardar el archivo como: `cvefixes_processed.csv`

## Columnas requeridas (en este orden):

| Columna   | Tipo   | Descripción                          | Ejemplo                    |
|-----------|--------|--------------------------------------|----------------------------|
| code      | string | Código fuente vulnerable/seguro      | `strcpy(buf, input);`      |
| language  | string | Lenguaje de programación             | `c`, `python`, `java`      |
| safety    | string | Estado de seguridad                  | `safe` o `vulnerable`      |
| cwe_id    | string | Código CWE (null si safe)            | `CWE-119` o `null`         |

## Notas importantes:
- Los CWEs más comunes son ser: CWE-119, CWE-89, CWE-79, CWE-787, CWE-20
- El resto se agrupará como "Other"
- Registros safe deben tener `cwe_id` vacío o null