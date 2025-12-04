import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Importamos tu motor de seguridad validado
from security_service import SecurityFeatureExtractor

# Inicializar la aplicación y el extractor
app = FastAPI(title="Security Audit Microservice", version="2.0 (Function-Level)")
extractor = SecurityFeatureExtractor()

# ==============================================================================
# 1. MODELOS DE DATOS (Protocolo de Comunicación)
# ==============================================================================

# Lo que envia el CI/CD (Input)
class CodePayload(BaseModel):
    filename: str
    programming_language: str
    code: str

# El resultado de una sola función (Output Nivel 3)
class FunctionAnalysis(BaseModel):
    function_name: str
    start_line: int
    end_line: int
    risk_score: float
    # El vector numérico para tu modelo de IA [nloc, complexity, tokens...]
    features: Dict[str, Any] 
    # Detalles para el reporte humano
    findings: List[Dict[str, Any]]
    tags: List[str]

# El resultado de un archivo completo (Output Nivel 2)
class FileResult(BaseModel):
    filename: str
    total_functions: int
    functions: List[FunctionAnalysis]

# La respuesta global del lote (Output Nivel 1)
class BatchResponse(BaseModel):
    total_files_processed: int
    results: List[FileResult]

# ==============================================================================
# 2. ENDPOINTS
# ==============================================================================

@app.get("/")
def health_check():
    return {"status": "online", "mode": "function-level-analysis"}

@app.post("/analyze", response_model=BatchResponse)
def analyze_batch(payload: List[CodePayload]):
    """
    Recibe archivos -> Detecta Funciones -> Extrae Métricas y Riesgos.
    """
    processed_results = []
    
    print(f"📥 Recibiendo lote de {len(payload)} archivos...")

    for file_data in payload:
        try:
            # 1. Llamada al Motor (Analiza por funciones)
            # Retorna una lista de diccionarios (uno por función)
            functions_data = extractor.analyze_file_functions(
                code=file_data.code,
                language=file_data.programming_language,
                filename=file_data.filename
            )
            
            # 2. Mapeo de datos internos a la respuesta de la API
            api_functions = []
            for func in functions_data:
                # Construimos el objeto FunctionAnalysis
                f_obj = FunctionAnalysis(
                    function_name=func['function_name'],
                    start_line=func['start_line'],
                    end_line=func['end_line'],
                    risk_score=func['risk_score'],
                    features=func['features'],  # Aquí van NLOC, Complexity, etc.
                    findings=func['context']['static_findings'], # Reglas rotas
                    tags=func['context']['suspicious_tags']      # Tokens peligrosos
                )
                api_functions.append(f_obj)

            # 3. Agregamos el resultado del archivo a la lista global
            processed_results.append(FileResult(
                filename=file_data.filename,
                total_functions=len(api_functions),
                functions=api_functions
            ))
            
        except Exception as e:
            print(f"❌ Error procesando {file_data.filename}: {e}")
            # En producción podrías agregar un objeto de error a la lista, 
            # aquí simplemente saltamos el archivo corrupto.
            continue

    return BatchResponse(
        total_files_processed=len(payload),
        results=processed_results
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)