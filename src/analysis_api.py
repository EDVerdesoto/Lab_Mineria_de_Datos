import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Importamos tu motor de seguridad validado
from security_service import SecurityFeatureExtractor

# Importamos el predictor de vulnerabilidades (CodeBERT)
from v3.predictor import VulnerabilityPredictor

# ==============================================================================
# CONFIGURACIÓN DEL MODELO (Parámetros del notebook 04_codebert_ft.ipynb)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "codebert_vuln", "best_model.bin")
MODEL_NAME = "microsoft/codebert-base"

# Inicializar la aplicación y el extractor
app = FastAPI(title="Security Audit Microservice", version="3.0 (CodeBERT)")
extractor = SecurityFeatureExtractor()

# Inicializar el predictor de vulnerabilidades (lazy loading)
vuln_predictor: VulnerabilityPredictor = None

def get_vuln_predictor() -> VulnerabilityPredictor:
    """Carga el modelo de forma lazy (solo cuando se necesita)."""
    global vuln_predictor
    if vuln_predictor is None:
        print(f"🔄 Cargando modelo CodeBERT desde {MODEL_PATH}...")
        vuln_predictor = VulnerabilityPredictor(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME
        )
        print("✅ Modelo cargado exitosamente!")
    return vuln_predictor

# ==============================================================================
# 1. MODELOS DE DATOS (Protocolo de Comunicación)
# ==============================================================================

# Lo que envia el CI/CD (Input)
class CodePayload(BaseModel):
    filename: str
    programming_language: str
    code: str

# Predicción del modelo CodeBERT (Output)
class VulnPrediction(BaseModel):
    label: str                          # Ej: "CWE-79", "Safe", "Other"
    confidence: float                   # Confianza de la predicción
    probabilities: Dict[str, float]     # Probabilidades por clase

# El resultado de una sola función (Output Nivel 3)
class FunctionAnalysis(BaseModel):
    function_name: str
    start_line: int
    end_line: int
    risk_score: float
    # Predicción del modelo CodeBERT
    codebert_prediction: Optional[VulnPrediction] = None
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

@app.on_event("startup")
async def startup_event():
    """Pre-carga el modelo al iniciar la API (opcional, mejora latencia)."""
    try:
        get_vuln_predictor()
    except Exception as e:
        print(f"⚠️ No se pudo pre-cargar el modelo: {e}")
        print("El modelo se cargará en la primera solicitud.")

@app.get("/")
def health_check():
    model_loaded = vuln_predictor is not None
    return {
        "status": "online", 
        "mode": "function-level-analysis + CodeBERT",
        "model_loaded": model_loaded
    }

@app.post("/analyze", response_model=BatchResponse)
def analyze_batch(payload: List[CodePayload], use_codebert: bool = True):
    """
    Recibe archivos -> Detecta Funciones -> Extrae Métricas y Riesgos.
    
    Args:
        payload: Lista de archivos a analizar
        use_codebert: Si True, usa el modelo CodeBERT para predicción (default: True)
    """
    processed_results = []
    
    # Obtener predictor si se necesita
    predictor = get_vuln_predictor() if use_codebert else None
    
    print(f"📥 Recibiendo lote de {len(payload)} archivos...")
    if use_codebert:
        print(f"🤖 Usando modelo CodeBERT para predicciones")

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
                # Extraer el código de la función para CodeBERT
                func_code = func.get('code', '')
                
                # Predicción con CodeBERT (si está habilitado y hay código)
                codebert_pred = None
                if predictor and func_code:
                    try:
                        pred_result = predictor.predict(func_code)
                        codebert_pred = VulnPrediction(
                            label=pred_result['label'],
                            confidence=pred_result['confidence'],
                            probabilities=pred_result['probabilities']
                        )
                    except Exception as pred_err:
                        print(f"⚠️ Error en predicción CodeBERT: {pred_err}")
                
                # Construimos el objeto FunctionAnalysis
                f_obj = FunctionAnalysis(
                    function_name=func['function_name'],
                    start_line=func['start_line'],
                    end_line=func['end_line'],
                    risk_score=func['risk_score'],
                    codebert_prediction=codebert_pred,
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

# ==============================================================================
# 3. ENDPOINT DE PREDICCIÓN DIRECTA (Solo CodeBERT)
# ==============================================================================

class CodeSnippet(BaseModel):
    code: str

@app.post("/predict", response_model=VulnPrediction)
def predict_vulnerability(snippet: CodeSnippet):
    """
    Predice la vulnerabilidad de un fragmento de código usando CodeBERT.
    
    Útil para:
    - Pruebas rápidas del modelo
    - Integración directa sin análisis de métricas
    - Validación de funciones individuales
    """
    predictor = get_vuln_predictor()
    
    try:
        result = predictor.predict(snippet.code)
        return VulnPrediction(
            label=result['label'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)