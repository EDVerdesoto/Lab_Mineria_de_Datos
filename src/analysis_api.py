import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from telegram_notify import send_telegram_report
from datetime import datetime
import os

# Importamos tu motor de seguridad validado
from security_service import SecurityFeatureExtractor

# Importamos el predictor de vulnerabilidades (CodeBERT)
from v3.predictor import VulnerabilityPredictor

# ==============================================================================
# CONFIGURACIÓN DEL MODELO (Parámetros del notebook 04_codebert_ft.ipynb)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
        logging.info(f"Cargando modelo CodeBERT desde {MODEL_PATH}...")
        vuln_predictor = VulnerabilityPredictor(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME
        )
        logging.info("Modelo cargado exitosamente!")
    return vuln_predictor

# ==============================================================================
# 1. MODELOS DE DATOS (Protocolo de Comunicación)
# ==============================================================================

# Lo que envia el CI/CD (Input)
class CodePayload(BaseModel):
    filename: str
    programming_language: str
    code: str

class AnalysisRequest(BaseModel):
    pr_title: str
    telegram_chat_id: Optional[str] = None
    files: List[CodePayload]
    use_codebert: bool = True

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
    codebert_prediction: Optional[VulnPrediction] = None
    features: Dict[str, Any] 
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

class CodeSnippet(BaseModel):
    code: str

# ==============================================================================
# 2. REPORTES EN MARKDOWN PARA TELEGRAM
# ==============================================================================
def generate_reports(pr_title: str, results: List[FileResult]):
    """
    Retorna dos cosas:
    1. Un resumen corto para el mensaje de chat.
    2. El contenido completo para el archivo Markdown.
    """
    total_files = len(results)
    total_funcs = sum(r.total_functions for r in results)
    high_risks = 0
    all_details = ""

    # Construcción del detalle completo
    for file in results:
        file_has_issues = False
        file_buffer = f"## 📂 Archivo: {file.filename}\n"
        
        for func in file.functions:
            # Lógica de riesgo
            vuln_label = "Safe"
            is_risky = False
            
            if func.codebert_prediction:
                vuln_label = func.codebert_prediction.label
            
            # Criterio de riesgo (puedes ajustarlo)
            if func.risk_score > 60 or (vuln_label != "Safe" and vuln_label != "N/A"):
                is_risky = True
                high_risks += 1
                file_has_issues = True
                
                icon = "🔴" if func.risk_score > 80 else "🟠"
                file_buffer += (
                    f"- {icon} **{func.function_name}** (Líneas {func.start_line}-{func.end_line})\n"
                    f"  - **Score:** {func.risk_score:.2f} | **IA:** {vuln_label}\n"
                )
                # Agregar hallazgos estáticos si existen
                if func.findings:
                    for finding in func.findings:
                        file_buffer += f"  - 🔍 *Regla:* {finding.get('rule', 'General')}\n"

        if file_has_issues:
            all_details += file_buffer + "\n"

    # 1. El Resumen (Mensaje de Chat)
    status = "✅ APROBADO" if high_risks == 0 else "⚠️ REVISIÓN REQUERIDA"
    summary_msg = (
        f"🛡️ *Security Audit: {pr_title}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Estado: *{status}*\n"
        f"• Archivos: `{total_files}`\n"
        f"• Vulnerabilidades: `{high_risks}`\n"
    )
    if high_risks > 0:
        summary_msg += "\n👇 _Ver archivo adjunto para detalles._"

    # 2. El Reporte Completo (Contenido del Archivo)
    full_report_content = (
        f"# Reporte de Seguridad - {pr_title}\n"
        f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Resumen\n"
        f"- Total Archivos: {total_files}\n"
        f"- Total Funciones: {total_funcs}\n"
        f"- Hallazgos Críticos: {high_risks}\n\n"
        f"--- \n"
        f"## Detalle de Vulnerabilidades\n\n"
        f"{all_details if all_details else '🎉 Sin vulnerabilidades detectadas.'}"
    )

    return summary_msg, full_report_content

def process_and_notify(request: AnalysisRequest, results: List[FileResult]):
    """Tarea en segundo plano"""
    if request.telegram_chat_id:
        try:
            # Generamos ambos textos
            summary, full_content = generate_reports(request.pr_title, results)
            
            # Decidimos si enviar solo resumen o resumen + archivo
            # Si no hay riesgos, quizás no quieras mandar archivo (opcional)
            # Aquí mandamos archivo siempre si hay contenido detallado.
            
            filename = f"security_report_{datetime.now().strftime('%H%M%S')}.md"
            
            send_telegram_report(
                chat_id=request.telegram_chat_id,
                summary_text=summary,
                file_content=full_content,
                filename=filename
            )
        except Exception as e:
            logging.exception(f"Error en background task de Telegram: {e}")

# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """Pre-carga el modelo al iniciar la API (opcional, mejora latencia)."""
    try:
        get_vuln_predictor()
    except Exception as e:
        print(f"No se pudo pre-cargar el modelo: {e}")
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
def analyze_batch(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analiza código y opcionalmente envía reporte a Telegram en segundo plano.
    """
    processed_results = []
    
    # Obtener predictor
    predictor = get_vuln_predictor() if request.use_codebert else None
    
    print(f"Analizando PR: '{request.pr_title}' ({len(request.files)} archivos)")

    for file_data in request.files:
        try:
            # 1. Llamada al Motor (Feature Extraction)
            functions_data = extractor.analyze_file_functions(
                code=file_data.code,
                language=file_data.programming_language,
                filename=file_data.filename
            )
            
            # 2. Procesamiento y CodeBERT
            api_functions = []
            for func in functions_data:
                func_code = func.get('code', '')
                
                codebert_pred = None
                if predictor and func_code:
                    try:
                        pred_result = predictor.predict(func_code)
                        codebert_pred = VulnPrediction(
                            label=pred_result['label'],
                            confidence=pred_result['confidence'],
                            probabilities=pred_result['probabilities']
                        )
                    except Exception:
                        logging.exception("Error en predicción CodeBERT")
                        pass
                
                # Mapeo a objeto FunctionAnalysis
                f_obj = FunctionAnalysis(
                    function_name=func['function_name'],
                    start_line=func['start_line'],
                    end_line=func['end_line'],
                    risk_score=func['risk_score'],
                    codebert_prediction=codebert_pred,
                    features=func['features'], 
                    findings=func['context']['static_findings'],
                    tags=func['context']['suspicious_tags']
                )
                api_functions.append(f_obj)

            processed_results.append(FileResult(
                filename=file_data.filename,
                total_functions=len(api_functions),
                functions=api_functions
            ))
            
        except Exception as e:
            logging.exception(f"Error procesando {file_data.filename}: {e}")
            continue

    # Agregamos la tarea de envío a Telegram a la cola (se ejecuta DESPUÉS de return)
    if request.telegram_chat_id:
        background_tasks.add_task(process_and_notify, request, processed_results)

    return BatchResponse(
        total_files_processed=len(request.files),
        results=processed_results
    )

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
    
@app.post("/admin/reload-model", status_code=status.HTTP_200_OK)
def reload_model_hot():
    """
    Fuerza la recarga del modelo desde el disco sin reiniciar el servidor.
    Útil cuando acabas de reemplazar el archivo .bin
    """
    global vuln_predictor
    
    # 1. Liberar memoria del modelo anterior
    vuln_predictor = None 
    
    # 2. Forzar carga inmediata (opcional, o dejar que se cargue en el siguiente request)
    try:
        get_vuln_predictor()
        return {"message": "Modelo recargado exitosamente desde disco."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando nuevo modelo: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)