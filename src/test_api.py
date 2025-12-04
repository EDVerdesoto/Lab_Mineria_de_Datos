import requests
import json

# URL local donde corre tu API
API_URL = "http://localhost:8000/analyze"

# CÓDIGO JAVASCRIPT DE PRUEBA
# Incluye una función limpia y una "tóxica" típica de frontend inseguro
js_code = """
function calculateTax(price) {
    // Función segura: Cálculo matemático puro
    const taxRate = 0.15;
    return price * taxRate;
}

function renderUserProfile(inputString) {
    // PELIGRO: Vulnerabilidades Clásicas de JS
    
    // 1. Uso de 'var' (Mala práctica / Code Smell)
    var userData = inputString; 
    
    // 2. Eval (Inyección de Código - Critical)
    eval("console.log(" + userData + ")");
    
    // 3. Document.write (XSS - High)
    document.write("<div>" + userData + "</div>");
}
"""

payload = [
    {
        "filename": "src/frontend/user_profile.js",
        "programming_language": "javascript", # Ojo aquí: "javascript" o "js"
        "code": js_code
    }
]

print("🚀 Enviando Payload JavaScript a la API...")

try:
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ ¡RESPUESTA EXITOSA!\n")
        
        for archivo in data['results']:
            print(f"📂 Archivo: {archivo['filename']}")
            
            for func in archivo['functions']:
                # Icono según riesgo
                icon = "🟢" if func['risk_score'] < 0.5 else "🔴"
                
                print(f"\n  {icon} Función '{func['function_name']}'")
                print(f"     Riesgo Calculado: {func['risk_score']:.2f}")
                print(f"     📊 Vector IA: {func['features']}")
                
                # Mostrar hallazgos si los hay
                if func['findings']:
                    print(f"     ⚠️  Alertas de Seguridad:")
                    for find in func['findings']:
                        print(f"         - [Línea {find['line']}] {find['severity']}: {find['message']}")
                        
                # Mostrar tags sospechosos
                if func['tags']:
                    print(f"     👀 Tags: {func['tags']}")

    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Error de conexión: {e}")
    print("¿Está corriendo 'python main.py' en otra consola?")