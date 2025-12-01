import pandas as pd
import os
import requests
import sys

PREDICTIONS_FILE = "data/processed/predictions.csv"
THRESHOLD = 70.0 # Umbral de probabilidad para alerta

def create_github_issue(filename, prob, features):
    """Crea un issue en GitHub usando la API."""
    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")
    
    if not repo or not token:
        print("⚠️ GITHUB_REPOSITORY o GITHUB_TOKEN no configurados. Saltando creación de issue.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Cuerpo del Issue
    title = f"🔴 Security Alert: {filename} ({prob}%)"
    body = f"""
    ### 🛡️ ML Security Alert
    
    The Machine Learning model has detected a high probability of vulnerability.
    
    - **File:** `{filename}`
    - **Vulnerability Probability:** **{prob}%**
    - **Risk Factor:** High
    
    #### Detected Metrics:
    - LOC: {features.get('loc')}
    - Complexity: {features.get('complexity')}
    - Dangerous Functions: {features.get('uses_dangerous_funcs')}
    
    **Recommended Action:**
    Please review the code for potential CWE flaws (Command Injection, XSS, etc.).
    """
    
    payload = {"title": title, "body": body, "labels": ["security", "automated-scan"]}
    
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 201:
        print(f"✅ Issue creado para {filename}")
    else:
        print(f"❌ Error creando issue: {resp.content}")

def main():
    if not os.path.exists(PREDICTIONS_FILE):
        print("No predictions file found.")
        return

    df = pd.read_csv(PREDICTIONS_FILE)
    
    # Filtrar solo las vulnerabilidades altas
    vulnerables = df[df['probability'] > THRESHOLD]
    
    if vulnerables.empty:
        print("✨ No high-risk vulnerabilities detected.")
        return

    print(f"🚨 Detectadas {len(vulnerables)} vulnerabilidades críticas. Generando alertas...")
    
    for _, row in vulnerables.iterrows():
        # Pasamos métricas extra para contexto
        features = {
            'loc': row.get('loc'), 
            'complexity': row.get('complexity'),
            'uses_dangerous_funcs': row.get('uses_dangerous_funcs')
        }
        create_github_issue(row['file_id'], row['probability'], features)

if __name__ == "__main__":
    main()