import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import xgboost as xgb

from config import DATA_PATH, NUM_CLASSES, MODEL_PATH, ID_TO_CWE
from feature_extractor import FeatureExtractor
from codebert import CodeBERTTrainer, CODEBERT_PATH


def compare_models(sample_size=5000):
    """
    Compara XGBoost vs CodeBERT en el mismo subset de datos.
    """
    print("="*60)
    print("       COMPARACIÓN: XGBoost vs CodeBERT")
    print("="*60)
    
    # Cargar muestra del dataset
    print(f"\n[INFO] Cargando {sample_size:,} muestras...")
    df = pd.read_csv(DATA_PATH, nrows=sample_size)
    df = df.dropna(subset=['code', 'target'])
    
    # Aplicar mapeo de CWE si es necesario
    from config import CWE_MAP
    if 'target' not in df.columns:
        df['target'] = df['cwe_id'].map(CWE_MAP).fillna(CWE_MAP['Other'])
        df.loc[df['cwe_id'].isna(), 'target'] = CWE_MAP['Safe']
    
    df['target'] = df['target'].astype(int)
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['target'], random_state=42
    )
    
    results = {}
    
    # === XGBoost ===
    print("\n" + "-"*40)
    print("Entrenando XGBoost...")
    print("-"*40)
    
    extractor = FeatureExtractor()
    
    languages_train = train_df['language'] if 'language' in train_df.columns else None
    languages_test = test_df['language'] if 'language' in test_df.columns else None
    
    X_train, _ = extractor.transform(train_df['code'].astype(str), languages_train)
    X_test, _ = extractor.transform(test_df['code'].astype(str), languages_test)
    
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        objective='multi:softprob',
        num_class=NUM_CLASSES,
        tree_method='hist',
        n_jobs=-1
    )
    xgb_model.fit(X_train, train_df['target'])
    xgb_time = time.time() - start
    
    xgb_preds = xgb_model.predict(X_test)
    
    results['XGBoost'] = {
        'f1': f1_score(test_df['target'], xgb_preds, average='weighted'),
        'accuracy': accuracy_score(test_df['target'], xgb_preds),
        'time': xgb_time,
        'preds': xgb_preds
    }
    print(f"[SUCCESS] XGBoost - F1: {results['XGBoost']['f1']:.4f} | Tiempo: {xgb_time:.2f}s")
    
    # === CodeBERT ===
    print("\n" + "-"*40)
    print("Entrenando CodeBERT...")
    print("-"*40)
    
    start = time.time()
    bert_trainer = CodeBERTTrainer(num_classes=NUM_CLASSES)
    bert_trainer.train_on_chunk(train_df, epochs=3, batch_size=8)
    bert_time = time.time() - start
    
    bert_preds, _ = bert_trainer.predict(test_df['code'].astype(str).tolist())
    
    results['CodeBERT'] = {
        'f1': f1_score(test_df['target'], bert_preds, average='weighted'),
        'accuracy': accuracy_score(test_df['target'], bert_preds),
        'time': bert_time,
        'preds': bert_preds
    }
    print(f"[SUCCESS] CodeBERT - F1: {results['CodeBERT']['f1']:.4f} | Tiempo: {bert_time:.2f}s")
    
    # === Resultados Finales ===
    print("\n" + "="*60)
    print("              RESULTADOS FINALES")
    print("="*60)
    print(f"\n{'Modelo':<15} {'F1-Score':<12} {'Accuracy':<12} {'Tiempo':<10}")
    print("-"*50)
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['f1']:<12.4f} {metrics['accuracy']:<12.4f} {metrics['time']:<10.2f}s")
    
    # Mejor modelo
    best = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\n[INFO] Mejor modelo: {best[0]} (F1: {best[1]['f1']:.4f})")
    
    # Reporte detallado por clase
    print("\n" + "="*60)
    print("         REPORTE POR CLASE - XGBoost")
    print("="*60)
    labels = sorted(set(test_df['target']))
    target_names = [ID_TO_CWE.get(i, f"Class_{i}") for i in labels]
    print(classification_report(test_df['target'], results['XGBoost']['preds'], 
                                target_names=target_names, zero_division=0))
    
    print("\n" + "="*60)
    print("         REPORTE POR CLASE - CodeBERT")
    print("="*60)
    print(classification_report(test_df['target'], results['CodeBERT']['preds'], 
                                target_names=target_names, zero_division=0))
    
    return results


if __name__ == "__main__":
    compare_models(sample_size=5000)