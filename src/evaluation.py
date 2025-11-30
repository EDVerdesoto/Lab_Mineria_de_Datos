import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from config import ID_TO_CWE

def report_evaluation(y_true, y_pred, save_path=None):
    """
    Reporte completo para clasificación multi-clase.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*50)
    print("         REPORTE DE EVALUACIÓN")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Reporte por clase
    print("\n" + "="*50)
    print("         REPORTE POR CLASE")
    print("="*50)
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [ID_TO_CWE.get(i, f"Class_{i}") for i in labels]
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n[INFO] Matriz guardada en: {save_path}")
    plt.show()

    return metrics