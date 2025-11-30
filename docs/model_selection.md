# SELECCION DEL MODELO: XGBOOST

Se seleccionó **XGBoost (Extreme Gradient Boosting)** como algoritmo principal por las siguientes razones:

### 1. Ensemble de Árboles con Gradient Boosting
XGBoost construye múltiples árboles de decisión de forma secuencial, donde cada árbol corrige los errores del anterior mediante optimización del gradiente. Esto produce un modelo más robusto que un árbol individual y reduce tanto el sesgo como la varianza.

### 2. Manejo de Desbalanceo de Clases
En clasificación de vulnerabilidades, la clase "Safe" típicamente domina el dataset (>70% de muestras). XGBoost permite:
- **Stratified sampling**: Mantiene proporciones de clases en train/validation splits
- **Ponderación de clases**: Parámetro `scale_pos_weight` para penalizar errores en clases minoritarias
- **Métrica `mlogloss`**: Optimiza probabilidades multi-clase, no solo clasificación binaria

### 3. Entrenamiento Incremental (Out-of-Core)
El dataset CVEfixes contiene millones de registros. XGBoost soporta:
- **Carga por chunks**: Procesa datos en fragmentos sin cargar todo en memoria
- **Warm start**: Continúa entrenamiento desde un modelo previamente guardado (`xgb_model` parameter)
- **Actualización progresiva**: Ideal para pipelines donde llegan nuevos datos continuamente

### 4. Regularización Incorporada
Incluye regularización L1 (Lasso) y L2 (Ridge) que previenen overfitting, especialmente útil cuando las features de texto (HashingVectorizer) generan espacios de alta dimensionalidad.

### 5. Eficiencia Computacional
- **Tree method `hist`**: Usa histogramas para acelerar el entrenamiento (10x más rápido que método exacto)
- **Paralelización nativa**: Aprovecha múltiples núcleos (`n_jobs=-1`)
- **Soporte para matrices sparse**: Compatible con la salida de HashingVectorizer sin conversión a dense

## Alternativas Consideradas

| Modelo            | Motivo de descarte
|-------------------|----------------------------------------------
| Random Forest     | No soporta entrenamiento incremental nativo
| SVM               | No escala bien con datasets >100k muestras
| LightGBM          | Similar rendimiento, pero XGBoost tiene mejor documentación para clasificación multi-clase
| Redes Neuronales  | Requiere más datos y GPU para superar a XGBoost en datos tabulares (Se hará la comparación e implementación con CodeBERT ya que se cuenta con GPUs para testing)