# 🔬 Proyecto Higgs Boson - Clasificación H→WW* vs DibosonWW

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proyecto de Machine Learning para clasificación de eventos de física de partículas del detector ATLAS en el LHC (CERN). El objetivo es distinguir eventos de desintegración del bosón de Higgs (H→WW*) del fondo de producción directa de dibosones (WW).

---

## 📊 **Dataset**

- **Eventos totales**: 26,277
  - **Señal (Higgs)**: 11,340 eventos (43.2%)
  - **Fondo (WW)**: 14,937 eventos (56.8%)
- **Features originales**: 35 variables cinemáticas y topológicas
- **Features ingenieradas**: +15 variables físicas derivadas
- **Fuente**: Datos simulados de ATLAS (Monte Carlo)

### Variables clave:
| Variable | Descripción | Importancia |
|----------|-------------|-------------|
| `mLL` | Masa invariante dileptónica | Alta |
| `pTll` | Momento transverso del sistema ll | Alta |
| `met_et` | Energía transversa faltante (MET) | Alta |
| `dphi_ll_met` | Ángulo azimutal entre ll y MET | Alta |
| `delta_R_ll` | Separación angular entre leptones | Media |
| `MT_ll_met` | Masa transversa ll-MET | Alta |

---

## 🚀 **Instalación**

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd Higgs
```

### 2. Crear entorno virtual
```bash
# Crear entorno
python -m venv venv

# Activar (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activar (Linux/Mac)
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Paquetes principales:**
- `pandas>=2.0` - Manipulación de datos
- `numpy>=1.24` - Computación numérica
- `scikit-learn>=1.3` - Machine Learning
- `xgboost>=3.1` - Gradient Boosting
- `lightgbm>=4.0` - Gradient Boosting alternativo
- `catboost>=1.2` - Gradient Boosting alternativo
- `optuna>=3.0` - Optimización bayesiana
- `shap>=0.40` - Interpretabilidad
- `matplotlib>=3.7` - Visualización
- `seaborn>=0.12` - Visualización estadística
- `jupyter>=1.0` - Notebooks

---

## 📁 **Estructura del Proyecto**

```
Higgs/
│
├── data/
│   ├── raw/                          # Datos originales
│   │   ├── datos_filtrados_Higgs.csv
│   │   ├── datos_filtrados_DibosonWW.csv
│   │   └── Higgs8TeVPipeline.ipynb
│   ├── interim/                      # Datos procesados intermedios
│   │   ├── merged_raw.pkl           # Dataset combinado
│   │   └── folded/                  # Folds para CV
│   └── processed/                    # Datos finales
│
├── notebooks/                        # Notebooks Jupyter
│   ├── 01_data_understanding.ipynb  # EDA y análisis exploratorio
│   ├── 02_pipeline.ipynb            # Pipeline completo de entrenamiento
│   ├── 03_resultados.ipynb          # Análisis de resultados y métricas
│   └── 04_mejora_modelo.ipynb       # Optimización e ingeniería de features
│
├── src/                              # Código fuente
│   ├── data/
│   │   ├── load.py                  # Carga de datos
│   │   └── merge_data.py            # Combinación de datasets
│   ├── features/
│   │   └── feature_engineering.py   # Ingeniería de features
│   ├── models/
│   │   ├── boosting.py              # Modelos de boosting
│   │   ├── trainer.py               # Entrenamiento
│   │   └── metrics.py               # Métricas personalizadas
│   ├── fold.split.py                # Estratificación de folds
│   └── selectors.py                 # Selección de features
│
├── models/                           # Modelos entrenados
│   ├── best_model.pkl               # Mejor modelo baseline
│   ├── best_model_optimized.pkl     # Modelo optimizado
│   ├── final_features.json          # Features seleccionadas
│   ├── enhanced_features.json       # Features extendidas
│   ├── best_hyperparams.json        # Hiperparámetros óptimos
│   └── folds/
│       └── fold_results.csv         # Resultados por fold
│
├── reports/                          # Reportes y figuras
│
├── requirements.txt                  # Dependencias Python
└── README.md                         # Este archivo
```

---

## 🎯 **Workflow del Proyecto**

### **Fase 1: Exploración de Datos** (`01_data_understanding.ipynb`)
- Carga y verificación del dataset
- Análisis de distribuciones (mLL, pTll, dphi_ll, etc.)
- Matriz de correlación optimizada
- Análisis de balance de clases
- Identificación de variables discriminantes
- Feature importance con Random Forest

**Salidas:**
- Visualizaciones de distribuciones
- Heatmap de correlaciones
- Conclusiones sobre estrategia de modelado

---

### **Fase 2: Pipeline de Entrenamiento** (`02_pipeline.ipynb`)
1. **Carga de datos**: `merged_raw.pkl`
2. **Feature engineering**: Variables físicas derivadas
3. **Generación de folds**: StratifiedKFold (5-fold)
4. **Entrenamiento con CV**: XGBoost, LightGBM, CatBoost
5. **Selección de features**: Importancia en ≥3 folds
6. **Modelo final**: Re-entrenamiento con features seleccionadas
7. **Validación**: Evaluación en conjunto completo

**Métricas calculadas:**
- AUC-ROC
- Accuracy
- F1-Score
- AMS (Approximate Median Significance)

**Salidas:**
- `models/best_model.pkl`
- `models/final_features.json`
- `models/folds/fold_results.csv`

---

### **Fase 3: Análisis de Resultados** (`03_resultados.ipynb`)
- Métricas promedio de CV
- Gráficas de métricas por fold
- Curva ROC del modelo final
- Matriz de confusión
- Importancia de variables (feature importance)
- Análisis SHAP (interpretabilidad)
- Conclusiones automatizadas

**Visualizaciones:**
- AUC, AMS, Accuracy, F1 por fold
- Curva ROC con AUC
- Matriz de confusión
- Top 20 features más importantes
- SHAP summary plot
- SHAP bar plot

---

### **Fase 4: Optimización de Modelo** (`04_mejora_modelo.ipynb`)

#### **Estrategias implementadas:**

1. **Optimización Bayesiana (Optuna)**
   - 50 trials en espacio de hiperparámetros
   - Validación cruzada 5-fold
   - Búsqueda en 9 parámetros clave

2. **Feature Engineering Avanzado**
   - 15 nuevas variables físicas
   - Variables derivadas: HT, centrality, ll_boost, etc.
   - Evaluación individual de importancia

3. **Análisis Comparativo**
   - 5 estrategias probadas
   - Selección automática del mejor modelo
   - Comparación multi-métrica

#### **Modelos comparados:**
| Estrategia | Descripción | Features |
|------------|-------------|----------|
| Baseline | Modelo original | 15 |
| Optuna + 8 Features | Hiperparámetros optimizados + top 8 | 23 |
| Solo Hyperparams | Optimización sin nuevas features | 15 |
| Original + Top 3 | Conservador | 18 |
| Original + Top 5 | Híbrido | 20 |

**Salidas:**
- `models/best_model_optimized.pkl`
- `models/enhanced_features.json`
- `models/best_hyperparams.json`
- Tabla comparativa de estrategias

---

## 📈 **Resultados**

### **Baseline (Modelo Original)**
```
AUC:      0.8651
Accuracy: 0.7777
F1-Score: 0.7520
AMS:      117.60
Features: 15
```

### **Mejor Estrategia** (identificada automáticamente)
- Ver output de `04_mejora_modelo.ipynb` celda final
- La mejor estrategia se selecciona dinámicamente por AUC

---

## 🔧 **Uso**

### **Entrenamiento desde cero**

```python
# 1. Preparar datos
from src.data.merge_data import merge_and_save
merge_and_save()

# 2. Ejecutar pipeline completo
# Ejecutar notebook: 02_pipeline.ipynb

# 3. Analizar resultados
# Ejecutar notebook: 03_resultados.ipynb

# 4. Optimizar modelo (opcional)
# Ejecutar notebook: 04_mejora_modelo.ipynb
```

### **Predicción con modelo entrenado**

```python
import joblib
import pandas as pd
from src.features.feature_engineering import add_feature_engineering

# Cargar modelo
model = joblib.load("models/best_model.pkl")

# Cargar features
with open("models/final_features.json", "r") as f:
    features = json.load(f)

# Preparar datos nuevos
df_new = pd.read_csv("new_data.csv")
df_new = add_feature_engineering(df_new)

# Predecir
X_new = df_new[features]
y_pred = model.predict_proba(X_new)[:, 1]  # Probabilidad de Higgs

# Clasificar
threshold = 0.5
predictions = (y_pred >= threshold).astype(int)
```

### **Optimización de hiperparámetros**

```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # ... más parámetros
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Mejor AUC: {study.best_value:.4f}")
```

---

## 📊 **Métricas de Evaluación**

### **AUC-ROC** (Area Under the Curve)
- Rango: [0.5, 1.0]
- **Interpretación**: Capacidad de discriminar entre Higgs y WW
- **Objetivo**: ≥ 0.85 (excelente)

### **Accuracy** (Precisión Global)
- Rango: [0.0, 1.0]
- **Interpretación**: % de eventos correctamente clasificados
- **Objetivo**: ≥ 0.75

### **F1-Score** (Media armónica Precisión-Recall)
- Rango: [0.0, 1.0]
- **Interpretación**: Balance entre precisión y sensibilidad
- **Objetivo**: ≥ 0.70

### **AMS** (Approximate Median Significance)
- Rango: [0, ∞)
- **Fórmula**: AMS = √(2·((s+b)·ln(1+s/b) - s))
- **Interpretación**: Significancia estadística en física de partículas
- **Objetivo**: Maximizar (típicamente > 50)

---

## 🧠 **Feature Engineering**

### **Features básicas** (implementadas en `feature_engineering.py`)

```python
def add_feature_engineering(df):
    # Masa transversa ll-MET
    df['MT_ll_met'] = sqrt(2·pTll·MET·(1-cos(Δφ)))
    
    # Separación angular entre leptones
    df['delta_R_ll'] = sqrt(Δη² + Δφ²)
    
    # Ratio de momento transverso
    df['pt_ratio'] = lep_pt_0 / lep_pt_1
    
    # ... +8 variables más
    return df
```

### **Features avanzadas** (para optimización)

```python
def add_advanced_features(df):
    # Energía transversa total
    df['HT'] = lep_pt_0 + lep_pt_1 + MET
    
    # Centrality (posición en detector)
    df['centrality'] = (lep_eta_0² + lep_eta_1²) / 2
    
    # Boost del sistema dileptónico
    df['ll_boost'] = sqrt(pTll² + mLL²)
    
    # ... +12 variables más
    return df
```

---

## 🔍 **Interpretabilidad**

### **SHAP (SHapley Additive exPlanations)**

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values, X, plot_type="dot")

# Bar plot
shap.summary_plot(shap_values, X, plot_type="bar")
```

**Interpretación:**
- **Rojo**: Valores altos de la feature
- **Azul**: Valores bajos de la feature
- **Eje X**: Impacto en la predicción (positivo = más probable Higgs)

---

## 🛠️ **Configuración Avanzada**

### **Hiperparámetros XGBoost recomendados**

```python
params = {
    'n_estimators': 500,          # Número de árboles
    'max_depth': 6,               # Profundidad máxima
    'learning_rate': 0.1,         # Tasa de aprendizaje
    'subsample': 0.8,             # Fracción de datos por árbol
    'colsample_bytree': 0.8,      # Fracción de features por árbol
    'min_child_weight': 3,        # Peso mínimo por hoja
    'gamma': 0.1,                 # Regularización mínima para split
    'reg_alpha': 0.01,            # L1 regularization
    'reg_lambda': 1.0,            # L2 regularization
    'random_state': 42,
    'eval_metric': 'auc'
}
```

### **Validación Cruzada Estratificada**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    # ... evaluar
```

---

## 🐛 **Troubleshooting**

### **Error: ModuleNotFoundError: No module named 'src'**
```python
# Agregar al inicio del notebook
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
```

### **Error: KeyError en features**
```bash
# Verificar que las features existen
python -c "import pandas as pd; df = pd.read_pickle('data/interim/merged_raw.pkl'); print(df.columns.tolist())"
```

### **Error: ams_score con arrays**
```python
# La función actualizada acepta arrays completos
from src.models.boosting import ams_score
score = ams_score(y_true, y_pred_proba)  # y_pred_proba son probabilidades
```

### **Error: Optuna no encuentra mejor trial**
```python
# Aumentar número de trials
study.optimize(objective, n_trials=100)  # En lugar de 50
```

---

## 📚 **Referencias**

### **Física**
- [ATLAS Collaboration](https://atlas.cern/)
- [Higgs Discovery Paper (2012)](https://www.sciencedirect.com/science/article/pii/S037026931200857X)
- [H→WW* Analysis](https://arxiv.org/abs/1412.2641)

### **Machine Learning**
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Paper](https://arxiv.org/abs/1907.10902)
- [SHAP Documentation](https://shap.readthedocs.io/)

### **Kaggle Competition**
- [Higgs Boson Challenge (2014)](https://www.kaggle.com/c/higgs-boson)
- [AMS Metric Explanation](https://www.kaggle.com/c/higgs-boson/overview/evaluation)

---

## 🤝 **Contribuciones**

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -am 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

---

## 📝 **TODO / Mejoras Futuras**

- [ ] Implementar ensemble stacking (XGB + LGBM + CatBoost)
- [ ] Agregar calibración de probabilidades (Platt scaling)
- [ ] Optimización de threshold para maximizar AMS
- [ ] Implementar data augmentation (SMOTE)
- [ ] Agregar modelo de Deep Learning (MLP)
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard interactivo con Streamlit/Dash
- [ ] Monitoreo de data drift en producción
- [ ] Tests unitarios para módulos críticos
- [ ] Documentación de API con Sphinx

---

## 👨‍💻 **Autor**

**Tu Nombre**
- Email: tu.email@example.com
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

---

## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 **Agradecimientos**

- **CERN/ATLAS** por los datos simulados
- **Kaggle** por la competencia Higgs Boson Challenge
- **XGBoost Team** por la excelente herramienta
- **Optuna** por la optimización bayesiana eficiente
- **Comunidad de ML en Física de Partículas**

---

## 📞 **Soporte**

Si encuentras algún problema o tienes preguntas:

1. Revisa la sección [Troubleshooting](#troubleshooting)
2. Busca en [Issues](https://github.com/tu-usuario/Higgs/issues)
3. Abre un nuevo Issue con detalles del problema
4. Contacta al autor por email

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**

---

*Última actualización: 30 de noviembre de 2025*
