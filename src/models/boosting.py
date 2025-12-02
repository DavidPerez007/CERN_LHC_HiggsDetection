import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =====================================================
#  AMS METRIC (High Energy Physics)
# =====================================================

def ams_score(y_true, y_pred, threshold=0.5):
    """
    Significancia aproximada AMS (Approximate Median Significance)
    
    Definida como:
    AMS = sqrt( 2 * ( (s+b)*ln(1+s/b) - s ) )
    
    Args:
        y_true: Array de etiquetas verdaderas (0 o 1)
        y_pred: Array de probabilidades predichas o etiquetas predichas
        threshold: Umbral para convertir probabilidades en clases
    
    Returns:
        float: Score AMS
    """
    # Si y_pred son probabilidades, convertir a clases
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    # Calcular TP y FP
    s = np.sum((y_true == 1) & (y_pred_binary == 1))  # True Positives (señal)
    b = np.sum((y_true == 0) & (y_pred_binary == 1))  # False Positives (fondo)
    
    # Evitar división por cero
    if b == 0 or s == 0:
        return 0.0
    
    # Fórmula AMS
    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    
    return ams



# =====================================================
#  BASE MODELS (Boosting)
# =====================================================

def get_xgboost_model():
    """
    XGBoost baseline optimizado para física.
    """
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )
    return model


def get_lightgbm_model():
    """
    LightGBM: rápido, eficiente y excelente baseline.
    """
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=40,
        reg_alpha=0.2,
        reg_lambda=0.4,
        random_state=42,
        objective='binary'
    )
    return model


def get_catboost_model():
    """
    CatBoost: ideal para distribuciones físicas,
    no requiere escalado y maneja no gaussianidades.
    """
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )
    return model



# =====================================================
#  TRAINING WRAPPER
# =====================================================

def train_boosting(model, X_train, y_train, X_val, y_val):
    """
    Entrena cualquier modelo boosting y regresa métricas.
    """

    print("Entrenando modelo...")
    model.fit(X_train, y_train)

    # Predicciones probabilísticas
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    # Métricas
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred_class)
    f1 = f1_score(y_val, y_pred_class)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_class).ravel()

    # AMS (señal = TP, fondo = FP)
    ams = ams_score(tp, fp)

    metrics = {
        "AUC": auc,
        "Accuracy": acc,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "AMS": ams
    }

    return model, metrics
