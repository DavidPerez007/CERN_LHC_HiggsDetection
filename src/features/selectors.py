import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


# ====================================================
# 1) MUTUAL INFORMATION
# ====================================================
def mutual_info_selection(X: pd.DataFrame, y: pd.Series, n_features: int = None):
    """
    Selección basada en Mutual Information.
    Retorna las variables ordenadas por importancia MI.
    """

    print("Calculando Mutual Information...")
    mi = mutual_info_classif(X, y, random_state=42)

    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    if n_features is None:
        return mi_series

    return mi_series.head(n_features).index.tolist()



# ====================================================
# 2) XGBoost Feature Importance
# ====================================================
def xgb_feature_importance(X: pd.DataFrame, y: pd.Series, n_features: int = None):
    """
    Entrena un XGBoost pequeño para obtener importancias.
    """

    print("Entrenando XGBoost para importancia de variables...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    if n_features is None:
        return importances

    return importances.head(n_features).index.tolist()



# ====================================================
# 3) Permutation Importance
# ====================================================
def permutation_feature_importance(X: pd.DataFrame, y: pd.Series, n_features: int = None):
    """
    Calcula importancia por permutación usando XGBoost como modelo base.
    """

    print("Calculando Permutation Importance...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_importances = pd.Series(
        result.importances_mean,
        index=X.columns
    ).sort_values(ascending=False)

    if n_features is None:
        return perm_importances

    return perm_importances.head(n_features).index.tolist()



# ====================================================
# 4) COMBINED SELECTION  (MI + XGB + Permutation)
# ====================================================
def combined_selection(X: pd.DataFrame, y: pd.Series, top_k: int = 15):
    """
    Selector combinado promediando rankings de:
    - Mutual Information
    - XGBoost importance
    - Permutation importance

    Retorna las TOP-K variables finales.
    """

    print("\n========== SELECCIÓN DE VARIABLES COMBINADA ==========\n")

    # Obtener rankings individuales
    mi_rank = mutual_info_selection(X, y)
    xgb_rank = xgb_feature_importance(X, y)
    perm_rank = permutation_feature_importance(X, y)

    # Convertir a ranking numérico
    def rank_series(s):
        return pd.Series(
            data=np.arange(1, len(s) + 1),
            index=s.index
        )

    mi_r = rank_series(mi_rank)
    xgb_r = rank_series(xgb_rank)
    perm_r = rank_series(perm_rank)

    # Alinear índices
    all_features = X.columns
    rankings = pd.DataFrame({
        "mi": mi_r.reindex(all_features),
        "xgb": xgb_r.reindex(all_features),
        "perm": perm_r.reindex(all_features)
    })

    # Promediar ranking
    rankings["mean_rank"] = rankings.mean(axis=1)

    # Seleccionar top-k
    rankings_sorted = rankings.sort_values(by="mean_rank")
    selected = rankings_sorted.head(top_k).index.tolist()

    print("\nVariables seleccionadas:")
    for f in selected:
        print(f"- {f}")

    return selected, rankings_sorted
