import os
import pandas as pd
import joblib
from src.features.feature_engineering import add_feature_engineering
from src.features.selectors import combined_selection
from src.models.boosting import (
    get_lightgbm_model,
    get_xgboost_model,
    train_boosting)

# Cargar folds
def load_fold(path_train, path_val):
    train = pd.read_pickle(path_train)
    val = pd.read_pickle(path_val)
    return train, val


def train_with_folds(
    folds_dir="data/interim/folded/",
    output_dir="models/folds/",
    model_type="xgboost",
    top_k_features=15
):
    """
    Pipeline completo:
        - Feature engineering
        - Feature selection (por fold)
        - Entrenamiento boosting
        - Evaluación
        - Guardar modelos y métricas
    """

    os.makedirs(output_dir, exist_ok=True)

    results = []
    selected_features_all_folds = []

    print("\n Entrenamiento por folds\n")

    for i in range(1, 6):   # 5 folds
        print(f"\nFOLD {i}\n")

        train_path = os.path.join(folds_dir, f"fold_{i}_train.pkl")
        val_path   = os.path.join(folds_dir, f"fold_{i}_val.pkl")

        train_df, val_df = load_fold(train_path, val_path)

        # Separar features y target
        y_train = train_df["target"]
        y_val = val_df["target"]

        X_train = train_df.drop(columns=["target", "sample"])
        X_val   = val_df.drop(columns=["target", "sample"])


        # 1) Feature Engineering
      
        X_train_fe = add_feature_engineering(X_train)
        X_val_fe   = add_feature_engineering(X_val)

   
        # 2) Feature Selection (por fold)
    
        selected_features, ranking_table = combined_selection(
            X_train_fe,
            y_train,
            top_k=top_k_features
        )

        selected_features_all_folds.append(selected_features)

        X_train_sel = X_train_fe[selected_features]
        X_val_sel   = X_val_fe[selected_features]

  
        # 3) Select model
       
        if model_type.lower() == "lightgbm":
            model = get_lightgbm_model()
        else:
            model = get_xgboost_model()

     
        # 4) Train model
      
        model, metrics = train_boosting(
            model,
            X_train_sel,
            y_train,
            X_val_sel,
            y_val
        )

        print(f"Métricas fold {i}:", metrics)


        model_path = os.path.join(output_dir, f"model_fold_{i}.pkl")
        joblib.dump(model, model_path)


        results.append({
            "fold": i,
            "model_path": model_path,
            **metrics
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "fold_results.csv"), index=False)

    print("\n Entrenamiento por folds completado\n")
    return results_df, selected_features_all_folds


# Entrenamiento modelo final

def train_final_model(
    merged_path="data/interim/merged_raw.pkl",
    features_list=None,
    model_type="xgboost",
    output_path="models/best_model.pkl"
):
    """
    Entrena modelo final con TODO el dataset usando las features seleccionadas
    por promedio o intersección de folds.
    """

    print("\nEntrenamiento modelo final\n")

    df = pd.read_pickle(merged_path)

    y = df["target"]
    X = df.drop(columns=["target", "sample"])

    X_fe = add_feature_engineering(X)

    X_final = X_fe[features_list]

    if model_type.lower() == "lightgbm":
        model = get_lightgbm_model()
    else:
        model = get_xgboost_model()

    print(f"Entrenando modelo final con {len(features_list)} features")
    model.fit(X_final, y)

    joblib.dump(model, output_path)

    print(f"Modelo final guardado en: {output_path}")

    return model
