import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_folds(
    merged_path: str = "data/interim/merged_raw.pkl",
    output_dir: str = "data/interim/folded/",
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Genera folds estratificados SIN preprocesamiento.
    Guarda cada fold por separado.
    """

    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Cargar dataset unificado
    print(f"Cargando dataset desde: {merged_path}")
    df = pd.read_pickle(merged_path)

    # Extraer features y target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Crear estructura de folds
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    print(f"\nGenerando {n_splits} folds estratificados...\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        train_path = os.path.join(output_dir, f"fold_{fold_idx}_train.pkl")
        val_path   = os.path.join(output_dir, f"fold_{fold_idx}_val.pkl")

        train_df.to_pickle(train_path)
        val_df.to_pickle(val_path)

        print(f"Fold {fold_idx}:")
        print(f"  - Train: {train_df.shape[0]} filas → {train_path}")
        print(f"  - Val:   {val_df.shape[0]} filas → {val_path}\n")

    print("✔ Folds generados correctamente.\n")
