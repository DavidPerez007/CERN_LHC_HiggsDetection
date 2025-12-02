import os
import pandas as pd
from sklearn.utils import shuffle
from .load import load_csv, assign_target

def merge_and_save(
    higgs_path: str,
    diboson_path: str,
    output_path: str = "data/interim/merged_raw.pkl"
):
    """
    Une los datasets Higgs y DibosonWW.
    Crea la columna target correctamente.
    Mezcla los datos y los guarda en un .pkl.
    """
    print("Cargando datos filtrados reales...")
    higgs_df = load_csv(higgs_path)
    dib_df = load_csv(diboson_path)

    print(f"Higgs: {len(higgs_df)} eventos")
    print(f"DibosonWW: {len(dib_df)} eventos")

    # Asignar target ANTES de concatenar para evitar ambigüedad
    higgs_df["target"] = 1
    dib_df["target"] = 0

    # Unión
    df = pd.concat([higgs_df, dib_df], ignore_index=True)

    # Mezclar
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Asegurar la carpeta
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar
    df.to_pickle(output_path)
    print(f"Dataset combinado guardado en: {output_path}")

    return df


if __name__ == "__main__":
    merged_df = merge_and_save(
        "data/raw/datos_filtrados_Higgs.csv",
        "data/raw/datos_filtrados_DibosonWW.csv"
    )
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(merged_df.head())
