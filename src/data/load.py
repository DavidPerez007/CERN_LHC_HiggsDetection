import os
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Carga un CSV y verifica columnas esperadas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)

    expected_cols = [
        "trigE", "trigM",
        "lep_n", "jet_n",
        "met_et", "met_phi",
        "lep_pt_0", "lep_pt_1",
        "lep_eta_0", "lep_eta_1",
        "lep_E_0", "lep_E_1",
        "lep_phi_0", "lep_phi_1",
        "lep_charge_0", "lep_charge_1",
        "lep_type_0", "lep_type_1",
        "lep_ptcone30_0", "lep_ptcone30_1",
        "jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_MV2c10",
        "sample",   
        "mLL", "pTll", "dphi_ll", "dphi_ll_met"
    ]

    # Comprobación
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("Advertencia: faltan columnas:", missing)
    else:
        print("Todas las columnas físicas esperadas están presentes.")

    return df


def assign_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la columna 'sample' en un target binario:
        Higgs = 1
        DibosonWW = 0
    """
    df = df.copy()

    if "sample" not in df.columns:
        raise ValueError("La columna 'sample' no existe en el dataset.")

    df["target"] = df["sample"].apply(lambda x: 1 if "Higgs" in x else 0)

    return df
