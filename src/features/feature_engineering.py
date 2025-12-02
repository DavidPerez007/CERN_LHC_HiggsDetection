import numpy as np
import pandas as pd

def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables físicas derivadas al DataFrame original.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1) ΔR entre leptones - ΔR = sqrt((Δη)^2 + (Δφ)^2)
    # ---------------------------------------------------------
    df["delta_eta_ll"] = abs(df["lep_eta_0"] - df["lep_eta_1"])
    df["delta_phi_ll_signed"] = df["lep_phi_0"] - df["lep_phi_1"]
    df["delta_phi_ll_signed"] = \
        df["delta_phi_ll_signed"].apply(lambda x: np.arctan2(np.sin(x), np.cos(x)))

    df["delta_R_ll"] = np.sqrt(df["delta_eta_ll"]**2 + df["delta_phi_ll_signed"]**2)

    # ---------------------------------------------------------
    # 2) pt_ratio: relación entre leptones
    # ---------------------------------------------------------
    df["pt_ratio"] = df["lep_pt_0"] / (df["lep_pt_1"] + 1e-6)

    # ---------------------------------------------------------
    # 3) Suma de pT leptónico
    # ---------------------------------------------------------
    df["pt_sum_ll"] = df["lep_pt_0"] + df["lep_pt_1"]

    # ---------------------------------------------------------
    # 4) Energía total de leptones
    # ---------------------------------------------------------
    df["E_sum_ll"] = df["lep_E_0"] + df["lep_E_1"]

    # ---------------------------------------------------------
    # 5) Energía total del sistema leptones + MET
    # ---------------------------------------------------------
    df["ptll_met"] = df["pTll"] + df["met_et"]

    # ---------------------------------------------------------
    # 6) Transverse mass MT (muy útil en H→WW) - MT = sqrt(2 * pTll * MET * (1 - cos(Δφ_ll_MET)))
    # ---------------------------------------------------------
    df["MT_ll_met"] = np.sqrt(
        2 * df["pTll"] * df["met_et"] *
        (1 - np.cos(df["dphi_ll_met"]))
    )

    # ---------------------------------------------------------
    # 7) Momento del dileptón rest-frame (proxy)
    # ---------------------------------------------------------
    df["pt_balance"] = abs(df["lep_pt_0"] - df["lep_pt_1"])

    # ---------------------------------------------------------
    # 8) cos(theta*) — angular del decay (aproximado)
    # ---------------------------------------------------------
    df["cos_theta_star"] = (df["lep_pt_0"] - df["lep_pt_1"]) / \
                           (df["lep_pt_0"] + df["lep_pt_1"] + 1e-6)

    # ---------------------------------------------------------
    # 9) Masa Transversa del sistema Higgs (curr_mt) - sqrt(2 * pT_ll * MET * (1 - cos(DeltaPhi_ll_met)))
    # ---------------------------------------------------------
    df["curr_mt"] = np.sqrt(
        2 * df["pTll"] * df["met_et"] * (1 - np.cos(df["dphi_ll_met"]))
    )

    # ---------------------------------------------------------
    # 10) Variable de Cluster (cluster_mass)
    # ---------------------------------------------------------
    df["cluster_mass"] = np.sqrt(df["mLL"]**2 + df["curr_mt"]**2)

    return df


def add_advanced_features(df):
    """
    Agrega features avanzadas para optimización de modelo.
    Incluye variables físicas de nivel superior para maximizar discriminación.
    
    Args:
        df: DataFrame con columnas originales y features básicas
        
    Returns:
        DataFrame con features avanzadas adicionales
    """
    df = df.copy()
    
    # 1. Energía transversa total (HT)
    df['HT'] = df['lep_pt_0'] + df['lep_pt_1'] + df['met_et']
    
    # 2. Ángulos azimutales normalizados
    df['dphi_ll_norm'] = np.abs(df['dphi_ll']) / np.pi
    df['dphi_ll_met_norm'] = np.abs(df['dphi_ll_met']) / np.pi
    
    # 3. Asimetría de momento transverso
    df['pt_asym'] = (df['lep_pt_0'] - df['lep_pt_1']) / (df['lep_pt_0'] + df['lep_pt_1'])
    df['pt_lead_met_ratio'] = df['lep_pt_0'] / (df['met_et'] + 1e-5)
    
    # 4. Masa transversa del sistema completo
    df['MT_total'] = np.sqrt(2 * df['pTll'] * df['met_et'] * (1 - np.cos(df['dphi_ll_met'])))
    
    # 5. Centrality (pseudorapidity cuadrática media)
    df['centrality'] = (df['lep_eta_0']**2 + df['lep_eta_1']**2) / 2
    
    # 6. Boost del sistema dileptónico
    df['ll_boost'] = np.sqrt(df['pTll']**2 + df['mLL']**2)
    
    # 7. Variables de aislamiento combinadas
    df['iso_sum'] = df['lep_ptcone30_0'] + df['lep_ptcone30_1']
    df['iso_prod'] = df['lep_ptcone30_0'] * df['lep_ptcone30_1']
    
    # 8. Energía invariante estimada del sistema ll
    df['E_inv'] = np.sqrt(df['mLL']**2 + df['pTll']**2)
    
    # 9. Separación angular ponderada por pT
    df['weighted_dR'] = df['delta_R_ll'] * df['pTll']
    
    # 10. Ratio masa invariante / MET
    df['mLL_met_ratio'] = df['mLL'] / (df['met_et'] + 1e-5)
    
    # 11. Colinearidad aproximada (útil para tau decays)
    df['collinearity'] = np.abs(df['lep_phi_0'] - df['lep_phi_1'])
    
    # 12. Energía missing transversa significance
    df['met_significance'] = df['met_et'] / np.sqrt(df['HT'] + 1e-5)
    
    return df


# Listas de features generadas
BASIC_ENGINEERED_FEATURES = [
    'delta_R_ll',
    'pt_ratio',
    'pt_sum_ll',
    'E_sum_ll',
    'ptll_met',
    'MT_ll_met',
    'pt_balance',
    'cos_theta_star',
    'curr_mt',           
    'cluster_mass'        
]

ADVANCED_FEATURES = [
    'HT',
    'dphi_ll_norm',
    'dphi_ll_met_norm',
    'pt_asym',
    'pt_lead_met_ratio',
    'MT_total',
    'centrality',
    'll_boost',
    'iso_sum',
    'iso_prod',
    'E_inv',
    'weighted_dR',
    'mLL_met_ratio',
    'collinearity',
    'met_significance'
]
