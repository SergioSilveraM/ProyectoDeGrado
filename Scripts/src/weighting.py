import pandas as pd

# Clase 0 (alto riesgo): 0.01
# Clase 1 (riesgo medio): 0.5
# Clase 2 (riesgo bajo): 1.0
DEFAULT_WEIGHTS = [0.01, 0.5, 1.0]

def apply_weighting(proba_df: pd.DataFrame, weights: list[float] = DEFAULT_WEIGHTS) -> pd.Series:
    """
    Aplica ponderación suave al vector de salida del modelo para convertirlo en un score continuo.
    Entre 0.01 (default alto) y 1.0 (riesgo bajo).
    """
    proba_cols = [col for col in proba_df.columns if col.startswith("proba_clase_")]
    
    if len(proba_cols) != len(weights):
        raise ValueError(f"Cantidad de pesos ({len(weights)}) no coincide con clases detectadas ({len(proba_cols)}).")

    proba_matrix = proba_df[proba_cols].values
    weighted_scores = proba_matrix @ weights  # producto escalar

    return pd.Series(weighted_scores, name="score_ponderado")
