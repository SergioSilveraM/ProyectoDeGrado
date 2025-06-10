import pandas as pd
import joblib
import os
from src.xgb_wrapper import XGBWrapper

def load_model(path: str) -> XGBWrapper:
    loaded_obj = joblib.load(path)
    if isinstance(loaded_obj, XGBWrapper):
        return loaded_obj
    elif isinstance(loaded_obj, dict):
        return XGBWrapper.load(path)
    else:
        raise ValueError("El archivo no contiene un modelo XGBWrapper válido.")

def prepare_numerics(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                    .replace(r"^[-\\s]*$", "0", regex=True)
                    .astype(float)
                )
    return df


def validate_columns(df: pd.DataFrame, model: XGBWrapper):
    """Verifica que las columnas coincidan exactamente con las del modelo."""
    expected = set(model.numeric_features + model.categorical_features)
    found = set(df.columns)
    if not expected.issubset(found):
        missing = expected - found
        extra = found - expected
        msg = "Error en columnas:\n"
        if missing:
            msg += f" - Faltan: {missing}\n"
        if extra:
            msg += f" - Sobrantes: {extra}\n"
        raise ValueError(msg)

def predict_probabilities(model: XGBWrapper, df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df, model)
    df = prepare_numerics(df, model.numeric_features)
    y_proba = model.predict_proba(df)
    proba_df = pd.DataFrame(y_proba, columns=[f"proba_clase_{i}" for i in range(y_proba.shape[1])])
    return pd.concat([df.reset_index(drop=True), proba_df], axis=1)
