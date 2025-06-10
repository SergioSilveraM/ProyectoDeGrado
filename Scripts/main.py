import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from src.preprocessing import clean_input_data
from src.predictor import load_model, predict_probabilities
from src.weighting import apply_weighting
from src.fuzzy_engine import evaluar_batch_fuzzy
from src.xgb_wrapper import XGBWrapper

load_dotenv(dotenv_path="src/.env")
PATH_A = os.getenv('PATH_A')
PATH_B = os.getenv('PATH_B')
PAGOS_PATH = os.getenv('PAGOS_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
OUTPUT_PATH = None

VARIABLES_DIFUSAS = [
    "risk", "a_days", "total_balance_CDC", "total_balance_TDC",
    "overdue_balance", "overdue_installment", "prioritization",
    "payment_habit", "total_contacts", "total_agreements"
]

model = load_model(MODEL_PATH)

try:
    df_a = pd.read_excel(PATH_A)
except Exception as e:
    raise RuntimeError(f"Error leyendo PATH_A: {e}")

try:
    df_b = pd.read_excel(PATH_B)
except Exception as e:
    raise RuntimeError(f"Error leyendo PATH_B: {e}")

try:
    df_pagos = pd.read_excel(PAGOS_PATH)
except Exception as e:
    raise RuntimeError(f"Error leyendo archivo de pagos: {e}")

print("Preprocesando datos crudos...")
df_a = clean_input_data(df_a)

df_pagos["Cuenta"] = df_pagos["Cuenta"].astype(str)
df_a["Cuenta"] = df_a["Cuenta"].astype(str)

habito_cols = ["HabitoPago01", "HabitoPago03", "HabitoPago06", "HabitoPago11"]
df_a = df_a.merge(df_pagos[["Cuenta"] + habito_cols], on="Cuenta", how="left")

for col in habito_cols:
    df_a[col] = df_a[col].fillna(0).astype(int)

print("Generando predicciones...")
X_clean = df_a.copy()
df_proba = predict_probabilities(model, X_clean)
df_proba["score_ponderado"] = apply_weighting(df_proba)
df_proba["Cuenta"] = df_a["Cuenta"].astype(str)

df_b["Cuenta"] = df_b["Cuenta"].astype(str)
df_b = df_b.merge(df_proba[["Cuenta", "score_ponderado"]], on="Cuenta", how="left")
df_b = df_b.rename(columns={"score_ponderado": "risk"})

print("Ejecutando motor de inferencia difusa...")
df_b_fuzzy = df_b[VARIABLES_DIFUSAS].copy()
mask_valid = df_b_fuzzy.notnull().all(axis=1)
df_b_valid = df_b_fuzzy[mask_valid]
result_fuzzy = evaluar_batch_fuzzy(df_b_valid)

result_fuzzy_full = pd.DataFrame(index=df_b_fuzzy.index, columns=["valor_crisp", "segmento_fuzzy"])
result_fuzzy_full.update(result_fuzzy)

df_final = pd.concat([
    df_a.reset_index(drop=True),
    df_proba[[col for col in df_proba.columns if col.startswith("proba_") or col == "score_ponderado"]].reset_index(drop=True),
    result_fuzzy_full.reset_index(drop=True)
], axis=1)

if OUTPUT_PATH is None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fecha_hoy = datetime.now().strftime("%Y_%m_%d")
    OUTPUT_PATH = os.path.join(script_dir, f"Salida_Segm_FBV_{fecha_hoy}.xlsx")

print("Guardando resultado final...")
columns_to_keep = ['Cuenta', 'Num_documento', 'saldo_total', 'Rango', 'HabitoPago11', 'proba_clase_0', 'proba_clase_1', 'proba_clase_2', 'segmento_fuzzy' ]
df_final[columns_to_keep].to_excel(OUTPUT_PATH, index=False)
print(f"Archivo final generado: {OUTPUT_PATH}")
