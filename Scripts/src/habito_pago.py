import pandas as pd
from datetime import datetime

def agregar_habitos_pago(df_principal: pd.DataFrame, path_archivo_pago: str) -> pd.DataFrame:
    """Agrega columnas HabitoPago01, HabitoPago03, HabitoPago06 y HabitoPago11 al DataFrame principal."""

    df_pagos = pd.read_excel(path_archivo_pago)

    df_pagos.columns = (
        df_pagos.columns
        .str.strip()
        .str.upper()
        .str.replace("Á", "A").str.replace("É", "E").str.replace("Í", "I").str.replace("Ó", "O").str.replace("Ú", "U")
    )

    fecha_cols = [col for col in df_pagos.columns if col.startswith("FECHA PAGO")]
    fecha_cols = list(dict.fromkeys(fecha_cols))  
    fecha_cols = fecha_cols[-12:]  

    for col in fecha_cols:
        df_pagos[col] = pd.to_datetime(df_pagos[col], errors='coerce')

    def calcular_habito(fechas):
        hoy = datetime.now()
        base = hoy.replace(day=1) - pd.DateOffset(months=1)
        fechas_validas = [f for f in fechas if pd.notna(f) and f < hoy]

        habito_01 = int(any(base - pd.DateOffset(months=1) <= f < base for f in fechas_validas))
        habito_03 = sum(f >= base - pd.DateOffset(months=3) for f in fechas_validas)
        habito_06 = sum(f >= base - pd.DateOffset(months=6) for f in fechas_validas)
        habito_11 = sum(f >= base - pd.DateOffset(months=12) for f in fechas_validas)
        return pd.Series([habito_11, habito_06, habito_03, habito_01])

    df_pagos[['HabitoPago11', 'HabitoPago06', 'HabitoPago03', 'HabitoPago01']] = df_pagos[fecha_cols].apply(calcular_habito, axis=1)

    df_pagos['CUENTA'] = df_pagos['CUENTA'].astype(str)
    df_principal['Cuenta'] = df_principal['Cuenta'].astype(str)

    for col in ['HabitoPago11', 'HabitoPago06', 'HabitoPago03', 'HabitoPago01']:
        if col in df_principal.columns:
            df_principal.drop(columns=[col], inplace=True)

    df_merge = df_pagos[['CUENTA', 'HabitoPago11', 'HabitoPago06', 'HabitoPago03', 'HabitoPago01']]
    df_principal = df_principal.merge(df_merge, left_on='Cuenta', right_on='CUENTA', how='left')
    df_principal.drop(columns=['CUENTA'], inplace=True)

    for col in ['HabitoPago11', 'HabitoPago06', 'HabitoPago03', 'HabitoPago01']:
        df_principal[col] = df_principal[col].fillna(0).astype(int)

    return df_principal
