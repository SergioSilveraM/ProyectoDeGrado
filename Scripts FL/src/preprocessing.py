import pandas as pd
import re
from datetime import datetime
from src.script_gestiones import obtener_gestiones_y_compromisos

def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['SaldoCliente'] = df.groupby('Num_documento')['saldo_total'].transform('sum')
    filtros = (df['GRUPO'] != 6) & (df['Rango'] != '(01) Al Día')
    df = df[filtros]
    df.loc[df['Oficina'] != "Digital", 'Oficina'] = "Presencial"
    df['Tipo_Normalizacion'] = df['Tipo_Normalizacion'].fillna("No Aplicado")
    df['Tipo_Linea_1'] = df['Tipo_Linea_1'].fillna("Sin Linea")
    df['Ultimo_Canal_Gestion'] = df['Ultimo_Canal_Gestion'].fillna("Sin Gestion")
    df['CR_Ultima_Gestion_Cliente'] = df['CR_Ultima_Gestion_Cliente'].fillna("Sin Gestion")
    df['CR_Mejor_Gestion_Cliente'] = df['CR_Mejor_Gestion_Cliente'].fillna("Sin Gestion")

    def map_canal(valor):
        if pd.isna(valor):
            return "Sin Gestion"
        valor = str(valor).lower()
        if "whatsapp" in valor:
            return "WhatsApp"
        if "mensaje de texto" in valor:
            return "Masivo"
        if re.search(r'env[ií]?o.*mail', valor):
            return "Masivo"
        return "Llamada"

    df['Ultimo_Canal_Gestion'] = df['Ultimo_Canal_Gestion'].apply(map_canal)

    def map_cr_ultima(valor):
        if pd.isna(valor):
            return "Sin Gestion"
        valor = str(valor).strip().lower()
        if valor in [
            "mensaje enviado", "email enviado"
        ]:
            return "Masivo"
        if valor in [
            "promesa de pago", "refuerzo promesa",
            "contacto con titular sin promesa de pag", "pago por confirmar",
            "proceso de rediferido ( ofrecimiento)", "tramite modificado",
            "contactado", "tramite rediferido",
            "posible suplantación", "cliente pago",
            "posible suplantación"
        ]:
            return "Contacto Directo"
        return "No contactado"

    df['CR_Ultima_Gestion_Cliente'] = df['CR_Ultima_Gestion_Cliente'].apply(map_cr_ultima)

    def map_cr_mejor(valor):
        if pd.isna(valor):
            return "Sin Gestion"
        valor = str(valor).strip().lower()
        if valor in [
            "mensaje enviado", "email enviado"
        ]:
            return "Masivo"
        if valor in [
            "promesa de pago", "refuerzo promesa",
            "contacto con titular sin promesa de pag", "pago por confirmar",
            "proceso de rediferido ( ofrecimiento)", "tramite modificado",
            "contactado", "tramite rediferido",
            "posible suplantación", "cliente pago",
            "posible suplantación"
        ]:
            return "Contacto Directo"
        return "No contactado"

    df['CR_Mejor_Gestion_Cliente'] = df['CR_Mejor_Gestion_Cliente'].apply(map_cr_mejor)

    fecha_corte = pd.Timestamp(datetime.now().year, datetime.now().month, 1) - pd.Timedelta(days=1)
    df['Ultima_Gestion_Cliente'] = pd.to_datetime(df['Ultima_Gestion_Cliente'], errors='coerce')
    df['DiasUltimaGestion'] = (fecha_corte - df['Ultima_Gestion_Cliente']).dt.days
    df['DiasUltimaGestion'] = df['DiasUltimaGestion'].fillna(99)

    df['Fecha_Mejor_Gestion'] = pd.to_datetime(df['Fecha_Mejor_Gestion'], errors='coerce')
    df['DiasMejorGestion'] = (fecha_corte - df['Fecha_Mejor_Gestion']).dt.days
    df['DiasMejorGestion'] = df['DiasMejorGestion'].fillna(99)

    df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'], format='%Y%m%d', errors='coerce')
    df['antiguedad'] = (fecha_corte - df['fecha_apertura']).dt.days / 365
    df.drop(columns=["fecha_apertura"], inplace=True)

    mapa_rango = {
        "(01) Al Día": "0 a 1",
        "(02) 1 a 30": "1 a 2",
        "(03) 31 a 60": "2 a 3",
        "(04) 61 a 90": "3 a 4",
        "(05) 91 a 120": "4 a 5",
        "(06) 121 a 150": "5 a 6",
        "(07) 151 a 180": "6 a 7"
    }
    df['NumeroCuotas'] = df['Rango'].map(mapa_rango).fillna("Sin Datos")

    df['ParticipacionProducto'] = df['saldo_total'] / df['SaldoCliente']

    cuotas_cols = [f"Cuota_Vcda{i}" for i in range(1, 13) if f"Cuota_Vcda{i}" in df.columns]
    df['SaldoVencido'] = df[cuotas_cols].sum(axis=1)

    df['IndiceMora'] = df['SaldoVencido'] / df['saldo_total']

    def calcular_pgmin(row):
        suma_cuotas = row[cuotas_cols].sum()
        if suma_cuotas == 0:
            return 10000
        for col in cuotas_cols:
            if row[col] > 10000:
                return row[col]
        return 10000

    df['PgMin'] = df.apply(calcular_pgmin, axis=1)

    hoy = datetime.now()
    gest_data = obtener_gestiones_y_compromisos(anio=hoy.year, mes=hoy.month)
    df_gestiones = gest_data['gestiones']
    df_compromisos = gest_data['compromisos']

    df = df[df['Num_documento'].isin(df_gestiones['rut'])]
    df = df.merge(df_gestiones, left_on="Num_documento", right_on="rut", how="left")
    df.drop(columns=['rut'], inplace=True)

    df = df.merge(df_compromisos, left_on="Cuenta", right_on="operacion", how="left")
    df.drop(columns=["operacion"], inplace=True)
    print(df.columns.to_list())
    df['TotalGestionesCompromiso'] = df['totalgestionescompromiso'].fillna(0).astype(int)

    masivos = ['(EML) Mail Enviado']
    cd = ['(CD) Contacto Directo']
    ci = ['(CI) Contacto Indirecto']
    nc = ['(CF) Contacto. Fallecido', '(NE) No Existe Fono', '(NR) No Responde',
          '(NS) Sin Servicio', '(NU) No Ubicable', '(SC) Sin Contacto']

    for col in masivos + cd + ci + nc:
        if col not in df.columns:
            df[col] = 0

    df['TotalGestionesMasivas'] = df[masivos].sum(axis=1)
    df['TotalGestionesCD'] = df[cd].sum(axis=1)
    df['TotalGestionesCI'] = df[ci].sum(axis=1)
    df['TotalGestionesNC'] = df[nc].sum(axis=1)

    df['TotalGestiones'] = (
        df['TotalGestionesMasivas'] +
        df['TotalGestionesCD'] +
        df['TotalGestionesCI'] +
        df['TotalGestionesCompromiso'] +
        df['TotalGestionesNC']
    )

    df['PorcentajeGestionesEfectivas'] = (
        (df['TotalGestionesCD'] + df['TotalGestionesCompromiso']) /
        df['TotalGestiones'].replace(0, pd.NA)
    ).fillna(0)

    return df
