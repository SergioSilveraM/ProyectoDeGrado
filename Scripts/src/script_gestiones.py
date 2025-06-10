import pandas as pd
from src.conexion import Conexion
from datetime import datetime
from dateutil.relativedelta import relativedelta

def obtener_gestiones_y_compromisos(anio: int, mes: int, empresa: str = 'FBV') -> dict:
    """Consulta única de gestiones y compromisos para los últimos 3 meses (excluyendo el actual).
    Retorna un diccionario con DataFrames procesados por separado.
    """
    conn = Conexion.obtenerConexion()

    fecha_fin = datetime(anio, mes, 1)
    fecha_inicio = fecha_fin - relativedelta(months=3)

    query_gestiones = f"""
    SELECT 
        rut, 
        tipo_contacto, 
        fecha_respuesta, 
        TO_CHAR(fecha_respuesta, 'YYYY-MM') AS mes_gestion
    FROM gestiones_falabella
    WHERE fecha_respuesta >= '{fecha_inicio:%Y-%m-%d} 00:00:00'
      AND fecha_respuesta < '{fecha_fin:%Y-%m-%d} 00:00:00'
      AND empresa = '{empresa}'
      AND tipo_contacto != '(CP) Compromiso'
    """
    df_gestiones = pd.read_sql(query_gestiones, conn)
    df_gestiones['fecha_respuesta'] = pd.to_datetime(df_gestiones['fecha_respuesta'], errors='coerce')

    df_sin_comp = df_gestiones[df_gestiones['tipo_contacto'] != '(CP) Compromiso']
    df_gestiones_pivot = df_sin_comp.pivot_table(
        index=['rut', 'mes_gestion'],
        columns='tipo_contacto',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    df_gestiones_pivot['fecha_referencia'] = df_sin_comp.groupby(['rut', 'mes_gestion'])['fecha_respuesta'].transform('max')
    df_gestiones_pivot = df_gestiones_pivot.sort_values('fecha_referencia', ascending=False)
    df_gestiones_pivot = df_gestiones_pivot.drop_duplicates(subset='rut', keep='first')
    df_gestiones_pivot = df_gestiones_pivot.drop(columns=['fecha_referencia'])

    query_compromisos = f"""
        SELECT operacion, COUNT(*) AS TotalGestionesCompromiso, TO_CHAR(fecha_compromiso, 'YYYY-MM') AS mes_gestion
        FROM compromisos_falabella
        WHERE fecha_compromiso >= '{fecha_inicio:%Y-%m-%d} 00:00:00'
          AND fecha_compromiso < '{fecha_fin:%Y-%m-%d} 00:00:00'
          AND empresa = '{empresa}'
        GROUP BY operacion, TO_CHAR(fecha_compromiso, 'YYYY-MM')
    """
    df_compromisos = pd.read_sql(query_compromisos, conn)
    df_compromisos = df_compromisos.sort_values('mes_gestion', ascending=False)
    df_compromisos = df_compromisos.drop_duplicates(subset='operacion', keep='first')

    return {
        'gestiones': df_gestiones_pivot,
        'compromisos': df_compromisos
    }

def exportar_a_excel_completo(dataframes: dict, nombre_archivo: str):
    with pd.ExcelWriter(nombre_archivo) as writer:
        for nombre, df in dataframes.items():
            df.to_excel(writer, sheet_name=nombre[:31], index=False)
    print(f"Archivo Excel generado: {nombre_archivo}")

def exportar_hasta_hoy(empresa: str = 'FBV') -> dict:
    conn = Conexion.obtenerConexion()
    hoy = datetime.now()
    primer_dia = hoy.replace(day=1)

    query = f"""
        SELECT *, TO_CHAR(fecha_respuesta, 'YYYY-MM') AS mes_gestion
        FROM gestiones_falabella
        WHERE fecha_respuesta >= '{primer_dia:%Y-%m-%d} 00:00:00'
          AND fecha_respuesta <= '{hoy:%Y-%m-%d} 23:59:59'
          AND empresa = '{empresa}'
    """
    df = pd.read_sql(query, conn)
    df['fecha_respuesta'] = pd.to_datetime(df['fecha_respuesta'], errors='coerce')

    df_gest = df[df['tipo_contacto'] != '(CP) Compromiso']
    df_pivot = df_gest.pivot_table(index='rut', columns='tipo_contacto', aggfunc='size', fill_value=0).reset_index()

    df_comp_gestiones = df[df['tipo_contacto'] == '(CP) Compromiso']
    df_comp_gestiones = df_comp_gestiones.groupby('operacion').size().reset_index(name='TotalGestionesCompromiso')

    query_compromisos = f"""
        SELECT operacion, COUNT(*) AS TotalGestionesCompromiso
        FROM compromisos_falabella
        WHERE fecha_compromiso >= '{primer_dia:%Y-%m-%d} 00:00:00'
          AND fecha_compromiso <= '{hoy:%Y-%m-%d} 23:59:59'
          AND empresa = '{empresa}'
        GROUP BY operacion
    """
    df_compromisos_actuales = pd.read_sql(query_compromisos, conn)

    return {
        'gestiones_hoy': df_pivot,
        'compromisos_hoy': df_compromisos_actuales
    }

if __name__ == "__main__":
    anio = datetime.now().year
    mes = datetime.now().month
    fecha_str = datetime.now().strftime("%Y_%m_%d")

    actual = exportar_hasta_hoy()
    exportar_a_excel_completo(actual, f"gestiones_y_compromisos_hasta_hoy_{fecha_str}.xlsx")
