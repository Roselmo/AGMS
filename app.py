# ==============================================================================
# APP: Dashboard AGMS – Ventas (Pestaña 1 completa y funcional)
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# ---- Plotly (con chequeo) ----
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="📊", layout="wide")

LOGO_CANDIDATES = ["ag2.jpg", "logo.png", "AGMS_logo.jpg", "ag_logo.jpg"]
logo_path = next((p for p in LOGO_CANDIDATES if os.path.exists(p)), None)

l, m, r = st.columns([1, 2, 1])
with l:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with m:
    st.title("Dashboard AGMS: Ventas (Pestaña 1)")
st.markdown("---")

if not PLOTLY_OK:
    st.error("No se encontró **plotly**. Agrega `plotly` a tu requirements.txt y vuelve a desplegar.")
    st.stop()

# ==============================================================================
# UTILIDADES ROBUSTAS (dinero y fechas)
# ==============================================================================

def parse_money(x):
    """
    Convierte montos con formato mixto:
    - '1.234.567,89' -> 1234567.89
    - '1,234,567.89' -> 1234567.89
    - '1234567,89'   -> 1234567.89
    - '1234567.89'   -> 1234567.89
    - '12.345.678'   -> 12345678
    - '12,345,678'   -> 12345678
    Devuelve np.nan si no se puede convertir o si dice 'No aplica'.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('$', '').replace(' ', '')
    if s == "" or s.lower().startswith("no"):
        return np.nan

    has_comma = ',' in s
    has_dot = '.' in s

    if has_comma and has_dot:
        # el separador decimal suele ser el que está más a la derecha
        if s.rfind(',') > s.rfind('.'):
            # '1.234.567,89' => quita puntos (miles) y coma -> punto
            s = s.replace('.', '').replace(',', '.')
        else:
            # '1,234,567.89' => quita comas (miles)
            s = s.replace(',', '')
    else:
        if has_comma:
            parts = s.split(',')
            if len(parts[-1]) in (1, 2):
                s = s.replace('.', '')  # por si hay puntos de miles
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')  # coma como miles
        elif has_dot:
            parts = s.split('.')
            if len(parts[-1]) in (1, 2):
                s = s.replace(',', '')
            else:
                s = s.replace('.', '')

    try:
        return float(s)
    except Exception:
        return np.nan

def parse_col_money_safe(series):
    if series is None:
        return pd.Series(np.nan)
    return series.apply(parse_money)

def parse_fecha_col(col):
    """
    Convierte una columna heterogénea a datetime soportando:
    - Textos (dd/mm/yyyy, yyyy-mm-dd, etc.)
    - Seriales de Excel (días desde 1899-12-30)
    """
    s = pd.Series(col)

    # 1) Texto (día-primero)
    d1 = pd.to_datetime(s.astype(str), dayfirst=True, errors='coerce')

    # 2) Texto (año-primero)
    d2 = pd.to_datetime(s.astype(str), yearfirst=True, errors='coerce')

    # 3) Serial Excel SOLO para valores claramente numéricos
    def _num_or_nan(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)) and np.isfinite(x):
            return float(x)
        xs = str(x).strip()
        if xs.replace('.', '', 1).isdigit():
            try:
                return float(xs)
            except Exception:
                return np.nan
        return np.nan

    nums = s.map(_num_or_nan)
    d3 = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    mask = nums.notna()
    if mask.any():
        d3.loc[mask] = pd.to_datetime(nums.loc[mask].astype('float64'),
                                      unit='d', origin='1899-12-30', errors='coerce')

    return d1.combine_first(d2).combine_first(d3)

# ==============================================================================
# CARGA DE DATOS (solo Ventas y Productos, necesarios para Pestaña 1)
# ==============================================================================
@st.cache_data
def load_data_tab1():
    file_path = 'DB_AGMS.xlsx'
    try:
        df_ventas    = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        try:
            df_productos = pd.read_excel(file_path, sheet_name='Productos')
        except Exception:
            df_productos = pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None

    # ---- Ventas: fechas y totales robustos
    if 'FECHA VENTA' not in df_ventas.columns:
        st.error("No se encuentra la columna 'FECHA VENTA' en la hoja Ventas.")
        return None, None

    df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas['FECHA VENTA'])
    df_ventas = df_ventas.dropna(subset=['FECHA VENTA'])

    # Total_num consistente
    if 'Total' in df_ventas.columns:
        df_ventas['Total_num'] = parse_col_money_safe(df_ventas['Total']).fillna(0.0)
    else:
        df_ventas['Total_num'] = 0.0

    # Cliente y Producto base
    if 'Cliente/Empresa' in df_ventas.columns:
        df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

    if 'Producto_Nombre' not in df_ventas.columns:
        if 'Producto' in df_ventas.columns:
            df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).str.split(' - ').str[0].str.strip()
        else:
            df_ventas['Producto_Nombre'] = "(DESCONOCIDO)"

    # Derivadas temporales útiles
    df_ventas['Año']      = df_ventas['FECHA VENTA'].dt.year
    df_ventas['Mes_P']    = df_ventas['FECHA VENTA'].dt.to_period('M')
    df_ventas['Semana_P'] = df_ventas['FECHA VENTA'].dt.to_period('W')
    df_ventas['Día']      = df_ventas['FECHA VENTA'].dt.date
    df_ventas['Mes']      = df_ventas['Mes_P'].astype(str)     # 'YYYY-MM'
    df_ventas['Semana']   = df_ventas['Semana_P'].astype(str)  # 'YYYY-Wxx'
    df_ventas['DiaSemana'] = df_ventas['FECHA VENTA'].dt.day_name()

    # ---- Productos: normaliza campos para filtros
    if df_productos is not None and not df_productos.empty:
        ren = {
            'LISTA PRODUCTOS': 'Producto_Nombre',
            'TIPO DE PIEL': 'Tipo_Piel',
            'CONDICION': 'Condicion',
            'PROFESIONA o RETAIL': 'Canal',
            'PROFESIONAL o RETAIL': 'Canal',
            'PRECIO PRO': 'Precio_Pro',
            'PRECIO RETAIL': 'Precio_Retail',
            'Marca': 'Marca'
        }
        df_productos = df_productos.rename(columns={c: ren.get(c, c) for c in df_productos.columns}).copy()
        if 'Producto_Nombre' in df_productos.columns:
            df_productos['Producto_Nombre'] = df_productos['Producto_Nombre'].astype(str).str.strip()
    else:
        df_productos = pd.DataFrame()

    return df_ventas, df_productos

df_ventas, df_productos = load_data_tab1()
if df_ventas is None:
    st.stop()

# ==============================================================================
# TABS (solo trabajamos la primera)
# ==============================================================================
tab1, = st.tabs(["Análisis de Ventas"])

# ---------------------------------------------------------------------------------
# TAB 1: ANÁLISIS DE VENTAS (completa)
# ---------------------------------------------------------------------------------
with tab1:
    st.header("Análisis General de Ventas")

    dfv = df_ventas.copy()

    # ========================= KPIs Generales =========================
    total_ventas = float(dfv["Total_num"].sum())
    total_transacciones = len(dfv)
    clientes_unicos = dfv["Cliente/Empresa"].nunique() if "Cliente/Empresa" in dfv.columns else 0
    ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventas Totales (histórico)", f"${total_ventas:,.0f}")
    c2.metric("Transacciones", f"{total_transacciones:,}")
    c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
    st.markdown("---")

    # ========================= Subpestañas internas =========================
    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Resumen", "Productos", "Clientes", "Mapa de calor"])

    # -------- Resumen (Evolución + Consulta puntual)
    with tab_r1:
        st.subheader("Evolución de Ventas (rango)")

        # Rango de fechas (por defecto todo el histórico)
        min_date = dfv['FECHA VENTA'].min().date()
        max_date = dfv['FECHA VENTA'].max().date()
        rango = st.date_input(
            "Selecciona rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="t1_rango"
        )

        # Filtrado seguro por rango
        if isinstance(rango, (list, tuple)) and len(rango) == 2:
            d_ini = pd.to_datetime(rango[0])
            d_fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            dfv_rango = dfv[(dfv['FECHA VENTA'] >= d_ini) & (dfv['FECHA VENTA'] <= d_fin)]
        else:
            dfv_rango = dfv

        # Serie diaria con resample (rellena días sin ventas con 0)
        if dfv_rango.empty:
            st.info("No hay ventas en el rango seleccionado.")
        else:
            # nos aseguramos de cubrir todo el rango aún si no hay ventas algún día
            full_idx = pd.date_range(start=dfv_rango['FECHA VENTA'].min().normalize(),
                                     end=dfv_rango['FECHA VENTA'].max().normalize(),
                                     freq='D')
            serie = (dfv_rango
                     .set_index('FECHA VENTA')
                     .resample('D')['Total_num']
                     .sum()
                     .reindex(full_idx, fill_value=0.0)
                     .rename_axis('Fecha')
                     .reset_index())

            # Garantiza nombres correctos y tipos válidos para Plotly
            serie.columns = ['Fecha', 'Total_num']
            serie['Fecha'] = pd.to_datetime(serie['Fecha'])

            if serie.empty or 'Fecha' not in serie.columns or 'Total_num' not in serie.columns:
                st.info("No hay datos para graficar en el rango seleccionado.")
            else:
                fig_line = px.line(
                    serie, x="Fecha", y="Total_num",
                    markers=True, title="Evolución temporal de ventas"
                )
                st.plotly_chart(fig_line, use_container_width=True, key="t1_line")

        st.markdown("### Consulta puntual por período")
        periodo = st.radio("Periodo", ["Año","Mes","Semana","Día"], horizontal=True, key="t1_periodo")

        valor = 0.0
        if periodo == "Año":
            opciones = sorted(dfv["Año"].unique().tolist())
            if opciones:
                sel = st.selectbox("Selecciona año", opciones, key="t1_sel_year")
                valor = float(dfv.loc[dfv["Año"] == sel, "Total_num"].sum())
            else:
                st.info("No hay años disponibles.")

        elif periodo == "Mes":
            opciones = sorted(dfv["Mes"].unique().tolist())
            if opciones:
                sel = st.selectbox("Selecciona mes (YYYY-MM)", opciones, key="t1_sel_month")
                valor = float(dfv.loc[dfv["Mes"] == sel, "Total_num"].sum())
            else:
                st.info("No hay meses disponibles.")

        elif periodo == "Semana":
            opciones = sorted(dfv["Semana"].unique().tolist())
            if opciones:
                sel = st.selectbox("Selecciona semana (YYYY-Wxx)", opciones, key="t1_sel_week")
                valor = float(dfv.loc[dfv["Semana"] == sel, "Total_num"].sum())
            else:
                st.info("No hay semanas disponibles.")

        else:  # Día
            opciones = sorted(dfv["Día"].unique().tolist())
            if opciones:
                sel = st.selectbox("Selecciona día", opciones, key="t1_sel_day")
                valor = float(dfv.loc[dfv["Día"] == sel, "Total_num"].sum())
            else:
                st.info("No hay días disponibles.")

        st.metric("Valor seleccionado", f"${valor:,.0f}")

    # -------- Productos
    with tab_r2:
        st.subheader("Análisis por Productos")

        # Top-N dinámico
        if 'Producto_Nombre' in dfv.columns:
            colp1, colp2 = st.columns(2)
            with colp1:
                topn = st.selectbox("Top-N Productos", [5,10,15,20,30], index=1, key="t1_topprod")
                prod = (dfv.groupby("Producto_Nombre", as_index=False)["Total_num"].sum()
                          .sort_values("Total_num", ascending=False).head(topn))
                if prod.empty:
                    st.info("No hay ventas para mostrar.")
                else:
                    fig = px.bar(prod, x="Total_num", y="Producto_Nombre",
                                 orientation="h", title=f"Top {topn} Productos")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            with colp2:
                st.dataframe(prod, use_container_width=True)

        # Filtros por hoja Productos: Tipo de piel y Condición
        if df_productos is not None and not df_productos.empty:
            prod_raw = df_productos.copy()
            campos_ok = {'Producto_Nombre','Tipo_Piel','Condicion'}
            if campos_ok.issubset(prod_raw.columns):
                st.markdown("#### Filtros por características del producto")
                tipos_piel = sorted(prod_raw['Tipo_Piel'].dropna().astype(str).unique())
                conds      = sorted(prod_raw['Condicion'].dropna().astype(str).unique())
                f_piel = st.selectbox("Tipo de Piel", ["(Todos)"] + tipos_piel, key="t1_piel")
                f_cond = st.selectbox("Condición", ["(Todos)"] + conds, key="t1_cond")

                prod_map = prod_raw[['Producto_Nombre','Tipo_Piel','Condicion']].drop_duplicates()
                dfv_merge = dfv.merge(prod_map, on="Producto_Nombre", how="left")

                if f_piel != "(Todos)":
                    dfv_merge = dfv_merge[dfv_merge['Tipo_Piel'] == f_piel]
                if f_cond != "(Todos)":
                    dfv_merge = dfv_merge[dfv_merge['Condicion'] == f_cond]

                if dfv_merge.empty:
                    st.info("No hay ventas para el filtro seleccionado.")
                else:
                    resumen = (dfv_merge.groupby("Producto_Nombre", as_index=False)["Total_num"]
                                       .sum()
                                       .sort_values("Total_num", ascending=False))
                    fig2 = px.bar(resumen, x="Total_num", y="Producto_Nombre",
                                  orientation="h",
                                  title="Ventas filtradas por Tipo de Piel / Condición")
                    fig2.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(resumen, use_container_width=True)
            else:
                st.info("La hoja 'Productos' no tiene columnas suficientes para filtros (se esperan: LISTA PRODUCTOS, TIPO DE PIEL, CONDICION).")

    # -------- Clientes
    with tab_r3:
        st.subheader("Top Clientes por Ventas")
        topn_cli = st.selectbox("Top-N Clientes", [5,10,15,20,30], index=1, key="t1_topcli")
        if 'Cliente/Empresa' in dfv.columns:
            cli = (dfv.groupby("Cliente/Empresa", as_index=False)["Total_num"].sum()
                   .sort_values("Total_num", ascending=False).head(topn_cli))
            colc1, colc2 = st.columns(2)
            with colc1:
                if cli.empty:
                    st.info("No hay ventas para mostrar.")
                else:
                    fig = px.bar(cli, x="Total_num", y="Cliente/Empresa",
                                 orientation="h", title=f"Top {topn_cli} Clientes")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            with colc2:
                st.dataframe(cli, use_container_width=True)

    # -------- Mapa de calor
    with tab_r4:
        st.subheader("Mapa de calor (Día x Mes)")
        heat = (dfv.groupby(['DiaSemana','Mes'], as_index=False)['Total_num']
                  .sum())
        if heat.empty:
            st.info("No hay datos suficientes para el mapa de calor.")
        else:
            orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat['DiaSemana'] = pd.Categorical(heat['DiaSemana'], categories=orden_dias, ordered=True)
            heat_pivot = heat.pivot(index='DiaSemana', columns='Mes', values='Total_num').fillna(0)
            st.plotly_chart(px.imshow(heat_pivot, aspect="auto", title="Heatmap de Ventas (Día de semana x Mes)"),
                            use_container_width=True)



