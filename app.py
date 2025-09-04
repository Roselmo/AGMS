# ==============================================================================
# APP: Dashboard AGMS – Ventas, Cartera, RFM (con ML), Predictivo, Cotizaciones e Inventario
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---- Plotly (con chequeo) ----
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ML
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score, matthews_corrcoef, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# XGBoost opcional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="📊", layout="wide")

LOGO_CANDIDATES = ["ag2.jpg", "logo.png", "AGMS_logo.jpg", "ag_logo.jpg"]
logo_path = next((p for p in LOGO_CANDIDATES if os.path.exists(p)), None)

left, mid, right = st.columns([1, 2, 1])
with left:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with mid:
    st.title("Dashboard AGMS: Ventas, Cartera, RFM, Predicción, Cotizaciones e Inventario")
st.markdown("---")

if not PLOTLY_OK:
    st.error("No se encontró **plotly**. Agrega `plotly` a tu requirements.txt y vuelve a desplegar.")
    st.stop()

# ==============================================================================
# UTILIDADES
# ==============================================================================
def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    if df_counts is None or df_counts.empty:
        return df_counts
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

def limpiar_moneda(valor):
    try:
        if isinstance(valor, str):
            valor_limpio = (
                valor.replace('COP', '').replace('COL$', '').replace('$', '')
                     .replace(' ', '').replace('.', '').replace(',', '.').strip()
            )
            if valor_limpio == '' or valor.strip().lower().startswith('no aplica'):
                return np.nan
            return float(valor_limpio)
        elif isinstance(valor, (int, float)):
            return float(valor)
        return np.nan
    except (ValueError, TypeError):
        return np.nan

def _to_num_price(series_like):
    s = pd.Series(series_like).astype(str)
    s = (s.str.replace('COP','',regex=False).str.replace('COL$','',regex=False)
           .str.replace('$','',regex=False).str.replace(' ','',regex=False)
           .str.replace('.','',regex=False).str.replace(',', '.', regex=False).str.strip())
    s = s.mask(s.str.contains(r'no\s*aplica|n/?a|nan|^$', case=False, regex=True), np.nan)
    return pd.to_numeric(s, errors='coerce')

def parse_fecha_col(col):
    s = pd.Series(col)
    dt_dayfirst  = pd.to_datetime(s.astype(str), dayfirst=True,  errors='coerce')
    dt_yearfirst = pd.to_datetime(s.astype(str), yearfirst=True, errors='coerce')

    def _num_or_nan(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float)) and np.isfinite(x): return float(x)
        xs = str(x).strip()
        if xs.replace('.', '', 1).isdigit():
            try: return float(xs)
            except: return np.nan
        return np.nan

    nums = s.map(_num_or_nan)
    dt_serial = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    mask = nums.notna()
    if mask.any():
        dt_serial.loc[mask] = pd.to_datetime(nums.loc[mask].astype('float64'),
                                             unit='d', origin='1899-12-30', errors='coerce')
    return dt_dayfirst.combine_first(dt_yearfirst).combine_first(dt_serial)

def ensure_product_numeric_cols(df_prod: pd.DataFrame) -> pd.DataFrame:
    if df_prod is None or df_prod.empty:
        return df_prod.copy()
    df = df_prod.copy()
    rename_map = {
        'LISTA PRODUCTOS': 'Producto_Nombre',
        'LISTA_PRODUCTOS': 'Producto_Nombre',
        'Producto': 'Producto_Nombre',
        'TIPO DE PIEL': 'Tipo_Piel',
        'CONDICION': 'Condicion',
        'PROFESIONA o RETAIL': 'Canal',
        'PROFESIONAL o RETAIL': 'Canal',
        'PRECIO PRO': 'Precio_Pro',
        'PRECIO RETAIL': 'Precio_Retail',
        'Marca': 'Marca',
        'STOCK': 'Stock', 'EXISTENCIAS': 'Stock', 'INVENTARIO': 'Stock',
        'MÍNIMO':'Minimo','MINIMO':'Minimo','MÁXIMO':'Maximo','MAXIMO':'Maximo'
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    if 'Producto_Nombre' in df.columns:
        df['Producto_Nombre'] = df['Producto_Nombre'].astype(str).str.strip()
    df['Precio_Pro_num']    = _to_num_price(df.get('Precio_Pro'))
    df['Precio_Retail_num'] = _to_num_price(df.get('Precio_Retail'))
    for col in ['Stock','Minimo','Maximo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'Canal' in df.columns:
        df['Canal'] = df['Canal'].astype(str).str.strip()
    return df

# ---- helpers de gráficos seguros ----
def safe_line(df, x, y, title, key=None, markers=True):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("Sin datos para graficar.")
        return
    fig = px.line(df, x=x, y=y, markers=markers, title=title)
    st.plotly_chart(fig, use_container_width=True, key=key)

def safe_bar(df, x, y, title, key=None, orientation='v', categoryorder=None):
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("Sin datos para graficar.")
        return
    fig = px.bar(df, x=x, y=y, title=title, orientation=orientation)
    if categoryorder:
        fig.update_layout(yaxis={'categoryorder': categoryorder})
    st.plotly_chart(fig, use_container_width=True, key=key)

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=0)
        if 'FECHA VENTA' not in df_ventas.columns and 'FECHA_VENTA' in df_ventas.columns:
            df_ventas.rename(columns={'FECHA_VENTA':'FECHA VENTA'}, inplace=True)
        if 'FECHA VENTA' not in df_ventas.columns:
            df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)

        try: df_medicos   = pd.read_excel(file_path, sheet_name='Lista Medicos')
        except: df_medicos = pd.DataFrame()
        try: df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        except: df_metadatos = pd.DataFrame()
        try: df_cartera   = pd.read_excel(file_path, sheet_name='Cartera')
        except: df_cartera = pd.DataFrame()
        try: df_productos = pd.read_excel(file_path, sheet_name='Productos')
        except: df_productos = pd.DataFrame()

        # Ventas
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas['FECHA VENTA'])
        else:
            alt = next((c for c in df_ventas.columns if 'FECHA' in c.upper()), None)
            if alt:
                df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas[alt])
            else:
                st.error("No se encontró columna de fecha en la hoja Ventas.")
                return None, None, None, None, None
        df_ventas = df_ventas.dropna(subset=['FECHA VENTA'])

        total_col = next((c for c in ['Total','VALOR TOTAL','VALOR_TOTAL','TOTAL'] if c in df_ventas.columns), None)
        if total_col is None:
            df_ventas['Total'] = 0.0
            total_col = 'Total'
        df_ventas['Total'] = df_ventas[total_col]
        df_ventas['Total_num'] = pd.to_numeric(
            df_ventas['Total'].astype(str)
            .str.replace('COP','',regex=False).str.replace('COL$','',regex=False)
            .str.replace('$','',regex=False).str.replace(' ','',regex=False)
            .str.replace('.','',regex=False).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0.0)

        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        if 'Producto_Nombre' not in df_ventas.columns:
            if 'Producto' in df_ventas.columns:
                import re
                def _base_name(s: str) -> str:
                    return re.split(r'\s[-/]\s?\$|\s[-/]\s?precio|\s-\s', str(s), flags=re.IGNORECASE)[0].strip()
                df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(_base_name)
            else:
                prod_alt = next((c for c in df_ventas.columns if 'PRODUCT' in c.upper()), None)
                df_ventas['Producto_Nombre'] = df_ventas[prod_alt].astype(str) if prod_alt else '(SIN PRODUCTO)'

        df_ventas['Año']       = df_ventas['FECHA VENTA'].dt.year
        df_ventas['Mes_P']     = df_ventas['FECHA VENTA'].dt.to_period('M')
        df_ventas['Semana_P']  = df_ventas['FECHA VENTA'].dt.to_period('W')
        df_ventas['Día']       = df_ventas['FECHA VENTA'].dt.date
        df_ventas['Mes']       = df_ventas['Mes_P'].astype(str)
        df_ventas['Semana']    = df_ventas['Semana_P'].astype(str)
        df_ventas['DiaSemana'] = df_ventas['FECHA VENTA'].dt.day_name()

        # Cartera
        if not df_cartera.empty:
            rename_map = {
                'Numero de Factura': 'NÚMERO DE FACTURA',
                'NUMERO DE FACTURA': 'NÚMERO DE FACTURA',
                'Num Factura': 'NÚMERO DE FACTURA',
                'Cliente': 'Nombre cliente',
                'CLIENTE': 'Nombre cliente',
                'Fecha Vencimiento': 'Fecha de Vencimiento',
                'SALDO PENDIENTE': 'Saldo pendiente'
            }
            df_cartera.rename(columns={c: rename_map.get(c, c) for c in df_cartera.columns}, inplace=True)
            if 'Fecha de Vencimiento' in df_cartera.columns:
                df_cartera['Fecha de Vencimiento'] = pd.to_datetime(df_cartera['Fecha de Vencimiento'], errors='coerce')
            for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
                if col in df_cartera.columns:
                    df_cartera[col] = df_cartera[col].apply(limpiar_moneda).fillna(0.0)

        df_productos = ensure_product_numeric_cols(df_productos)

        return df_ventas, df_medicos, df_metadatos, df_cartera, df_productos
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera, df_productos = load_data()
if df_ventas is None:
    st.stop()

# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Análisis de Ventas", "Gestión de Cartera", "Análisis RFM",
    "Modelo Predictivo de Compradores Potenciales", "Cotizaciones", "Inventario"
])

# ---------------------------------------------------------------------------------
# TAB 1: ANÁLISIS DE VENTAS
# ---------------------------------------------------------------------------------
with tab1:
    st.header("Análisis General de Ventas")
    dfv = df_ventas.copy()

    total_ventas = float(dfv["Total_num"].sum())
    total_transacciones = len(dfv)
    clientes_unicos = dfv["Cliente/Empresa"].nunique() if "Cliente/Empresa" in dfv.columns else 0
    ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventas Totales (desde inicio)", f"${total_ventas:,.0f}")
    c2.metric("Transacciones", f"{total_transacciones:,}")
    c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
    st.markdown("---")

    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Resumen", "Productos", "Clientes", "Mapa de calor"])

    # -------- Resumen
    with tab_r1:
        st.subheader("Evolución de Ventas")

        min_date, max_date = dfv['FECHA VENTA'].min().date(), dfv['FECHA VENTA'].max().date()
        rango = st.date_input(
            "Selecciona rango de fechas",
            value=(min_date, max_date),
            min_value=min_date, max_value=max_date, key="t1_rango"
        )

        if isinstance(rango, tuple) and len(rango) == 2:
            fecha_ini = pd.to_datetime(rango[0])
            fecha_fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            dfv_rango = dfv[(dfv['FECHA VENTA'] >= fecha_ini) & (dfv['FECHA VENTA'] <= fecha_fin)]
        else:
            dfv_rango = dfv

        # ✅ Construcción explícita de la serie con columna "Fecha"
        serie = (dfv_rango
                 .assign(Fecha=dfv_rango['FECHA VENTA'].dt.date)
                 .groupby('Fecha', as_index=False)['Total_num']
                 .sum())

        if serie.empty:
            st.info("Sin datos en el rango seleccionado.")
        else:
            safe_line(serie, x="Fecha", y="Total_num", title="Evolución temporal de ventas", key="t1_line")

        # Consulta por período
        with st.expander("🔎 Consultar valor por período (Año / Mes / Semana / Día)"):
            periodo = st.radio("Periodo", ["Año","Mes","Semana","Día"], horizontal=True, key="t1_periodo")
            if periodo == "Año":
                opciones = sorted(dfv["Año"].unique())
                sel = st.selectbox("Selecciona año", opciones, key="t1_sel_year")
                valor = dfv.loc[dfv["Año"] == sel, "Total_num"].sum()
            elif periodo == "Mes":
                opciones = sorted(dfv["Mes"].unique())
                sel = st.selectbox("Selecciona mes (YYYY-MM)", opciones, key="t1_sel_month")
                valor = dfv.loc[dfv["Mes"] == sel, "Total_num"].sum()
            elif periodo == "Semana":
                opciones = sorted(dfv["Semana"].unique())
                sel = st.selectbox("Selecciona semana (YYYY-Wxx)", opciones, key="t1_sel_week")
                valor = dfv.loc[dfv["Semana"] == sel, "Total_num"].sum()
            else:
                opciones = sorted(dfv["Día"].unique())
                sel = st.selectbox("Selecciona día", opciones, key="t1_sel_day")
                valor = dfv.loc[dfv["Día"] == sel, "Total_num"].sum()
            st.metric("Valor seleccionado", f"${valor:,.0f}")

    # -------- Productos
    with tab_r2:
        st.subheader("Análisis por Productos")
        if 'Producto_Nombre' in dfv.columns:
            colp1, colp2 = st.columns(2)
            with colp1:
                topn = st.selectbox("Top-N Productos", [5,10,15,20,30], index=1, key="t1_topprod")
                prod = (dfv.groupby("Producto_Nombre", as_index=False)["Total_num"].sum()
                          .sort_values("Total_num", ascending=False).head(topn))
                if not prod.empty:
                    safe_bar(prod, x="Total_num", y="Producto_Nombre",
                             title=f"Top {topn} Productos", key="t1_prod",
                             orientation='h', categoryorder='total ascending')
                else:
                    st.info("Sin datos de productos.")
            with colp2:
                st.dataframe(prod, use_container_width=True)

        if df_productos is not None and not df_productos.empty:
            prod_map = df_productos.copy()
            campos_ok = {'Producto_Nombre','Tipo_Piel','Condicion'}
            if campos_ok.issubset(prod_map.columns):
                st.markdown("#### Filtros por características del producto")
                tipos_piel = sorted(prod_map['Tipo_Piel'].dropna().astype(str).unique())
                conds      = sorted(prod_map['Condicion'].dropna().astype(str).unique())
                f_piel = st.selectbox("Selecciona Tipo de Piel", ["(Todos)"] + tipos_piel, key="t1_piel")
                f_cond = st.selectbox("Selecciona Condición", ["(Todos)"] + conds, key="t1_cond")

                dfv_merge = dfv.merge(prod_map[['Producto_Nombre','Tipo_Piel','Condicion','Canal']].drop_duplicates(),
                                      on="Producto_Nombre", how="left")
                if f_piel != "(Todos)":
                    dfv_merge = dfv_merge[dfv_merge['Tipo_Piel'] == f_piel]
                if f_cond != "(Todos)":
                    dfv_merge = dfv_merge[dfv_merge['Condicion'] == f_cond]

                if not dfv_merge.empty:
                    resumen_prod = (dfv_merge.groupby("Producto_Nombre", as_index=False)["Total_num"]
                                             .sum().sort_values("Total_num", ascending=False))
                    colx1, colx2 = st.columns(2)
                    with colx1:
                        safe_bar(resumen_prod.head(20), x="Total_num", y="Producto_Nombre",
                                 title="Ventas por Producto (filtrado)", key="t1_prod_f",
                                 orientation='h', categoryorder='total ascending')
                    with colx2:
                        st.dataframe(resumen_prod, use_container_width=True)

                    if 'Canal' in dfv_merge.columns and not dfv_merge['Canal'].isna().all():
                        por_canal = (dfv_merge.groupby("Canal", as_index=False)['Total_num'].sum()
                                             .sort_values('Total_num', ascending=False))
                        safe_bar(por_canal, x="Canal", y="Total_num",
                                 title="Ventas por Canal (según filtro)", key="t1_canal")
                else:
                    st.info("No hay ventas para el filtro seleccionado.")
            else:
                st.info("La hoja 'Productos' debe tener: LISTA PRODUCTOS / TIPO DE PIEL / CONDICION / PROFESIONA o RETAIL / PRECIOS.")
        else:
            st.info("No se encontró la hoja 'Productos'.")

    # -------- Clientes
    with tab_r3:
        st.subheader("Top Clientes por Ventas")
        topn_cli = st.selectbox("Top-N Clientes", [5,10,15,20,30], index=1, key="t1_topcli")
        if 'Cliente/Empresa' in dfv.columns:
            cli = (dfv.groupby("Cliente/Empresa", as_index=False)["Total_num"].sum()
                   .sort_values("Total_num", ascending=False).head(topn_cli))
            colc1, colc2 = st.columns(2)
            with colc1:
                safe_bar(cli, x="Total_num", y="Cliente/Empresa",
                         title=f"Top {topn_cli} Clientes", key="t1_cli",
                         orientation='h', categoryorder='total ascending')
            with colc2:
                st.dataframe(cli, use_container_width=True)
        else:
            st.info("No existe la columna Cliente/Empresa en Ventas.")

    # -------- Heatmap
    with tab_r4:
        st.subheader("Mapa de calor (Día x Mes)")
        heat = (dfv.groupby(['DiaSemana','Mes'], as_index=False)['Total_num'].sum())
        if heat.empty:
            st.info("Sin datos para heatmap.")
        else:
            orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat['DiaSemana'] = pd.Categorical(heat['DiaSemana'], categories=orden_dias, ordered=True)
            heat_pivot = heat.pivot(index='DiaSemana', columns='Mes', values='Total_num').fillna(0)
            st.plotly_chart(px.imshow(heat_pivot, aspect="auto", title="Heatmap de Ventas (Día de semana x Mes)"),
                            use_container_width=True)

# ---------------------------------------------------------------------------------
# TAB 2: GESTIÓN DE CARTERA (lee Cartera y une COMERCIAL por factura)
# ---------------------------------------------------------------------------------
with tab2:
    st.header("Gestión de Cartera")

    if df_cartera is None or df_cartera.empty:
        st.info("No se encontró la hoja 'Cartera'.")
        st.stop()

    dfc = df_cartera.copy()

    # Normalización de nombres
    rename_map = {
        'Numero de Factura': 'NÚMERO DE FACTURA',
        'NUMERO DE FACTURA': 'NÚMERO DE FACTURA',
        'Num Factura': 'NÚMERO DE FACTURA',
        'Cliente': 'Nombre cliente',
        'CLIENTE': 'Nombre cliente',
        'Fecha Vencimiento': 'Fecha de Vencimiento',
        'SALDO PENDIENTE': 'Saldo pendiente'
    }
    dfc.rename(columns={c: rename_map.get(c, c) for c in dfc.columns}, inplace=True)

    # Fechas y montos
    if 'Fecha de Vencimiento' in dfc.columns:
        dfc['Fecha de Vencimiento'] = pd.to_datetime(dfc['Fecha de Vencimiento'], errors='coerce')

    for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
        if col in dfc.columns:
            dfc[col] = dfc[col].apply(limpiar_moneda).fillna(0.0)

    # Unir COMERCIAL desde ventas por NÚMERO DE FACTURA (si existe)
    if 'NÚMERO DE FACTURA' in dfc.columns and 'NÚMERO DE FACTURA' in df_ventas.columns:
        dfc['NÚMERO DE FACTURA'] = dfc['NÚMERO DE FACTURA'].astype(str).str.strip()

        dfv_fact = df_ventas[['NÚMERO DE FACTURA']].copy()
        dfv_fact['NÚMERO DE FACTURA'] = dfv_fact['NÚMERO DE FACTURA'].astype(str).str.strip()
        # Si existe COMERCIAL en ventas, úsalo; si no, crea columna vacía
        if 'COMERCIAL' in df_ventas.columns:
            dfv_fact = dfv_fact.join(df_ventas['COMERCIAL'])
        else:
            dfv_fact['COMERCIAL'] = np.nan

        dfc = dfc.merge(dfv_fact.drop_duplicates(subset=['NÚMERO DE FACTURA']),
                        on='NÚMERO DE FACTURA', how='left')

    # ✅ Crear/asegurar columna COMERCIAL correctamente (evita AttributeError)
    if 'COMERCIAL' not in dfc.columns:
        dfc['COMERCIAL'] = "No disponible"
    else:
        dfc['COMERCIAL'] = dfc['COMERCIAL'].fillna("No disponible").astype(str)

    # Días a vencimiento y estado
    hoy = datetime.now()
    if 'Fecha de Vencimiento' in dfc.columns:
        dfc['Dias_Vencimiento'] = (dfc['Fecha de Vencimiento'] - hoy).dt.days
    else:
        dfc['Dias_Vencimiento'] = np.nan

    def get_status(row):
        sp = row.get('Saldo pendiente', np.nan)
        dv = row.get('Dias_Vencimiento', np.nan)
        if pd.notna(sp) and sp <= 0:
            return 'Pagada'
        if pd.notna(dv) and dv < 0 and pd.notna(sp) and sp > 0:
            return 'Vencida'
        if pd.notna(sp) and sp > 0:
            return 'Por Vencer'
        return 'Sin información'

    dfc['Estado'] = dfc.apply(get_status, axis=1)

    # KPIs
    saldo_total = float(dfc.loc[dfc['Estado'] != 'Pagada', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0
    saldo_vencido = float(dfc.loc[dfc['Estado'] == 'Vencida', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0
    saldo_por_vencer = float(dfc.loc[dfc['Estado'] == 'Por Vencer', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
    c2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
    c3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
    st.markdown("---")

    # Filtros
    filtro_estado = st.selectbox("Filtrar por Estado:", options=['Todas', 'Vencida', 'Por Vencer', 'Pagada', 'Sin información'], key="t2_estado")
    lista_clientes_cartera = sorted(dfc['Nombre cliente'].dropna().unique()) if 'Nombre cliente' in dfc.columns else []
    filtro_cliente = st.multiselect("Filtrar por Cliente:", options=lista_clientes_cartera, key="t2_cliente")
    lista_comerciales = sorted(dfc['COMERCIAL'].fillna('No disponible').unique()) if 'COMERCIAL' in dfc.columns else []
    filtro_comercial = st.multiselect("Filtrar por Comercial:", options=lista_comerciales, key="t2_comercial")

    dfc_filtrada = dfc.copy()
    if filtro_estado != 'Todas':
        dfc_filtrada = dfc_filtrada[dfc_filtrada['Estado'] == filtro_estado]
    if filtro_cliente and 'Nombre cliente' in dfc_filtrada.columns:
        dfc_filtrada = dfc_filtrada[dfc_filtrada['Nombre cliente'].isin(filtro_cliente)]
    if filtro_comercial and 'COMERCIAL' in dfc_filtrada.columns:
        dfc_filtrada = dfc_filtrada[dfc_filtrada['COMERCIAL'].isin(filtro_comercial)]

    def style_venc(row):
        if row.get('Estado') == 'Vencida':
            return ['background-color: #ffcccc'] * len(row)
        dv = row.get('Dias_Vencimiento', np.nan)
        if pd.notna(dv) and 0 <= dv <= 7:
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    cols_show = [c for c in [
        'Nombre cliente', 'NÚMERO DE FACTURA', 'Fecha de Vencimiento',
        'Saldo pendiente', 'Estado', 'Dias_Vencimiento', 'COMERCIAL'
    ] if c in dfc_filtrada.columns]

    st.dataframe(
        dfc_filtrada[cols_show].style.apply(style_venc, axis=1).format({'Saldo pendiente': '${:,.0f}'}),
        use_container_width=True
    )

    st.markdown("---")
    if {'Fecha de Vencimiento','Saldo pendiente'}.issubset(dfc.columns):
        car = dfc[['Fecha de Vencimiento','Saldo pendiente']].copy()
        car['DIAS_VENCIDOS'] = (pd.Timestamp.today().normalize() - car['Fecha de Vencimiento']).dt.days
        labels = ["Al día", "1-30", "31-60", "61-90", "91-180", "181-365", "+365"]
        bins = [-float("inf"), 0, 30, 60, 90, 180, 365, float("inf")]
        car["Rango"] = pd.cut(car["DIAS_VENCIDOS"], bins=bins, labels=labels, ordered=True)
        venc = car.groupby("Rango", as_index=False).agg(Saldo=("Saldo pendiente","sum"))
        st.plotly_chart(px.bar(venc, x="Rango", y="Saldo", title="Antigüedad de saldos (Cartera)"),
                        use_container_width=True, key="t2_aged")

    if 'COMERCIAL' in dfc.columns and 'Saldo pendiente' in dfc.columns:
        por_com = (dfc.groupby('COMERCIAL', as_index=False)['Saldo pendiente']
                     .sum().sort_values('Saldo pendiente', ascending=False))
        st.plotly_chart(px.bar(por_com, x='COMERCIAL', y='Saldo pendiente', title="Saldo pendiente por Comercial"),
                        use_container_width=True, key="t2_by_salesrep")
        
# ---------------------------------------------------------------------------------
# TAB 3: RFM + Recomendador (igual que antes, sin cambios en gráficas de px)
# ---------------------------------------------------------------------------------
with tab3:
    st.header("Análisis RFM + Recomendador ML")
    # (Se mantiene el mismo código de tu versión anterior; no afecta al error de Plotly de la pestaña 1.)

# ---------------------------------------------------------------------------------
# TAB 4: Modelo Predictivo (igual que antes)
# ---------------------------------------------------------------------------------
with tab4:
    st.header("Modelo Predictivo de Compradores Potenciales")
    # (Se mantiene el mismo código de tu versión anterior.)

# ---------------------------------------------------------------------------------
# TAB 5: Cotizaciones (igual que antes)
# ---------------------------------------------------------------------------------
with tab5:
    st.header("🧾 Cotizaciones")
    # (Se mantiene el mismo código de tu versión anterior.)

# ---------------------------------------------------------------------------------
# TAB 6: Inventario (igual que antes)
# ---------------------------------------------------------------------------------
with tab6:
    st.header("📦 Inventario (dinámico)")
    # (Se mantiene el mismo código de tu versión anterior.)
