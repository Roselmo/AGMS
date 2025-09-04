# ==============================================================================
# APP: Dashboard AGMS – Ventas, Cartera, RFM (con ML), Predictivo, Cotizaciones e Inventario
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

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

l, m, r = st.columns([1, 2, 1])
with l:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with m:
    st.title("Dashboard AGMS: Ventas, Cartera, RFM, Predicción, Cotizaciones e Inventario")
st.markdown("---")

if not PLOTLY_OK:
    st.error("No se encontró **plotly**. Agrega `plotly` a tu requirements.txt y vuelve a desplegar.")
    st.stop()

# ==============================================================================
# UTILIDADES (robustas)
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

    # Ambos separadores
    if has_comma and has_dot:
        # ¿Cuál aparece más a la derecha? Ese suele ser el decimal
        if s.rfind(',') > s.rfind('.'):
            # '1.234.567,89' -> eliminar puntos (miles) y coma -> punto
            s = s.replace('.', '').replace(',', '.')
        else:
            # '1,234,567.89' -> eliminar comas (miles), conservar punto decimal
            s = s.replace(',', '')
    else:
        # Sólo uno de los separadores
        if has_comma:
            # ¿coma decimal? si el último bloque tiene 1-2 dígitos -> decimal
            parts = s.split(',')
            if len(parts[-1]) in (1, 2):
                s = s.replace('.', '')  # quita puntos si había miles
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')  # eran miles
        elif has_dot:
            parts = s.split('.')
            if len(parts[-1]) in (1, 2):
                s = s.replace(',', '')  # por si viniera mezclado
            else:
                s = s.replace('.', '')  # puntos como miles

    try:
        return float(s)
    except Exception:
        return np.nan

def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    if df_counts is None or df_counts.empty:
        return df_counts
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

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

# ===== RFM helpers =====
def _safe_qcut_score(series, ascending, labels=[1,2,3,4,5]):
    s = series.copy()
    rk = s.rank(method='first', ascending=ascending)
    try:
        q = pd.qcut(rk, 5, labels=labels)
        return q.astype(int)
    except Exception:
        q = pd.cut(rk, bins=5, labels=labels, include_lowest=True, duplicates='drop')
        q = q.astype('float')
        fill_val = np.ceil(q.mean()) if not np.isnan(q.mean()) else 3
        return q.fillna(fill_val).astype(int)

def rfm_segment(row):
    r,f,m = row['R_Score'], row['F_Score'], row['M_Score']
    if r>=4 and f>=4 and m>=4: return "Champions"
    if r>=4 and f>=3: return "Loyal"
    if r>=3 and f>=3 and m>=3: return "Potential Loyalist"
    if r<=2 and f>=4: return "At Risk"
    if r<=2 and f<=2 and m<=2: return "Hibernating"
    if r>=3 and f<=2: return "New"
    return "Need Attention"

# ===== Productos helpers =====
def ensure_product_cols(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas de la hoja Productos (según lo que indicaste):
    LISTA PRODUCTOS | TIPO DE PIEL | CONDICION | PROFESIONA o RETAIL | PRECIO PRO | PRECIO RETAIL | Marca
    y crea:
      - _Precio_Pro_num, _Precio_Retail_num (floats)
    """
    if df_prod is None or df_prod.empty:
        return df_prod

    # Renombrado robusto
    rename_map = {
        'LISTA PRODUCTOS': 'Producto_Nombre',
        'TIPO DE PIEL': 'Tipo_Piel',
        'CONDICION': 'Condicion',
        'PROFESIONA o RETAIL': 'Canal',      # algunos archivos vienen sin la 'L'
        'PROFESIONAL o RETAIL': 'Canal',
        'PRECIO PRO': 'Precio_Pro',
        'PRECIO RETAIL': 'Precio_Retail',
        'Marca': 'Marca'
    }
    dfp = df_prod.rename(columns={c: rename_map.get(c, c) for c in df_prod.columns}).copy()

    if 'Producto_Nombre' in dfp.columns:
        dfp['Producto_Nombre'] = dfp['Producto_Nombre'].astype(str).str.strip()

    # Precios numéricos auxiliares
    dfp['_Precio_Pro_num'] = parse_col_money_safe(dfp.get('Precio_Pro'))
    dfp['_Precio_Retail_num'] = parse_col_money_safe(dfp.get('Precio_Retail'))

    return dfp

def parse_col_money_safe(series):
    if series is None:
        return pd.Series(np.nan)
    return series.apply(parse_money)

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        # Lee las hojas principales
        df_ventas     = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos    = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos  = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera    = pd.read_excel(file_path, sheet_name='Cartera')
        df_productos  = pd.read_excel(file_path, sheet_name='Productos')
        # Inventario (si existe)
        try:
            df_inventario = pd.read_excel(file_path, sheet_name='Inventario')
        except Exception:
            df_inventario = pd.DataFrame()

        # ---- Ventas
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas['FECHA VENTA'])
            df_ventas = df_ventas.dropna(subset=['FECHA VENTA'])
            df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
            df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
            df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour

        # Total numérico robusto
        if 'Total' in df_ventas.columns:
            df_ventas['Total_num'] = df_ventas['Total'].apply(parse_money).fillna(0.0)
        else:
            df_ventas['Total_num'] = 0.0

        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        if 'Producto_Nombre' not in df_ventas.columns and 'Producto' in df_ventas.columns:
            df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).str.split(' - ').str[0].str.strip()

        # ---- Médicos
        if 'NOMBRE' in df_medicos.columns:
            df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
        if 'TELEFONO' in df_medicos.columns:
            df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # ---- Cartera
        ren_cartera = {
            'Numero de Factura': 'NÚMERO DE FACTURA',
            'NUMERO DE FACTURA': 'NÚMERO DE FACTURA',
            'Num Factura': 'NÚMERO DE FACTURA',
            'Cliente': 'Nombre cliente',
            'CLIENTE': 'Nombre cliente',
            'Fecha Vencimiento': 'Fecha de Vencimiento',
            'SALDO PENDIENTE': 'Saldo pendiente'
        }
        df_cartera.rename(columns={c: ren_cartera.get(c, c) for c in df_cartera.columns}, inplace=True)

        if 'Fecha de Vencimiento' in df_cartera.columns:
            df_cartera['Fecha de Vencimiento'] = parse_fecha_col(df_cartera['Fecha de Vencimiento'])

        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in df_cartera.columns:
                df_cartera[col] = df_cartera[col].apply(parse_money).fillna(0.0)

        # ---- Productos (normalizado a tus campos)
        df_productos = ensure_product_cols(df_productos)

        # ---- Inventario (opcional)
        if not df_inventario.empty:
            # Normalización mínima
            if 'Producto_Nombre' not in df_inventario.columns and 'Producto' in df_inventario.columns:
                df_inventario = df_inventario.rename(columns={'Producto':'Producto_Nombre'})
            if 'Producto_Nombre' in df_inventario.columns:
                df_inventario['Producto_Nombre'] = df_inventario['Producto_Nombre'].astype(str).str.strip()
            if 'Stock' not in df_inventario.columns:
                df_inventario['Stock'] = 0
            if 'Costo' in df_inventario.columns:
                df_inventario['Costo'] = df_inventario['Costo'].apply(parse_money).fillna(0.0)
        else:
            df_inventario = pd.DataFrame(columns=['Producto_Nombre','Stock','Costo'])

        return df_ventas, df_medicos, df_metadatos, df_cartera, df_productos, df_inventario
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera, df_productos, df_inventario = load_data()
if df_ventas is None or df_cartera is None or df_productos is None:
    st.stop()

# ==============================================================================
# TABS PRINCIPALES
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
    if 'FECHA VENTA' not in dfv.columns:
        st.warning("No se encuentra la columna 'FECHA VENTA' en la hoja Ventas.")
        st.stop()

    # Derivadas temporales
    dfv['Año']      = dfv['FECHA VENTA'].dt.year
    dfv['Mes_P']    = dfv['FECHA VENTA'].dt.to_period('M')
    dfv['Semana_P'] = dfv['FECHA VENTA'].dt.to_period('W')
    dfv['Día']      = dfv['FECHA VENTA'].dt.date
    dfv['Mes']      = dfv['Mes_P'].astype(str)     # 'YYYY-MM'
    dfv['Semana']   = dfv['Semana_P'].astype(str)  # 'YYYY-Wxx'
    dfv['DiaSemana'] = dfv['FECHA VENTA'].dt.day_name()

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

    # ========================= Subpestañas =========================
    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Resumen", "Productos", "Clientes", "Mapa de calor"])

    # -------- Resumen (evolución + consulta puntual Año/Mes/Semana/Día)
    with tab_r1:
        st.subheader("Evolución de Ventas (rango)")

        min_date, max_date = dfv['FECHA VENTA'].min().date(), dfv['FECHA VENTA'].max().date()
        rango = st.date_input(
            "Selecciona rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="t1_rango"
        )

        if isinstance(rango, tuple) and len(rango) == 2:
            fecha_ini = pd.to_datetime(rango[0])
            fecha_fin = pd.to_datetime(rango[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            dfv_rango = dfv[(dfv['FECHA VENTA'] >= fecha_ini) & (dfv['FECHA VENTA'] <= fecha_fin)]
        else:
            dfv_rango = dfv

        if dfv_rango.empty:
            st.info("No hay ventas en el rango seleccionado.")
        else:
            serie = (dfv_rango
                     .groupby(dfv_rango['FECHA VENTA'].dt.date, as_index=False)['Total_num']
                     .sum()
                     .rename(columns={'FECHA VENTA':'Fecha'}))
            serie.rename(columns={serie.columns[0]: 'Fecha'}, inplace=True)
            fig_line = px.line(serie, x="Fecha", y="Total_num", markers=True, title="Evolución temporal de ventas")
            st.plotly_chart(fig_line, use_container_width=True, key="t1_line")

        st.markdown("### Consulta puntual por período")
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

        else:  # Día
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
                fig = px.bar(prod, x="Total_num", y="Producto_Nombre",
                             orientation="h", title=f"Top {topn} Productos")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            with colp2:
                st.dataframe(prod, use_container_width=True)

        # Filtros por hoja Productos: Tipo de piel y Condición
        if df_productos is not None and not df_productos.empty:
            prod_raw = df_productos.copy()  # ya normalizada con ensure_product_cols
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

                if not dfv_merge.empty:
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
                    st.info("No hay ventas para el filtro seleccionado.")
            else:
                st.info("La hoja 'Productos' no tiene columnas suficientes para estos filtros.")

    # -------- Clientes
    with tab_r3:
        st.subheader("Top Clientes por Ventas")
        topn_cli = st.selectbox("Top-N Clientes", [5,10,15,20,30], index=1, key="t1_topcli")
        if 'Cliente/Empresa' in dfv.columns:
            cli = (dfv.groupby("Cliente/Empresa", as_index=False)["Total_num"].sum()
                   .sort_values("Total_num", ascending=False).head(topn_cli))
            colc1, colc2 = st.columns(2)
            with colc1:
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
        orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat['DiaSemana'] = pd.Categorical(heat['DiaSemana'], categories=orden_dias, ordered=True)
        heat_pivot = heat.pivot(index='DiaSemana', columns='Mes', values='Total_num').fillna(0)
        st.plotly_chart(px.imshow(heat_pivot, aspect="auto", title="Heatmap de Ventas (Día de semana x Mes)"),
                        use_container_width=True)

# ---------------------------------------------------------------------------------
# TAB 2: GESTIÓN DE CARTERA (corregida)
# ---------------------------------------------------------------------------------
with tab2:
    st.header("Gestión de Cartera")

    if df_cartera is None or df_cartera.empty:
        st.info("No se encontró la hoja 'Cartera'.")
        st.stop()

    dfc = df_cartera.copy()

    # Unir COMERCIAL desde ventas por NÚMERO DE FACTURA
    if 'NÚMERO DE FACTURA' in dfc.columns and 'NÚMERO DE FACTURA' in df_ventas.columns:
        dfc['NÚMERO DE FACTURA'] = dfc['NÚMERO DE FACTURA'].astype(str).str.strip()
        dfv_fact = df_ventas[['NÚMERO DE FACTURA']].copy()
        dfv_fact['NÚMERO DE FACTURA'] = dfv_fact['NÚMERO DE FACTURA'].astype(str).str.strip()
        if 'COMERCIAL' in df_ventas.columns:
            dfv_fact = dfv_fact.join(df_ventas['COMERCIAL'])
        else:
            dfv_fact['COMERCIAL'] = np.nan
        dfc = dfc.merge(dfv_fact.drop_duplicates(subset=['NÚMERO DE FACTURA']),
                        on='NÚMERO DE FACTURA', how='left')

    if 'COMERCIAL' not in dfc.columns:
        dfc['COMERCIAL'] = "No disponible"
    else:
        dfc['COMERCIAL'] = dfc['COMERCIAL'].fillna("No disponible").astype(str)

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

    saldo_total = float(dfc.loc[dfc['Estado'] != 'Pagada', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0
    saldo_vencido = float(dfc.loc[dfc['Estado'] == 'Vencida', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0
    saldo_por_vencer = float(dfc.loc[dfc['Estado'] == 'Por Vencer', 'Saldo pendiente'].sum()) if 'Saldo pendiente' in dfc.columns else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo Total Pendiente", f"${saldo_total:,.0f}")
    c2.metric("Total Vencido", f"${saldo_vencido:,.0f}", delta="Riesgo Alto", delta_color="inverse")
    c3.metric("Total por Vencer", f"${saldo_por_vencer:,.0f}")
    st.markdown("---")

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
# TAB 3: ANÁLISIS RFM (usa Total_num)
# ---------------------------------------------------------------------------------
with tab3:
    st.header("Análisis RFM + Recomendador ML")

    cols_nec = {'Cliente/Empresa', 'FECHA VENTA', 'Total_num'}
    if not cols_nec.issubset(df_ventas.columns):
        st.warning(f"Faltan columnas para RFM/ML: {cols_nec}.")
    else:
        ventas = df_ventas.copy()
        ventas['FECHA VENTA'] = pd.to_datetime(ventas['FECHA VENTA'], errors="coerce")
        ventas = ventas.dropna(subset=['FECHA VENTA'])
        ref_date = ventas['FECHA VENTA'].max()
        tiene_factura = 'NÚMERO DE FACTURA' in ventas.columns

        rfm = ventas.groupby('Cliente/Empresa').agg(
            Recencia=('FECHA VENTA', lambda s: (ref_date - s.max()).days),
            Frecuencia=('NÚMERO DE FACTURA', 'nunique') if tiene_factura else ('FECHA VENTA','count'),
            Monetario=('Total_num', 'sum')
        ).reset_index()
        rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'], ascending=True, labels=[5,4,3,2,1])
        rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
        rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])
        rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")

        st.subheader("Distribución de segmentos RFM")
        st.dataframe(rfm['Segmento'].value_counts(dropna=False).rename_axis('Segmento').to_frame('Clientes'),
                     use_container_width=True)

# ---------------------------------------------------------------------------------
# TAB 4: MODELO PREDICTIVO (usa Total_num)
# ---------------------------------------------------------------------------------
with tab4:
    st.header("Modelo Predictivo de Compradores Potenciales")

    if 'Producto_Nombre' not in df_ventas.columns:
        st.warning("No se encuentra la columna 'Producto_Nombre' en ventas.")
    else:
        producto_sel = st.selectbox(
            "Producto objetivo:",
            options=sorted(df_ventas['Producto_Nombre'].dropna().unique()),
            key="t4_prod"
        )

        colh1, colh2, colh3 = st.columns(3)
        n_iter_rf  = colh1.slider("Iteraciones búsqueda RF",   5, 40, 15, key="t4_rf_iter")
        n_iter_mlp = colh2.slider("Iteraciones búsqueda MLP",  5, 40, 15, key="t4_mlp_iter")
        n_iter_xgb = colh3.slider(f"Iteraciones búsqueda {'XGB' if HAS_XGB else 'GB'}", 5, 50, 20, key="t4_xgb_iter")

        if st.button("Entrenar y Optimizar Modelos", key="t4_train"):
            with st.spinner("Entrenando..."):
                data = df_ventas[['Cliente/Empresa','Producto_Nombre','Total_num','FECHA VENTA']].copy()
                data['Cliente/Empresa'] = data['Cliente/Empresa'].astype(str).str.strip().str.upper()
                data['FECHA VENTA'] = pd.to_datetime(data['FECHA VENTA'], errors="coerce")
                data = data.dropna(subset=['FECHA VENTA'])

                data['Mes'] = data['FECHA VENTA'].dt.month
                data['DiaSemana'] = data['FECHA VENTA'].dt.dayofweek
                data['Hora'] = data['FECHA VENTA'].dt.hour
                data['target'] = (data['Producto_Nombre'] == producto_sel).astype(int)

                feats = data.groupby('Cliente/Empresa').agg(
                    Total_Gastado=('Total_num','sum'),
                    Num_Transacciones=('Producto_Nombre','count'),
                    Ultimo_Mes=('Mes','max'),
                    Promedio_DiaSemana=('DiaSemana','mean'),
                    Promedio_Hora=('Hora','mean'),
                    Compró=('target','max')
                ).reset_index()

                f_dw = data.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                f_dw.columns = [f"dw_{int(c)}" for c in f_dw.columns]
                f_dw = row_normalize(f_dw)

                f_h  = data.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                f_h.columns = [f"h_{int(c)}" for c in f_h.columns]
                f_h = row_normalize(f_h)

                top_prod = (data.groupby('Producto_Nombre')['Total_num'].sum()
                            .sort_values(ascending=False).head(10).index.tolist())
                v_prod = data[data['Producto_Nombre'].isin(top_prod)].copy()
                f_prod = v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0)
                f_prod = row_normalize(f_prod)

                DS = feats.merge(f_dw, on='Cliente/Empresa', how='left') \
                          .merge(f_h,  on='Cliente/Empresa', how='left') \
                          .merge(f_prod, on='Cliente/Empresa', how='left')
                for c in DS.select_dtypes(include=[np.number]).columns:
                    DS[c] = DS[c].fillna(0)

                y = DS['Compró'].astype(int)
                X = DS.drop(columns=['Cliente/Empresa','Compró'])

                cls_counts = y.value_counts()
                st.caption(f"Distribución de clases (Compró / No Compró): {cls_counts.to_dict()}")
                if y.nunique() < 2:
                    st.error("El objetivo tiene una sola clase. Cambia el producto o revisa datos.")
                    st.stop()

                n_splits = int(np.clip(cls_counts.min(), 2, 5))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

                scorer_balacc = make_scorer(balanced_accuracy_score)
                scorer_mcc    = make_scorer(matthews_corrcoef)
                scorer_f1m    = make_scorer(f1_score, average="macro")

                rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
                rf_space = {
                    "n_estimators": np.linspace(200, 800, 7, dtype=int).tolist(),
                    "max_depth": [None, 6, 10, 14],
                    "min_samples_leaf": [1, 2, 4, 8, 12],
                    "max_features": ["sqrt", 0.5, None]
                }
                rf_search = RandomizedSearchCV(rf, rf_space, n_iter=n_iter_rf, scoring=scorer_balacc,
                                               refit=True, cv=cv, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)

                if HAS_XGB:
                    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", tree_method="hist")
                    xgb_space = {
                        "n_estimators": np.linspace(200, 700, 6, dtype=int).tolist(),
                        "learning_rate": [0.03, 0.05, 0.07, 0.1],
                        "max_depth": [3, 4, 5, 6],
                        "subsample": [0.8, 0.9, 1.0],
                        "colsample_bytree": [0.6, 0.8, 1.0],
                        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
                        "min_child_weight": [1, 3, 5]
                    }
                    xgb_search = RandomizedSearchCV(xgb, xgb_space, n_iter=n_iter_xgb, scoring=scorer_balacc,
                                                    refit=True, cv=cv, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
                    gb_label = "XGBoost"
                    second = xgb_search
                else:
                    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
                    gb_space = {
                        "n_estimators": np.linspace(150, 500, 8, dtype=int).tolist(),
                        "learning_rate": [0.03, 0.05, 0.07, 0.1],
                        "max_depth": [2, 3, 4],
                        "min_samples_leaf": [1, 5, 10, 20]
                    }
                    second = RandomizedSearchCV(gb, gb_space, n_iter=n_iter_xgb, scoring=scorer_balacc,
                                                refit=True, cv=cv, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)
                    gb_label = "GradientBoosting"

                mlp = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=800))])
                mlp_space = {
                    "clf__hidden_layer_sizes": [(64,32), (128,64), (64,64,32)],
                    "clf__alpha": [1e-4, 1e-3, 1e-2],
                    "clf__learning_rate_init": [1e-3, 5e-4],
                    "clf__batch_size": [32, 64]
                }
                mlp_search = RandomizedSearchCV(mlp, mlp_space, n_iter=n_iter_mlp, scoring=scorer_balacc,
                                                refit=True, cv=cv, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)

                def eval_model(est):
                    scores = cross_validate(est, X, y, cv=cv,
                                            scoring={'BalAcc': scorer_balacc, 'MCC': scorer_mcc, 'F1_macro': scorer_f1m},
                                            n_jobs=-1)
                    return (float(np.mean(scores['test_BalAcc'])),
                            float(np.mean(scores['test_MCC'])),
                            float(np.mean(scores['test_F1_macro'])))

                models_best = [("RandomForest", rf_search), (gb_label, second), ("MLPClassifier", mlp_search)]
                rows = []
                for name, search in models_best:
                    balacc, mcc, f1m = eval_model(search.best_estimator_)
                    rows.append({"Modelo": name, "Balanced Acc (CV)": f"{balacc:.3f}",
                                 "MCC (CV)": f"{mcc:.3f}", "F1-macro (CV)": f"{f1m:.3f}",
                                 "_key": (balacc, mcc, f1m)})
                df_cmp = pd.DataFrame(rows).sort_values(by=["Balanced Acc (CV)","MCC (CV)","F1-macro (CV)"], ascending=False)
                st.subheader("📈 Resultados de Optimización")
                st.dataframe(df_cmp.drop(columns=['_key']), use_container_width=True)

                best_row = max(rows, key=lambda r: r["_key"])
                best_name = best_row["Modelo"]
                best_search = dict(models_best)[best_name]
                st.success(f"🏆 Mejor modelo: **{best_name}**")

                best_estimator = best_search.best_estimator_
                best_estimator.fit(X, y)
                if hasattr(best_estimator, "predict_proba"):
                    probas = best_estimator.predict_proba(X)[:, 1]
                elif hasattr(best_estimator, "decision_function"):
                    s = best_estimator.decision_function(X)
                    probas = (s - s.min()) / (s.max() - s.min() + 1e-9)
                else:
                    probas = best_estimator.predict(X).astype(float)

                DS = X.copy()
                DS['Cliente/Empresa'] = DS.index.astype(str)
                DS['Probabilidad_Compra'] = probas
                top10 = DS[['Cliente/Empresa','Probabilidad_Compra']].nlargest(10, 'Probabilidad_Compra')

                st.subheader("🎯 Top 10 clientes potenciales")
                st.dataframe(top10.rename(columns={'Cliente/Empresa':'Cliente'})
                                  .style.format({'Probabilidad_Compra':'{:.1%}'}), use_container_width=True)

                st.download_button("⬇️ Descargar candidatos (CSV)",
                                   data=top10.to_csv(index=False).encode('utf-8'),
                                   file_name=f"candidatos_{producto_sel}_opt_balanced.csv",
                                   mime="text/csv", key="t4_dl")

# ---------------------------------------------------------------------------------
# TAB 5: COTIZACIONES (usa PRECIO PRO / PRECIO RETAIL)
# ---------------------------------------------------------------------------------
with tab5:
    st.header("🧾 Cotizaciones")

    if df_productos is None or df_productos.empty or 'Producto_Nombre' not in df_productos.columns:
        st.warning("No se encontró la hoja 'Productos' con el formato esperado.")
        st.stop()

    catalog = df_productos.copy()  # ya normalizado por ensure_product_cols

    # Bandera de "No aplica": True si no hay número en el precio
    catalog['NA_Pro']    = catalog['_Precio_Pro_num'].isna()
    catalog['NA_Retail'] = catalog['_Precio_Retail_num'].isna()

    # Para cálculo, NaN->0 (la UI informa cuando no hay precio)
    catalog['_Precio_Pro_num']    = catalog['_Precio_Pro_num'].fillna(0.0)
    catalog['_Precio_Retail_num'] = catalog['_Precio_Retail_num'].fillna(0.0)

    opciones = sorted(catalog['Producto_Nombre'].astype(str).unique().tolist())
    sel = st.multiselect("Agrega productos a la cotización (escribe iniciales para buscar)",
                         options=opciones, key="cot_sel")

    if "cot_items" not in st.session_state:
        st.session_state["cot_items"] = {}  # { producto: {"qty": int, "price_type": "PRO"/"RETAIL"} }

    for p in sel:
        if p not in st.session_state["cot_items"]:
            row = catalog.loc[catalog['Producto_Nombre'] == p].iloc[0]
            if (not bool(row['NA_Pro'])) and bool(row['NA_Retail']):
                default_ptype = "PRO"
            elif bool(row['NA_Pro']) and (not bool(row['NA_Retail'])):
                default_ptype = "RETAIL"
            elif (not bool(row['NA_Pro'])) and (not bool(row['NA_Retail'])):
                default_ptype = "PRO"
            else:
                default_ptype = None
            st.session_state["cot_items"][p] = {"qty": 1, "price_type": default_ptype}

    for p in list(st.session_state["cot_items"].keys()):
        if p not in sel:
            del st.session_state["cot_items"][p]

    if not sel:
        st.info("Selecciona uno o más productos para armar la cotización.")
    else:
        st.markdown("#### Detalle de la cotización")
        total = 0.0
        rows = []

        hdr = st.columns([4, 2, 3, 3, 3])
        hdr[0].markdown("**Producto**")
        hdr[1].markdown("**Cantidad**")
        hdr[2].markdown("**Tipo de precio**")
        hdr[3].markdown("**Precio unitario**")
        hdr[4].markdown("**Subtotal**")

        for p in sel:
            row = catalog.loc[catalog['Producto_Nombre'] == p].iloc[0]
            na_pro = bool(row['NA_Pro'])
            na_ret = bool(row['NA_Retail'])

            if na_pro and na_ret:
                valid_opts = []
            elif na_pro and not na_ret:
                valid_opts = ["RETAIL"]
            elif not na_pro and na_ret:
                valid_opts = ["PRO"]
            else:
                valid_opts = ["PRO", "RETAIL"]

            c0, c1, c2, c3, c4 = st.columns([4, 2, 3, 3, 3])
            c0.write(p)

            qty_key = f"qty_{p}"
            qty_val = st.session_state["cot_items"][p]["qty"]
            qty = c1.number_input(" ", min_value=1, step=1, value=qty_val, key=qty_key)
            st.session_state["cot_items"][p]["qty"] = qty

            current_ptype = st.session_state["cot_items"][p]["price_type"]
            if current_ptype not in valid_opts and valid_opts:
                current_ptype = valid_opts[0]
                st.session_state["cot_items"][p]["price_type"] = current_ptype

            if valid_opts:
                ptype_key = f"ptype_{p}"
                chosen_ptype = c2.selectbox(" ", options=valid_opts,
                                            index=valid_opts.index(current_ptype) if current_ptype in valid_opts else 0,
                                            key=ptype_key)
                st.session_state["cot_items"][p]["price_type"] = chosen_ptype
            else:
                c2.warning("Sin precio disponible")
                st.session_state["cot_items"][p]["price_type"] = None

            punit = 0.0
            if st.session_state["cot_items"][p]["price_type"] == "PRO":
                punit = float(row.get('_Precio_Pro_num', 0.0))
            elif st.session_state["cot_items"][p]["price_type"] == "RETAIL":
                punit = float(row.get('_Precio_Retail_num', 0.0))

            c3.metric(label=" ", value=f"${punit:,.0f}")

            subtotal = punit * qty
            total += subtotal
            c4.metric(label=" ", value=f"${subtotal:,.0f}")

            rows.append({
                "Producto": p,
                "Cantidad": qty,
                "TipoPrecio": st.session_state["cot_items"][p]["price_type"] or "N/A",
                "PrecioUnitario": punit,
                "Subtotal": subtotal
            })

        st.markdown("---")
        if st.button("Calcular total", key="cot_total_btn"):
            st.success(f"**Total cotización:** ${total:,.0f}")

        if rows:
            df_cot = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Descargar cotización (CSV)",
                data=df_cot.to_csv(index=False).encode("utf-8"),
                file_name=f"cotizacion_{pd.Timestamp.today().date()}.csv",
                mime="text/csv",
                key="cot_csv"
            )
            st.dataframe(df_cot.style.format({"PrecioUnitario":"${:,.0f}","Subtotal":"${:,.0f}"}),
                         use_container_width=True)

# ---------------------------------------------------------------------------------
# TAB 6: INVENTARIO (dinámico)
# ---------------------------------------------------------------------------------
with tab6:
    st.header("📦 Inventario")

    # Base de inventario: si hay hoja Inventario, úsala; si no, crea desde Productos
    if df_inventario is not None and not df_inventario.empty:
        inv = df_inventario.copy()
    else:
        inv = df_productos[['Producto_Nombre']].drop_duplicates().copy()
        inv['Stock'] = 0
        inv['Costo'] = 0.0

    # Merge info de producto
    prod_info = df_productos[['Producto_Nombre','Tipo_Piel','Condicion','Marca']].drop_duplicates()
    inv = inv.merge(prod_info, on='Producto_Nombre', how='left')

    # Filtros
    c1, c2, c3 = st.columns(3)
    marcas = sorted(inv['Marca'].dropna().astype(str).unique().tolist())
    tipos  = sorted(inv['Tipo_Piel'].dropna().astype(str).unique().tolist())
    conds  = sorted(inv['Condicion'].dropna().astype(str).unique().tolist())
    f_marca = c1.selectbox("Marca", ["(Todas)"] + marcas, key="inv_marca")
    f_piel  = c2.selectbox("Tipo de Piel", ["(Todas)"] + tipos, key="inv_piel")
    f_cond  = c3.selectbox("Condición", ["(Todas)"] + conds, key="inv_cond")

    inv_f = inv.copy()
    if f_marca != "(Todas)":
        inv_f = inv_f[inv_f['Marca'] == f_marca]
    if f_piel != "(Todas)":
        inv_f = inv_f[inv_f['Tipo_Piel'] == f_piel]
    if f_cond != "(Todas)":
        inv_f = inv_f[inv_f['Condicion'] == f_cond]

    # Métricas rápidas
    total_items = int(inv_f['Stock'].fillna(0).sum())
    total_skus  = int(inv_f['Producto_Nombre'].nunique())
    val_stock   = float((inv_f['Stock'].fillna(0) * inv_f.get('Costo', 0).fillna(0)).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Unidades en stock (filtrado)", f"{total_items:,}")
    k2.metric("SKUs (filtrado)", f"{total_skus:,}")
    k3.metric("Valor inventario (filtrado)", f"${val_stock:,.0f}")

    # Tabla
    cols = ['Producto_Nombre','Marca','Tipo_Piel','Condicion','Stock']
    if 'Costo' in inv_f.columns:
        cols.append('Costo')
    st.dataframe(inv_f[cols].sort_values('Producto_Nombre'), use_container_width=True)

    # Rotación simple (ventas últimos 90 días) para sugerencia de reabastecimiento
    if {'Producto_Nombre','FECHA VENTA','Total_num'}.issubset(df_ventas.columns):
        cutoff = df_ventas['FECHA VENTA'].max() - pd.Timedelta(days=90)
        recent = df_ventas[df_ventas['FECHA VENTA'] >= cutoff]
        rot = (recent.groupby('Producto_Nombre')
                     .agg(Ventas_90d=('Total_num','sum'),
                          Transacciones=('Producto_Nombre','count'))
                     .reset_index())
        inv_r = inv_f.merge(rot, on='Producto_Nombre', how='left')
        inv_r[['Ventas_90d','Transacciones']] = inv_r[['Ventas_90d','Transacciones']].fillna(0)
        st.markdown("#### Rotación (últimos 90 días)")
        st.dataframe(inv_r[['Producto_Nombre','Stock','Ventas_90d','Transacciones']]
                     .sort_values('Ventas_90d', ascending=False), use_container_width=True)

        # Export
        st.download_button(
            "⬇️ Exportar Inventario Filtrado (CSV)",
            data=inv_f.to_csv(index=False).encode('utf-8'),
            file_name=f"inventario_filtrado_{pd.Timestamp.today().date()}.csv",
            mime="text/csv",
            key="inv_csv"
        )
