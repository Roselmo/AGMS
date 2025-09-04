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
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="", layout="wide")

# ---- Portada con logo + título ----
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
    """Normaliza cada fila para convertir conteos a proporciones."""
    if df_counts is None or df_counts.empty:
        return df_counts
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

def limpiar_moneda(valor):
    try:
        if isinstance(valor, str):
            valor_limpio = (
                valor.replace('$', '')
                     .replace('COP', '')
                     .replace('COL$', '')
                     .replace(' ', '')
                     .replace('.', '')
                     .replace(',', '.')
                     .strip()
            )
            # Trata 'No aplica' o vacíos como NaN -> 0.0 después
            if valor_limpio == '' or valor.strip().lower().startswith('no aplica'):
                return np.nan
            return float(valor_limpio)
        elif isinstance(valor, (int, float)):
            return float(valor)
        return np.nan
    except (ValueError, TypeError):
        return np.nan

def _to_num_price(series_like):
    """Convierte precios tipo '$1.234.567,89' o 'No aplica' a float (NaN si no aplica)."""
    s = pd.Series(series_like).astype(str)
    s = (
        s.str.replace('COP', '', regex=False)
         .str.replace('COL$', '', regex=False)
         .str.replace('$', '', regex=False)
         .str.replace(' ', '', regex=False)
         .str.replace('.', '', regex=False)   # miles
         .str.replace(',', '.', regex=False)  # decimales
         .str.strip()
    )
    s = s.mask(s.str.contains(r'no\s*aplica|n/?a|nan|^$', case=False, regex=True), np.nan)
    return pd.to_numeric(s, errors='coerce')

def parse_fecha_col(col):
    """
    Convierte una columna heterogénea a datetime soportando:
    - Textos (dd/mm/yyyy, yyyy-mm-dd, etc.)
    - Seriales de Excel (días desde 1899-12-30)
    """
    s = pd.Series(col)

    # Primero intenta todo como texto
    dt_dayfirst  = pd.to_datetime(s.astype(str), dayfirst=True,  errors='coerce')
    dt_yearfirst = pd.to_datetime(s.astype(str), yearfirst=True, errors='coerce')

    # Serial de Excel SOLO para numéricos claros (enteros/decimales positivos)
    def _num_or_nan(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)) and np.isfinite(x):
            return float(x)
        xs = str(x).strip()
        # Admite números con punto decimal
        if xs.replace('.', '', 1).isdigit():
            try:
                return float(xs)
            except Exception:
                return np.nan
        return np.nan

    nums = s.map(_num_or_nan)
    dt_serial = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    mask = nums.notna()
    if mask.any():
        dt_serial.loc[mask] = pd.to_datetime(
            nums.loc[mask].astype('float64'),
            unit='d', origin='1899-12-30', errors='coerce'
        )

    return dt_dayfirst.combine_first(dt_yearfirst).combine_first(dt_serial)

def ensure_product_numeric_cols(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza columnas auxiliares en hoja Productos:
    - Producto_Nombre, Tipo_Piel, Condicion, Canal, Precio_Pro_num, Precio_Retail_num
    - Stock (si existe con varios nombres), Min/Max (si existen)
    """
    if df_prod is None or df_prod.empty:
        return df_prod.copy()

    df = df_prod.copy()

    # Renombrado estándar segun tus columnas declaradas
    rename_map = {
        'LISTA PRODUCTOS': 'Producto_Nombre',
        'LISTA_PRODUCTOS': 'Producto_Nombre',
        'Producto': 'Producto_Nombre',
        'TIPO DE PIEL': 'Tipo_Piel',
        'CONDICION': 'Condicion',
        'PROFESIONA o RETAIL': 'Canal',     # ojo al typo
        'PROFESIONAL o RETAIL': 'Canal',
        'PRECIO PRO': 'Precio_Pro',
        'PRECIO RETAIL': 'Precio_Retail',
        'Marca': 'Marca',
        'MARCA': 'Marca',
        'STOCK': 'Stock',
        'EXISTENCIAS': 'Stock',
        'INVENTARIO': 'Stock',
        'MÍNIMO': 'Minimo',
        'MINIMO': 'Minimo',
        'MÁXIMO': 'Maximo',
        'MAXIMO': 'Maximo'
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    if 'Producto_Nombre' in df.columns:
        df['Producto_Nombre'] = df['Producto_Nombre'].astype(str).str.strip()

    # Precios numéricos
    df['Precio_Pro_num']    = _to_num_price(df.get('Precio_Pro'))
    df['Precio_Retail_num'] = _to_num_price(df.get('Precio_Retail'))

    # Stock y min/max si existen
    for col in ['Stock', 'Minimo', 'Maximo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Canal en str
    if 'Canal' in df.columns:
        df['Canal'] = df['Canal'].astype(str).str.strip()

    return df

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        # ---- Ventas: intenta header=0; si no trae FECHA VENTA prueba header=1
        df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=0)
        if 'FECHA VENTA' not in df_ventas.columns and 'FECHA_VENTA' in df_ventas.columns:
            df_ventas.rename(columns={'FECHA_VENTA': 'FECHA VENTA'}, inplace=True)
        if 'FECHA VENTA' not in df_ventas.columns:
            df_ventas = pd.read_excel(file_path, sheet_name='Ventas', header=1)

        # ---- Otras hojas
        try:
            df_medicos = pd.read_excel(file_path, sheet_name='Lista Medicos')
        except Exception:
            df_medicos = pd.DataFrame()

        try:
            df_metadatos = pd.read_excel(file_path, sheet_name='Metadatos')
        except Exception:
            df_metadatos = pd.DataFrame()

        try:
            df_cartera = pd.read_excel(file_path, sheet_name='Cartera')
        except Exception:
            df_cartera = pd.DataFrame()

        try:
            df_productos = pd.read_excel(file_path, sheet_name='Productos')
        except Exception:
            df_productos = pd.DataFrame()

        # ===================== Limpieza Ventas =====================
        # FECHA VENTA robusta
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas['FECHA VENTA'])
        else:
            # intenta alguna variante común
            alt = next((c for c in df_ventas.columns if 'FECHA' in c.upper()), None)
            if alt:
                df_ventas['FECHA VENTA'] = parse_fecha_col(df_ventas[alt])
            else:
                st.error("No se encontró columna de fecha en la hoja Ventas.")
                return None, None, None, None, None

        df_ventas = df_ventas.dropna(subset=['FECHA VENTA'])

        # Total numérico robusto (intenta varias columnas)
        total_col_candidates = ['Total', 'VALOR TOTAL', 'VALOR_TOTAL', 'TOTAL']
        total_col = next((c for c in total_col_candidates if c in df_ventas.columns), None)
        if total_col is None:
            # crea Total a 0 si no existe
            df_ventas['Total'] = 0.0
            total_col = 'Total'
        df_ventas['Total'] = df_ventas[total_col]
        df_ventas['Total_num'] = pd.to_numeric(
            df_ventas['Total'].astype(str)
                              .str.replace('COP', '', regex=False)
                              .str.replace('COL$', '', regex=False)
                              .str.replace('$', '', regex=False)
                              .str.replace(' ', '', regex=False)
                              .str.replace('.', '', regex=False)
                              .str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0.0)

        # Cliente normalizado
        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        # Producto_Nombre deducido si no existe
        if 'Producto_Nombre' not in df_ventas.columns:
            if 'Producto' in df_ventas.columns:
                import re
                def _base_name(s: str) -> str:
                    s = str(s)
                    return re.split(r'\s[-/]\s?\$|\s[-/]\s?precio|\s-\s', s, flags=re.IGNORECASE)[0].strip()
                df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(_base_name)
            else:
                # busca alguna columna con producto
                prod_alt = next((c for c in df_ventas.columns if 'PRODUCT' in c.upper()), None)
                if prod_alt:
                    df_ventas['Producto_Nombre'] = df_ventas[prod_alt].astype(str)
                else:
                    df_ventas['Producto_Nombre'] = '(SIN PRODUCTO)'

        # Derivadas de tiempo
        df_ventas['Año']       = df_ventas['FECHA VENTA'].dt.year
        df_ventas['Mes_P']     = df_ventas['FECHA VENTA'].dt.to_period('M')
        df_ventas['Semana_P']  = df_ventas['FECHA VENTA'].dt.to_period('W')
        df_ventas['Día']       = df_ventas['FECHA VENTA'].dt.date
        df_ventas['Mes']       = df_ventas['Mes_P'].astype(str)    # 'YYYY-MM'
        df_ventas['Semana']    = df_ventas['Semana_P'].astype(str) # 'YYYY-Wxx'
        df_ventas['DiaSemana'] = df_ventas['FECHA VENTA'].dt.day_name()

        # ===================== Médicos =====================
        if not df_medicos.empty:
            if 'NOMBRE' in df_medicos.columns:
                df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
            if 'TELEFONO' in df_medicos.columns:
                df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # ===================== Cartera =====================
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

        # ===================== Productos =====================
        df_productos = ensure_product_numeric_cols(df_productos)

        return df_ventas, df_medicos, df_metadatos, df_cartera, df_productos
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera, df_productos = load_data()
if df_ventas is None:
    st.stop()

# ==============================================================================
# TABS PRINCIPALES (sin "Series" ni "Pareto") + Inventario nuevo
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Análisis de Ventas", "Gestión de Cartera", "Análisis RFM",
    "Modelo Predictivo de Compradores Potenciales", "Cotizaciones", "Inventario"
])

# ---------------------------------------------------------------------------------
# TAB 1: ANÁLISIS DE VENTAS (corregido)
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
    c1.metric("Ventas Totales (desde inicio)", f"${total_ventas:,.0f}")
    c2.metric("Transacciones", f"{total_transacciones:,}")
    c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
    st.markdown("---")

    # ========================= SUBPESTAÑAS =========================
    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(
        ["Resumen", "Productos", "Clientes", "Mapa de calor"]
    )

    # -------- Subpestaña: Resumen (evolución temporal + consulta por período)
    with tab_r1:
        st.subheader("Evolución de Ventas")

        # Rango dinámico (todo el histórico por defecto)
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

        # Serie diaria
        serie = (dfv_rango
                 .groupby(dfv_rango['FECHA VENTA'].dt.date, as_index=False)['Total_num']
                 .sum()
                 .rename(columns={'FECHA VENTA': 'Fecha'}))
        fig_line = px.line(
            serie, x="Fecha", y="Total_num",
            markers=True, title="Evolución temporal de ventas"
        )
        st.plotly_chart(fig_line, use_container_width=True, key="t1_line")

        # 🔎 Consulta por Año/Mes/Semana/Día (suma exacta)
        with st.expander("🔎 Consultar valor por período (Año / Mes / Semana / Día)"):
            periodo = st.radio("Periodo", ["Año","Mes","Semana","Día"], horizontal=True, key="t1_periodo")

            if periodo == "Año":
                opciones = sorted(dfv["Año"].unique())
                sel = st.selectbox("Selecciona año", opciones, key="t1_sel_year")
                valor = dfv.loc[dfv["Año"] == sel, "Total_num"].sum()

            elif periodo == "Mes":
                opciones = sorted(dfv["Mes"].unique())   # p.ej. 2024-08, 2024-09
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

    # -------- Subpestaña: Productos (con filtros Tipo de piel y Condición)
    with tab_r2:
        st.subheader("Análisis por Productos")

        # Top-N dinámico
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

        # Filtros con hoja Productos (Tipo_Piel/Condicion) y análisis de Canal según ventas
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

                # Resumen por producto (filtrado)
                if not dfv_merge.empty:
                    resumen_prod = (dfv_merge.groupby("Producto_Nombre", as_index=False)["Total_num"]
                                             .sum()
                                             .sort_values("Total_num", ascending=False))
                    colx1, colx2 = st.columns(2)
                    with colx1:
                        st.plotly_chart(
                            px.bar(resumen_prod.head(20), x="Total_num", y="Producto_Nombre",
                                   orientation="h", title="Ventas por Producto (filtrado)"),
                            use_container_width=True
                        )
                    with colx2:
                        st.dataframe(resumen_prod, use_container_width=True)

                    # Resumen por Canal (derivado de selección de piel/condición)
                    if 'Canal' in dfv_merge.columns and not dfv_merge['Canal'].isna().all():
                        por_canal = (dfv_merge.groupby("Canal", as_index=False)['Total_num'].sum()
                                             .sort_values('Total_num', ascending=False))
                        st.plotly_chart(
                            px.bar(por_canal, x="Canal", y="Total_num", title="Ventas por Canal (según filtro)"),
                            use_container_width=True
                        )
                else:
                    st.info("No hay ventas para el filtro seleccionado.")
            else:
                st.info("La hoja 'Productos' debe tener: LISTA PRODUCTOS / TIPO DE PIEL / CONDICION / PROFESIONA o RETAIL / PRECIOS.")
        else:
            st.info("No se encontró la hoja 'Productos'.")

    # -------- Subpestaña: Clientes
    with tab_r3:
        st.subheader("Top Clientes por Ventas")
        topn_cli = st.selectbox("Top-N Clientes", [5,10,15,20,30], index=1, key="t1_topcli")
        if 'Cliente/Empresa' in dfv.columns:
            cli = (dfv.groupby("Cliente/Empresa", as_index=False)["Total_num"].sum()
                   .sort_values("Total_num", ascending=False).head(topn_cli))
            colc1, colc2 = st.columns(2)
            with colc1:
                figc = px.bar(cli, x="Total_num", y="Cliente/Empresa",
                              orientation="h", title=f"Top {topn_cli} Clientes")
                figc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(figc, use_container_width=True)
            with colc2:
                st.dataframe(cli, use_container_width=True)

    # -------- Subpestaña: Mapa de calor
    with tab_r4:
        st.subheader("Mapa de calor (Día x Mes)")
        heat = (dfv.groupby(['DiaSemana','Mes'], as_index=False)['Total_num']
                  .sum())
        orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat['DiaSemana'] = pd.Categorical(heat['DiaSemana'], categories=orden_dias, ordered=True)
        heat_pivot = heat.pivot(index='DiaSemana', columns='Mes', values='Total_num').fillna(0)
        st.plotly_chart(
            px.imshow(heat_pivot, aspect="auto", title="Heatmap de Ventas (Día de semana x Mes)"),
            use_container_width=True
        )

# ---------------------------------------------------------------------------------
# TAB 2: GESTIÓN DE CARTERA
# ---------------------------------------------------------------------------------
with tab2:
    st.header("Gestión de Cartera")

    if df_cartera is None or df_cartera.empty:
        st.info("No se encontró la hoja 'Cartera'.")
    else:
        dfc = df_cartera.copy()

        if 'Fecha de Vencimiento' in dfc.columns:
            dfc['Fecha de Vencimiento'] = pd.to_datetime(dfc['Fecha de Vencimiento'], errors='coerce')

        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in dfc.columns:
                dfc[col] = dfc[col].apply(limpiar_moneda).fillna(0.0)

        # Unir COMERCIAL desde ventas por NÚMERO DE FACTURA
        if 'NÚMERO DE FACTURA' in dfc.columns and 'NÚMERO DE FACTURA' in df_ventas.columns:
            dfc['NÚMERO DE FACTURA'] = dfc['NÚMERO DE FACTURA'].astype(str).str.strip()
            dfv_fact = df_ventas[['NÚMERO DE FACTURA']].copy()
            if 'COMERCIAL' in df_ventas.columns:
                dfv_fact['COMERCIAL'] = df_ventas['COMERCIAL']
            else:
                dfv_fact['COMERCIAL'] = np.nan
            dfv_fact['NÚMERO DE FACTURA'] = dfv_fact['NÚMERO DE FACTURA'].astype(str).str.strip()
            dfc = dfc.merge(dfv_fact.drop_duplicates(), on='NÚMERO DE FACTURA', how='left')
        else:
            dfc['COMERCIAL'] = dfc.get('COMERCIAL', "No disponible").fillna("No disponible")

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
            dfc_filtrada = dfc_filtrada[dfc_filtrada['COMERCIAL'].fillna('No disponible').isin(filtro_comercial)]

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
            por_com = dfc.groupby('COMERCIAL', as_index=False)['Saldo pendiente'].sum().sort_values('Saldo pendiente', ascending=False)
            st.plotly_chart(px.bar(por_com, x='COMERCIAL', y='Saldo pendiente', title="Saldo pendiente por Comercial"),
                            use_container_width=True, key="t2_by_salesrep")

# ---------------------------------------------------------------------------------
# TAB 3: ANÁLISIS RFM + Recomendador ML
# ---------------------------------------------------------------------------------
with tab3:
    st.header("Análisis RFM + Recomendador ML")

    colp1, colp2, colp3, colp4 = st.columns(4)
    dias_recencia = colp1.slider("Ventana 'comprador reciente' (días)", 7, 120, 30, key="t3_rec")
    top_k_sugerencias = colp2.slider("Nº sugerencias a mostrar", 5, 30, 10, key="t3_top")
    usar_top_productos = colp3.checkbox("Usar señales de productos (Top 10)", value=True, key="t3_topprod")
    excluir_recencia = colp4.checkbox("Excluir 'Recencia' como feature", value=True, key="t3_exrec")

    dias_op = ["(Todos)","Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    dia_reporte = st.selectbox("Día deseado del reporte", dias_op, index=0, key="t3_dia")

    cols_nec = {'Cliente/Empresa', 'FECHA VENTA', 'Total'}
    if not cols_nec.issubset(df_ventas.columns):
        st.warning(f"Faltan columnas para RFM/ML: {cols_nec}.")
    else:
        ejecutar = st.button("🚀 Ejecutar RFM + Entrenar y Comparar Modelos", key="t3_run")
        if ejecutar:
            with st.spinner("Procesando..."):
                ventas = df_ventas.copy()
                ventas['Cliente/Empresa'] = ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()
                ventas['FECHA VENTA'] = pd.to_datetime(ventas['FECHA VENTA'], errors="coerce")
                ventas = ventas.dropna(subset=['FECHA VENTA'])
                ref_date = ventas['FECHA VENTA'].max()
                tiene_factura = 'NÚMERO DE FACTURA' in ventas.columns

                # RFM
                rfm = ventas.groupby('Cliente/Empresa').agg(
                    Recencia=('FECHA VENTA', lambda s: (ref_date - s.max()).days),
                    Frecuencia=('NÚMERO DE FACTURA', 'nunique') if tiene_factura else ('FECHA VENTA','count'),
                    Monetario=('Total', 'sum')
                ).reset_index()
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

                rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'], ascending=True, labels=[5,4,3,2,1])
                rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
                rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])
                rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")

                st.caption("Distribución de segmentos RFM")
                st.dataframe(rfm['Segmento'].value_counts(dropna=False).rename_axis('Segmento').to_frame('Clientes'),
                             use_container_width=True)

                # Features comportamiento
                ventas['DiaSemana'] = ventas['FECHA VENTA'].dt.dayofweek
                ventas['Hora'] = ventas['FECHA VENTA'].dt.hour
                feats_dia  = ventas.groupby(['Cliente/Empresa','DiaSemana']).size().unstack(fill_value=0)
                feats_dia.columns  = [f"dw_{int(c)}" for c in feats_dia.columns]
                feats_hora = ventas.groupby(['Cliente/Empresa','Hora']).size().unstack(fill_value=0)
                feats_hora.columns = [f"h_{int(c)}" for c in feats_hora.columns]
                feats_dia  = row_normalize(feats_dia)
                feats_hora = row_normalize(feats_hora)

                feats_prod = None
                if usar_top_productos and 'Producto_Nombre' in ventas.columns:
                    top10_prod = (ventas.groupby('Producto_Nombre')['Total'].sum()
                                  .sort_values(ascending=False).head(10).index.tolist())
                    v_prod = ventas[ventas['Producto_Nombre'].isin(top10_prod)].copy()
                    feats_prod = (v_prod.groupby(['Cliente/Empresa','Producto_Nombre']).size().unstack(fill_value=0))
                    feats_prod = row_normalize(feats_prod)

                df_feat = rfm.merge(feats_dia, on='Cliente/Empresa', how='left') \
                             .merge(feats_hora, on='Cliente/Empresa', how='left')
                if feats_prod is not None:
                    df_feat = df_feat.merge(feats_prod, on='Cliente/Empresa', how='left')
                for c in df_feat.select_dtypes(include=[np.number]).columns:
                    df_feat[c] = df_feat[c].fillna(0)

                # Target comprador reciente
                recientes = ventas[ventas['FECHA VENTA'] >= ref_date - pd.Timedelta(days=dias_recencia)]['Cliente/Empresa'].unique()
                df_feat['comprador_reciente'] = df_feat['Cliente/Empresa'].isin(recientes).astype(int)

                # Filtro de segmentos multi
                segmentos_all = sorted(df_feat['Segmento'].dropna().unique().tolist())
                seg_sel = st.multiselect("Filtrar por Segmento RFM (multi)", options=segmentos_all, default=segmentos_all, key="t3_seg")
                if seg_sel:
                    df_feat = df_feat[df_feat['Segmento'].isin(seg_sel)]

                feature_cols = ['Frecuencia','Monetario'] + [c for c in df_feat.columns if c.startswith('dw_') or c.startswith('h_')]
                if feats_prod is not None:
                    feature_cols += [c for c in df_feat.columns if c in feats_prod.columns]
                if not excluir_recencia:
                    feature_cols = ['Recencia'] + feature_cols
                X = df_feat[feature_cols]
                y = df_feat['comprador_reciente']

                if y.nunique() < 2:
                    st.warning("La variable objetivo tiene una sola clase. Ajusta la ventana/segmentos.")
                    st.stop()

                modelos = {
                    "LogisticRegression": LogisticRegression(max_iter=800, C=0.3, penalty="l2", class_weight='balanced'),
                    "RandomForest": RandomForestClassifier(
                        n_estimators=250, max_depth=6, min_samples_leaf=10,
                        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
                    ),
                }
                if HAS_XGB:
                    modelos["XGBoost"] = XGBClassifier(
                        n_estimators=350, learning_rate=0.06, max_depth=4,
                        min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                        reg_lambda=1.2, random_state=RANDOM_STATE, eval_metric='logloss', tree_method="hist"
                    )
                else:
                    modelos["GradientBoosting"] = GradientBoostingClassifier(
                        n_estimators=300, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE
                    )

                n_splits = int(np.clip(y.value_counts().min(), 2, 5))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
                resultados = []
                for nombre, modelo in modelos.items():
                    cv_res = cross_validate(modelo, X, y, cv=cv,
                                            scoring={'accuracy':'accuracy','f1':'f1','roc_auc':'roc_auc'},
                                            n_jobs=-1)
                    resultados.append({
                        "Modelo": nombre,
                        "Accuracy": f"{cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}",
                        "F1":       f"{cv_res['test_f1'].mean():.3f} ± {cv_res['test_f1'].std():.3f}",
                        "AUC":      f"{cv_res['test_roc_auc'].mean():.3f} ± {cv_res['test_roc_auc'].std():.3f}",
                        "_auc_mean": cv_res['test_roc_auc'].mean(),
                        "_f1_mean":  cv_res['test_f1'].mean()
                    })

                df_res = pd.DataFrame(resultados).sort_values(by=["_auc_mean","_f1_mean"], ascending=False)
                mejor_modelo_nombre = df_res.iloc[0]["Modelo"]
                st.subheader("Comparación de Modelos (CV)")
                st.dataframe(df_res.drop(columns=["_auc_mean","_f1_mean"]), use_container_width=True)
                st.success(f"🏆 Mejor modelo: **{mejor_modelo_nombre}**")

                best_model = modelos[mejor_modelo_nombre]
                best_model.fit(X, y)
                if hasattr(best_model, "predict_proba"):
                    probs_full = best_model.predict_proba(X)[:,1]
                elif hasattr(best_model, "decision_function"):
                    s_full = best_model.decision_function(X)
                    probs_full = (s_full - s_full.min()) / (s_full.max() - s_full.min() + 1e-9)
                else:
                    probs_full = best_model.predict(X)

                df_feat['Prob_Compra'] = probs_full

                # Candidatos (no recientes)
                candidatos = df_feat[df_feat['comprador_reciente'] == 0].copy()

                # Mejor día histórico (dw_)
                dia_cols = [c for c in candidatos.columns if c.startswith("dw_")]
                def mejor_dia(row):
                    if not dia_cols: return None
                    sub = row[dia_cols]
                    if (sub.max() == 0) or sub.isna().all(): return None
                    idx = int(sub.idxmax().split("_")[1])
                    mapa_dw = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
                    return mapa_dw.get(idx)
                candidatos['Dia_Contacto'] = candidatos.apply(mejor_dia, axis=1)

                # Producto sugerido
                if 'Producto_Nombre' in ventas.columns and not ventas['Producto_Nombre'].isna().all():
                    top_prod_cliente = (ventas.groupby(['Cliente/Empresa', 'Producto_Nombre'])['Total']
                                        .sum().reset_index())
                    idx = top_prod_cliente.groupby('Cliente/Empresa')['Total'].idxmax()
                    top_prod_cliente = top_prod_cliente.loc[idx][['Cliente/Empresa', 'Producto_Nombre']] \
                                                       .rename(columns={'Producto_Nombre':'Producto_Sugerido'})
                    candidatos = candidatos.merge(top_prod_cliente, on='Cliente/Empresa', how='left')
                else:
                    candidatos['Producto_Sugerido'] = None

                # Filtro día
                if dia_reporte != "(Todos)":
                    candidatos = candidatos[candidatos['Dia_Contacto'] == dia_reporte]

                if candidatos.empty:
                    st.info("No hay candidatos con los filtros seleccionados.")
                else:
                    topN = candidatos.nlargest(top_k_sugerencias, 'Prob_Compra')[
                        ['Cliente/Empresa','Prob_Compra','Producto_Sugerido','Dia_Contacto','Segmento']
                    ].copy()
                    asignaciones = (["Camila", "Andrea"] * ((len(topN)//2)+1))[:len(topN)]
                    topN['Asignado_a'] = asignaciones

                    st.subheader("🎯 Top clientes potenciales a contactar")
                    st.dataframe(
                        topN.rename(columns={'Cliente/Empresa':'Cliente','Prob_Compra':'Probabilidad_Compra'}) \
                            .style.format({'Probabilidad_Compra':'{:.1%}'}),
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Descargar sugerencias (CSV)",
                        data=topN.to_csv(index=False).encode('utf-8'),
                        file_name=f"sugerencias_rfm_ml_{pd.Timestamp.today().date()}.csv",
                        mime="text/csv",
                        key="t3_dl"
                    )

# ---------------------------------------------------------------------------------
# TAB 4: MODELO PREDICTIVO DE COMPRADORES POTENCIALES (Optimización con CV)
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
            with st.spinner("Construyendo dataset, optimizando hiperparámetros y seleccionando el mejor modelo..."):
                data = df_ventas[['Cliente/Empresa','Producto_Nombre','Total','FECHA VENTA']].copy()
                data['Cliente/Empresa'] = data['Cliente/Empresa'].astype(str).str.strip().str.upper()
                data['FECHA VENTA'] = pd.to_datetime(data['FECHA VENTA'], errors="coerce")
                data = data.dropna(subset=['FECHA VENTA'])

                data['Mes'] = data['FECHA VENTA'].dt.month
                data['DiaSemana'] = data['FECHA VENTA'].dt.dayofweek
                data['Hora'] = data['FECHA VENTA'].dt.hour
                data['target'] = (data['Producto_Nombre'] == producto_sel).astype(int)

                feats = data.groupby('Cliente/Empresa').agg(
                    Total_Gastado=('Total','sum'),
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

                top_prod = (data.groupby('Producto_Nombre')['Total'].sum()
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

                # RF
                rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
                rf_space = {
                    "n_estimators": np.linspace(200, 800, 7, dtype=int).tolist(),
                    "max_depth": [None, 6, 10, 14],
                    "min_samples_leaf": [1, 2, 4, 8, 12],
                    "max_features": ["sqrt", 0.5, None]
                }
                rf_search = RandomizedSearchCV(
                    rf, rf_space, n_iter=n_iter_rf, scoring=scorer_balacc, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                ).fit(X, y)

                # XGB / GB
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
                    xgb_search = RandomizedSearchCV(
                        xgb, xgb_space, n_iter=n_iter_xgb, scoring=scorer_balacc, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    ).fit(X, y)
                    gb_label = "XGBoost"
                else:
                    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
                    gb_space = {
                        "n_estimators": np.linspace(150, 500, 8, dtype=int).tolist(),
                        "learning_rate": [0.03, 0.05, 0.07, 0.1],
                        "max_depth": [2, 3, 4],
                        "min_samples_leaf": [1, 5, 10, 20]
                    }
                    xgb_search = RandomizedSearchCV(
                        gb, gb_space, n_iter=n_iter_xgb, scoring=scorer_balacc, refit=True,
                        cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                    ).fit(X, y)
                    gb_label = "GradientBoosting"

                # MLP
                mlp = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=800))
                ])
                mlp_space = {
                    "clf__hidden_layer_sizes": [(64,32), (128,64), (64,64,32)],
                    "clf__alpha": [1e-4, 1e-3, 1e-2],
                    "clf__learning_rate_init": [1e-3, 5e-4],
                    "clf__batch_size": [32, 64]
                }
                mlp_search = RandomizedSearchCV(
                    mlp, mlp_space, n_iter=n_iter_mlp, scoring=scorer_balacc, refit=True,
                    cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
                ).fit(X, y)

                def eval_model(est):
                    scores = cross_validate(
                        est, X, y, cv=cv,
                        scoring={'BalAcc': scorer_balacc, 'MCC': scorer_mcc, 'F1_macro': scorer_f1m},
                        n_jobs=-1
                    )
                    return (
                        float(np.mean(scores['test_BalAcc'])),
                        float(np.mean(scores['test_MCC'])),
                        float(np.mean(scores['test_F1_macro']))
                    )

                models_best = [
                    ("RandomForest", rf_search),
                    (gb_label,       xgb_search),
                    ("MLPClassifier", mlp_search)
                ]

                rows = []
                for name, search in models_best:
                    balacc, mcc, f1m = eval_model(search.best_estimator_)
                    rows.append({
                        "Modelo": name,
                        "Balanced Acc (CV)": f"{balacc:.3f}",
                        "MCC (CV)": f"{mcc:.3f}",
                        "F1-macro (CV)": f"{f1m:.3f}",
                        "Mejores Hiperparámetros": str(search.best_params_),
                        "_key": (balacc, mcc, f1m)
                    })

                df_cmp = pd.DataFrame(rows).sort_values(
                    by=["Balanced Acc (CV)","MCC (CV)","F1-macro (CV)"], ascending=False
                ).drop(columns=["_key"], errors="ignore")
                st.subheader("📈 Resultados de Optimización (mejor configuración por modelo)")
                st.dataframe(df_cmp, use_container_width=True)

                best_row = max(rows, key=lambda r: r["_key"])
                best_name = best_row["Modelo"]
                best_search = dict(models_best)[best_name]
                st.success(
                    f"🏆 Mejor modelo: **{best_name}** · "
                    f"Balanced Acc={best_row['Balanced Acc (CV)']} · "
                    f"MCC={best_row['MCC (CV)']} · F1-macro={best_row['F1-macro (CV)']}"
                )

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

                st.subheader("🎯 Top 10 clientes potenciales (mejor modelo optimizado)")
                st.dataframe(
                    top10.rename(columns={'Cliente/Empresa':'Cliente'}) \
                         .style.format({'Probabilidad_Compra':'{:.1%}'}),
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Descargar candidatos (CSV)",
                    data=top10.to_csv(index=False).encode('utf-8'),
                    file_name=f"candidatos_{producto_sel}_opt_balanced.csv",
                    mime="text/csv",
                    key="t4_dl"
                )

# ---------------------------------------------------------------------------------
# TAB 5: COTIZACIONES
# ---------------------------------------------------------------------------------
with tab5:
    st.header("🧾 Cotizaciones")

    if df_productos is None or df_productos.empty or 'Producto_Nombre' not in df_productos.columns:
        st.warning("No se encontró la hoja 'Productos' con el formato esperado.")
        st.stop()

    catalog = ensure_product_numeric_cols(df_productos.copy())

    # Auxiliares para cálculo (si no hay precio -> 0 en cálculo, pero se informa)
    catalog['Precio_Pro_num']    = catalog['Precio_Pro_num'].fillna(0.0)
    catalog['Precio_Retail_num'] = catalog['Precio_Retail_num'].fillna(0.0)

    # Selector de productos (autocompletable)
    opciones = sorted(catalog['Producto_Nombre'].astype(str).unique().tolist())
    sel = st.multiselect(
        "Agrega productos a la cotización (escribe iniciales para buscar)",
        options=opciones, key="cot_sel"
    )

    if "cot_items" not in st.session_state:
        st.session_state["cot_items"] = {}  # { producto: {"qty": int, "price_type": "Pro"/"Retail"} }

    # Sincronizar selección con sesión
    for p in sel:
        if p not in st.session_state["cot_items"]:
            row_p = catalog.loc[catalog['Producto_Nombre'] == p].iloc[0]
            # default: si existe precio pro, usarlo; si no, retail; si no, None
            if float(row_p['Precio_Pro_num']) > 0:
                default_ptype = "Pro"
            elif float(row_p['Precio_Retail_num']) > 0:
                default_ptype = "Retail"
            else:
                default_ptype = None
            st.session_state["cot_items"][p] = {"qty": 1, "price_type": default_ptype}

    # Eliminar del estado los productos des-seleccionados
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
            valid_opts = []
            if float(row['Precio_Pro_num']) > 0:    valid_opts.append("Pro")
            if float(row['Precio_Retail_num']) > 0: valid_opts.append("Retail")

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
            if st.session_state["cot_items"][p]["price_type"] == "Pro":
                punit = float(row['Precio_Pro_num'])
            elif st.session_state["cot_items"][p]["price_type"] == "Retail":
                punit = float(row['Precio_Retail_num'])

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
            st.dataframe(
                df_cot.style.format({"PrecioUnitario":"${:,.0f}","Subtotal":"${:,.0f}"}),
                use_container_width=True
            )

# ---------------------------------------------------------------------------------
# TAB 6: INVENTARIO (nuevo)
# ---------------------------------------------------------------------------------
with tab6:
    st.header("📦 Inventario (dinámico)")

    if df_productos is None or df_productos.empty:
        st.info("No se encontró la hoja 'Productos'.")
        st.stop()

    inv = ensure_product_numeric_cols(df_productos)

    # Detección de columnas de stock
    has_stock = 'Stock' in inv.columns
    if not has_stock:
        st.warning("No se detectó una columna de Stock/Existencias/Inventario en Productos. "
                   "Puedes continuar con stock=0 o cargar una columna llamada 'Stock' en la hoja.")

    # Parámetros de cálculo
    colA, colB, colC, colD = st.columns(4)
    ventana_ventas = colA.selectbox("Ventana de ventas (días)", [30, 60, 90, 120, 180], index=0, key="inv_win")
    lead_time      = colB.number_input("Lead Time (días)", min_value=0, value=7, step=1, key="inv_lt")
    safety_days    = colC.number_input("Safety Stock (días)", min_value=0, value=7, step=1, key="inv_ss")
    usar_precio    = colD.selectbox("Valor inventario con:", ["Precio_Pro_num", "Precio_Retail_num"], index=0, key="inv_val")

    # Ventas en ventana: tasa diaria por producto
    max_fecha = df_ventas['FECHA VENTA'].max()
    min_cut   = max_fecha - pd.Timedelta(days=ventana_ventas-1)
    ventas_win = df_ventas[df_ventas['FECHA VENTA'] >= min_cut]

    ventas_prod = (ventas_win.groupby('Producto_Nombre', as_index=False)['Total_num'].sum()
                   .rename(columns={'Total_num':'Ventas_ventana'}))
    dias_efectivos = max(1, (max_fecha.normalize() - min_cut.normalize()).days + 1)
    ventas_prod['Venta_diaria'] = ventas_prod['Ventas_ventana'] / dias_efectivos

    # Unir a inventario
    inv2 = inv.merge(ventas_prod[['Producto_Nombre','Venta_diaria']], on='Producto_Nombre', how='left')
    inv2['Venta_diaria'] = inv2['Venta_diaria'].fillna(0.0)
    if 'Stock' not in inv2.columns:
        inv2['Stock'] = 0.0
    inv2['Stock'] = pd.to_numeric(inv2['Stock'], errors='coerce').fillna(0.0)

    # Days of Cover y sugerencia de compra
    inv2['Days_of_Cover'] = np.where(inv2['Venta_diaria'] > 0, inv2['Stock'] / inv2['Venta_diaria'], np.inf)
    inv2['Demanda_LT']    = inv2['Venta_diaria'] * lead_time
    inv2['Safety_Stock']  = inv2['Venta_diaria'] * safety_days
    inv2['ROP']           = inv2['Demanda_LT'] + inv2['Safety_Stock']      # Reorder Point
    inv2['Sugerido_Compra'] = np.maximum(0.0, inv2['ROP'] - inv2['Stock'])

    # Valor inventario y KPIs
    inv2['Valor_unit'] = inv2[usar_precio].fillna(0.0)
    inv2['Valor_inventario'] = inv2['Stock'] * inv2['Valor_unit']

    total_skus = inv2['Producto_Nombre'].nunique()
    valor_total = float(inv2['Valor_inventario'].sum())
    skus_bajo_cobertura = int((inv2['Days_of_Cover'] < (lead_time + safety_days)).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("SKUs", f"{total_skus:,}")
    k2.metric("Valor total inventario", f"${valor_total:,.0f}")
    k3.metric("SKUs con cobertura baja", f"{skus_bajo_cobertura:,}")

    st.markdown("### Prioridades de reposición")
    top_rep = inv2.sort_values(['Sugerido_Compra','Days_of_Cover'], ascending=[False, True]).head(50)
    st.dataframe(
        top_rep[['Producto_Nombre','Stock','Venta_diaria','Days_of_Cover','ROP','Sugerido_Compra','Valor_unit']]
            .rename(columns={
                'Producto_Nombre':'Producto',
                'Venta_diaria':'Venta diaria',
                'Days_of_Cover':'Días de cobertura',
                'Sugerido_Compra':'Sugerido compra',
                'Valor_unit': f'Valor unitario ({usar_precio})'
            })
            .style.format({
                'Stock': '{:,.0f}',
                'Venta diaria': '{:,.2f}',
                'Días de cobertura': '{:,.1f}',
                'ROP': '{:,.1f}',
                'Sugerido compra': '{:,.0f}',
                f'Valor unitario ({usar_precio})': '${:,.0f}'
            }),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Descargar tabla de inventario (CSV)",
        data=inv2.to_csv(index=False).encode('utf-8'),
        file_name=f"inventario_{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
        key="inv_dl"
    )
