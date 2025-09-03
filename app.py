# ==============================================================================
# APP: Dashboard AGMS – Ventas, Cartera, RFM (con ML), Predictivo y Cotizaciones
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
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Dashboard de Ventas AGMS", page_icon="📊", layout="wide")

# ---- Portada con logo + título ----
LOGO_CANDIDATES = ["ag2.jpg", "logo.png", "AGMS_logo.jpg", "ag_logo.jpg"]
logo_path = next((p for p in LOGO_CANDIDATES if os.path.exists(p)), None)

left, mid, right = st.columns([1, 2, 1])
with left:
    if logo_path:
        st.image(logo_path, use_container_width=True)
with mid:
    st.title("Dashboard AGMS: Ventas, Cartera, RFM, Predicción y Cotizaciones")
st.markdown("---")

if not PLOTLY_OK:
    st.error("No se encontró **plotly**. Agrega `plotly` a tu requirements.txt y vuelve a desplegar.")
    st.stop()

# ==============================================================================
# UTILIDADES
# ==============================================================================
def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada fila para convertir conteos a proporciones."""
    if df_counts.empty:
        return df_counts
    sums = df_counts.sum(axis=1).replace(0, 1)
    return df_counts.div(sums, axis=0)

def build_time_derivatives(df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[fecha_col], errors="coerce")
    if "Mes" not in df.columns:
        df["Mes"] = dt.dt.to_period("M").astype(str)
    if "Semana" not in df.columns:
        df["Semana"] = dt.dt.to_period("W").astype(str)
    if "Día" not in df.columns:
        df["Día"] = dt.dt.date
    return df

def limpiar_moneda(valor):
    try:
        if isinstance(valor, str):
            valor_limpio = valor.replace('$', '').replace('.', '').replace(',', '.').strip()
            return float(valor_limpio)
        elif isinstance(valor, (int, float)):
            return float(valor)
        return 0.0
    except (ValueError, TypeError):
        return 0.0

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

def compute_rfm_table(dfv: pd.DataFrame) -> pd.DataFrame:
    if 'FECHA VENTA' not in dfv.columns:
        return pd.DataFrame()
    tmp = dfv.copy()
    tmp['FECHA VENTA'] = pd.to_datetime(tmp['FECHA VENTA'], errors="coerce")
    tmp = tmp.dropna(subset=['FECHA VENTA'])
    ref = tmp['FECHA VENTA'].max()
    tiene_factura = ('NÚMERO DE FACTURA' in tmp.columns)

    rfm = tmp.groupby('Cliente/Empresa').agg(
        Recencia=('FECHA VENTA', lambda s: (ref - s.max()).days),
        Frecuencia=('NÚMERO DE FACTURA','nunique') if tiene_factura else ('FECHA VENTA','count'),
        Monetario=('Total','sum')
    ).reset_index()

    rfm['R_Score'] = _safe_qcut_score(rfm['Recencia'],  ascending=True,  labels=[5,4,3,2,1])
    rfm['F_Score'] = _safe_qcut_score(rfm['Frecuencia'], ascending=False, labels=[1,2,3,4,5])
    rfm['M_Score'] = _safe_qcut_score(rfm['Monetario'],  ascending=False, labels=[1,2,3,4,5])
    rfm['Segmento'] = rfm.apply(rfm_segment, axis=1).fillna("Sin Segmento")
    return rfm

# ===== Utilidades de Productos (precios) =====
def _to_num_price(series_like):
    """
    Convierte precios tipo '$1.234.567,89' o 'No aplica' a float.
    Devuelve NaN cuando no aplica o no puede convertirse.
    """
    s = pd.Series(series_like).astype(str)
    s = (s.str.replace('$', '', regex=False)
           .str.replace('.', '', regex=False)   # miles
           .str.replace(',', '.', regex=False)  # decimales
           .str.strip())
    s = s.mask(s.str.contains(r'no\s*aplica|n/?a', case=False, regex=True), np.nan)
    return pd.to_numeric(s, errors='coerce')

def ensure_product_numeric_cols(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza que existan:
      - _Precio_Medico_num (float)
      - _Precio_Paciente_num (float)
      - NA_Medico (bool) y NA_Paciente (bool)
    aunque cambien los nombres originales de columnas.
    """
    if df_prod is None or df_prod.empty:
        return df_prod

    # mapa de nombres (sin espacios extra)
    name_map = {c.strip(): c for c in df_prod.columns}

    def _find_col(candidates):
        # intento exacto
        for cand in candidates:
            if cand in name_map:
                return name_map[cand]
        # intento normalizado (lower + underscores)
        for k in list(name_map.keys()):
            kn = k.lower().replace(' ', '_')
            if kn in [c.lower().replace(' ', '_') for c in candidates]:
                return name_map[k]
        return None

    col_pm = _find_col(['Precio_Medico', 'Precio Medico', 'PRECIO MEDICO', 'Precio_Médico', 'Precio Médico'])
    col_pp = _find_col(['Precio_Paciente', 'Precio Paciente', 'PRECIO PACIENTE'])

    # numéricos auxiliares
    if '_Precio_Medico_num' not in df_prod.columns:
        df_prod['_Precio_Medico_num'] = _to_num_price(df_prod[col_pm]) if col_pm else np.nan
    if '_Precio_Paciente_num' not in df_prod.columns:
        df_prod['_Precio_Paciente_num'] = _to_num_price(df_prod[col_pp]) if col_pp else np.nan

    # banderas NA_* (No aplica o sin precio)
    if 'NA_Medico' not in df_prod.columns:
        if col_pm:
            df_prod['NA_Medico'] = (
                df_prod[col_pm].astype(str).str.contains(r'no\s*aplica|n/?a', case=False, regex=True)
                | df_prod['_Precio_Medico_num'].isna()
            )
        else:
            df_prod['NA_Medico'] = True

    if 'NA_Paciente' not in df_prod.columns:
        if col_pp:
            df_prod['NA_Paciente'] = (
                df_prod[col_pp].astype(str).str.contains(r'no\s*aplica|n/?a', case=False, regex=True)
                | df_prod['_Precio_Paciente_num'].isna()
            )
        else:
            df_prod['NA_Paciente'] = True

    return df_prod

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================
@st.cache_data
def load_data():
    file_path = 'DB_AGMS.xlsx'
    try:
        df_ventas     = pd.read_excel(file_path, sheet_name='Ventas', header=1)
        df_medicos    = pd.read_excel(file_path, sheet_name='Lista Medicos')
        df_metadatos  = pd.read_excel(file_path, sheet_name='Metadatos')
        df_cartera    = pd.read_excel(file_path, sheet_name='Cartera')      # renombrada
        df_productos  = pd.read_excel(file_path, sheet_name='Productos')    # nueva hoja

        # Ventas
        if 'FECHA VENTA' in df_ventas.columns:
            df_ventas.dropna(subset=['FECHA VENTA'], inplace=True)
            df_ventas['FECHA VENTA'] = pd.to_datetime(df_ventas['FECHA VENTA'], errors='coerce')
            df_ventas['Mes'] = df_ventas['FECHA VENTA'].dt.to_period('M').astype(str)
            df_ventas['Dia_Semana'] = df_ventas['FECHA VENTA'].dt.day_name()
            df_ventas['Hora'] = df_ventas['FECHA VENTA'].dt.hour

        for col in ['Total', 'Cantidad', 'Precio Unidad']:
            if col in df_ventas.columns:
                df_ventas[col] = pd.to_numeric(df_ventas[col], errors='coerce').fillna(0)

        if 'Cliente/Empresa' in df_ventas.columns:
            df_ventas['Cliente/Empresa'] = df_ventas['Cliente/Empresa'].astype(str).str.strip().str.upper()

        if 'Producto' in df_ventas.columns:
            df_ventas['Producto_Nombre'] = df_ventas['Producto'].astype(str).apply(lambda x: x.split(' - ')[0])

        # Médicos
        if 'NOMBRE' in df_medicos.columns:
            df_medicos['NOMBRE'] = df_medicos['NOMBRE'].astype(str).str.strip().str.upper()
        if 'TELEFONO' in df_medicos.columns:
            df_medicos['TELEFONO'] = df_medicos['TELEFONO'].fillna('').astype(str)

        # Cartera: normaliza nombres
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
            df_cartera.dropna(subset=['Fecha de Vencimiento'], inplace=True)
            df_cartera['Fecha de Vencimiento'] = pd.to_datetime(df_cartera['Fecha de Vencimiento'], errors='coerce')
        for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
            if col in df_cartera.columns:
                df_cartera[col] = df_cartera[col].fillna(0).apply(limpiar_moneda)

        # Productos
        prod_rename = {
            'LISTA PRODUCTOS': 'Producto_Nombre',
            'TIPO DE PIEL': 'Tipo_Piel',
            'CONDICION': 'Condicion',
            'PROFESIONAL o RETAIL': 'Canal',
            'Precio Medico': 'Precio_Medico',
            'Precio Paciente': 'Precio_Paciente',
            'Marca': 'Marca'
        }
        df_productos.rename(columns={c: prod_rename.get(c, c) for c in df_productos.columns}, inplace=True)
        if 'Producto_Nombre' in df_productos.columns:
            df_productos['Producto_Nombre'] = df_productos['Producto_Nombre'].astype(str).str.strip()

        # Asegurar columnas numéricas auxiliares y banderas NA_* (robusto)
        df_productos = ensure_product_numeric_cols(df_productos)

        return df_ventas, df_medicos, df_metadatos, df_cartera, df_productos
    except Exception as e:
        st.error(f"Ocurrió un error al leer el Excel: {e}")
        return None, None, None, None, None

df_ventas, df_medicos, df_metadatos, df_cartera, df_productos = load_data()
if df_ventas is None or df_cartera is None or df_productos is None:
    st.stop()

# ==============================================================================
# TABS PRINCIPALES (actualizado: sin "Series" ni "Pareto")
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Análisis de Ventas", "Gestión de Cartera", "Análisis RFM",
    "Modelo Predictivo de Compradores Potenciales", "Cotizaciones"
])

# ---------------------------------------------------------------------------------
# TAB 1: ANÁLISIS DE VENTAS (actualizado)
# ---------------------------------------------------------------------------------
with tab1:
    st.header("Análisis General de Ventas")

    # Usamos la fecha real de ventas
    if 'FECHA VENTA' in df_ventas.columns:
        dv = df_ventas.copy()
        dv['FECHA VENTA'] = pd.to_datetime(dv['FECHA VENTA'], errors='coerce')
        dv = dv.dropna(subset=['FECHA VENTA'])
    else:
        st.warning("No se encuentra la columna 'FECHA VENTA' en ventas.")
        dv = df_ventas.copy()

    # =========================
    # KPIs superiores
    # =========================
    # Ventas 2024 → hoy
    inicio_2024 = pd.Timestamp(2024, 1, 1)
    hoy = pd.Timestamp.today().normalize()
    mask_2024 = (dv['FECHA VENTA'] >= inicio_2024) & (dv['FECHA VENTA'] <= hoy)
    total_2024_hoy = float(dv.loc[mask_2024, 'Total'].sum()) if 'Total' in dv.columns else 0.0

    # Totales generales
    total_ventas = float(dv['Total'].sum()) if 'Total' in dv.columns else 0.0
    total_transacciones = len(dv)
    clientes_unicos = dv['Cliente/Empresa'].nunique() if 'Cliente/Empresa' in dv.columns else 0
    ticket_prom = total_ventas / total_transacciones if total_transacciones else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventas 2024 → Hoy", f"${total_2024_hoy:,.0f}")
    c2.metric("Ventas Totales (todo el histórico)", f"${total_ventas:,.0f}")
    c3.metric("Clientes Únicos", f"{clientes_unicos:,}")
    c4.metric("Ticket Promedio", f"${ticket_prom:,.0f}")
    st.markdown("---")

    # =========================
    # Valor por período (Año / Mes / Semana / Día)
    # =========================
    with st.expander("🔍 Consultar valor por período (Año / Mes / Semana / Día)"):
        colp = st.columns(4)
        granular_op = colp[0].radio("Periodo", options=["Año", "Mes", "Semana", "Día"], index=0, horizontal=True, key="t1_kpi_periodo")

        if granular_op == "Año":
            serie = dv.groupby(dv['FECHA VENTA'].dt.year, as_index=False)['Total'].sum().rename(columns={'FECHA VENTA':'Año'})
            options = serie['FECHA VENTA'].astype(int).tolist() if 'FECHA VENTA' in serie.columns else serie.iloc[:,0].tolist()
            sel = colp[1].selectbox("Selecciona año", options=sorted(options), key="t1_kpi_year")
            valor = float(serie.loc[serie.iloc[:,0]==sel, 'Total'].sum())
            colp[2].metric("Valor seleccionado", f"${valor:,.0f}")

        elif granular_op == "Mes":
            serie = dv.groupby(dv['FECHA VENTA'].dt.to_period("M"), as_index=False)['Total'].sum()
            serie['Periodo'] = serie['FECHA VENTA'].astype(str)
            sel = colp[1].selectbox("Selecciona mes (YYYY-MM)", options=sorted(serie['Periodo'].tolist()), key="t1_kpi_month")
            valor = float(serie.loc[serie['Periodo']==sel, 'Total'].sum())
            colp[2].metric("Valor seleccionado", f"${valor:,.0f}")

        elif granular_op == "Semana":
            serie = dv.groupby(dv['FECHA VENTA'].dt.to_period("W"), as_index=False)['Total'].sum()
            serie['Periodo'] = serie['FECHA VENTA'].astype(str)
            sel = colp[1].selectbox("Selecciona semana (YYYY-Wxx)", options=sorted(serie['Periodo'].tolist()), key="t1_kpi_week")
            valor = float(serie.loc[serie['Periodo']==sel, 'Total'].sum())
            colp[2].metric("Valor seleccionado", f"${valor:,.0f}")

        else:  # Día
            serie = dv.groupby(dv['FECHA VENTA'].dt.date, as_index=False)['Total'].sum().rename(columns={'FECHA VENTA':'Fecha'})
            sel = colp[1].date_input("Selecciona día", value=serie['Fecha'].max() if not serie.empty else hoy, key="t1_kpi_day")
            valor = float(serie.loc[serie['Fecha']==pd.to_datetime(sel).date(), 'Total'].sum()) if not serie.empty else 0.0
            colp[2].metric("Valor seleccionado", f"${valor:,.0f}")

    st.markdown("---")

    # =========================
    # Evolución temporal (dinámica por ventana y agrupación)
    # =========================
    st.subheader("Evolución temporal")
    cA, cB = st.columns([2, 1])
    # Ventana de fechas
    min_date = dv['FECHA VENTA'].min() if 'FECHA VENTA' in dv.columns else pd.Timestamp(2024,1,1)
    max_date = dv['FECHA VENTA'].max() if 'FECHA VENTA' in dv.columns else hoy
    date_range = cA.date_input("Ventana de fechas", value=[min_date.date(), max_date.date()], key="t1_range")
    # Agrupación solo para esta gráfica
    agrup = cB.radio("Agrupar por", options=["Día","Semana","Mes","Año"], index=2, horizontal=True, key="t1_agg")

    if isinstance(date_range, (list, tuple)) and len(date_range)==2:
        f_ini = pd.to_datetime(date_range[0])
        f_fin = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # inclusive
        dvg = dv[(dv['FECHA VENTA']>=f_ini) & (dv['FECHA VENTA']<f_fin)].copy()
    else:
        dvg = dv.copy()

    if not dvg.empty:
        if agrup == "Día":
            serie = dvg.groupby(dvg['FECHA VENTA'].dt.date, as_index=False)['Total'].sum().rename(columns={'FECHA VENTA':'Fecha'})
            xcol = 'FECHA VENTA' if 'FECHA VENTA' in serie.columns else serie.columns[0]
            fig = px.line(serie, x=serie.columns[0], y="Total", markers=True, title="")
        elif agrup == "Semana":
            serie = dvg.groupby(dvg['FECHA VENTA'].dt.to_period("W"), as_index=False)['Total'].sum()
            serie['Periodo'] = serie['FECHA VENTA'].astype(str)
            fig = px.line(serie, x='Periodo', y='Total', markers=True, title="")
        elif agrup == "Mes":
            serie = dvg.groupby(dvg['FECHA VENTA'].dt.to_period("M"), as_index=False)['Total'].sum()
            serie['Periodo'] = serie['FECHA VENTA'].astype(str)
            fig = px.line(serie, x='Periodo', y='Total', markers=True, title="")
        else:  # Año
            serie = dvg.groupby(dvg['FECHA VENTA'].dt.year, as_index=False)['Total'].sum().rename(columns={'FECHA VENTA':'Año'})
            fig = px.line(serie, x=serie.columns[0], y='Total', markers=True, title="")

        st.plotly_chart(fig, use_container_width=True, key="t1_line_window")

    st.markdown("---")

    # =========================
    # Subpestañas: Resumen, Productos, Clientes, Mapa de calor
    # =========================
    tab_r1, tab_r3, tab_r4, tab_r6 = st.tabs(
        ["Resumen", "Productos", "Clientes", "Mapa de calor"]
    )

    # -------- Resumen (Top por dimensión simple) --------
    with tab_r1:
        # Top por Producto (gráfica + tabla)
        if "Producto_Nombre" in dv.columns:
            top_n_prod = st.selectbox("Top-N Productos", options=[5,10,15,20,30], index=1, key="t1_top_prod")
            prod = dv.groupby("Producto_Nombre", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
            top_prod = prod.head(top_n_prod)
            cA1, cB1 = st.columns(2)
            with cA1:
                figp = px.bar(top_prod, x="Total", y="Producto_Nombre", orientation="h", title=f"Top {top_n_prod} Productos")
                figp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(figp, use_container_width=True, key="t1_prod_bar")
            with cB1:
                st.dataframe(top_prod, use_container_width=True)

    # -------- Productos (dinámico usando hoja Productos) --------
    with tab_r3:
        st.caption("Explora el catálogo desde la hoja **Productos**")
        if df_productos is None or df_productos.empty or 'Producto_Nombre' not in df_productos.columns:
            st.warning("No se encontró la hoja 'Productos' con el formato esperado.")
        else:
            cat = df_productos.copy()
            # Filtros
            colf1, colf2, colf3 = st.columns(3)
            marcas  = sorted(cat['Marca'].dropna().unique().tolist()) if 'Marca' in cat.columns else []
            canales = sorted(cat['Canal'].dropna().unique().tolist()) if 'Canal' in cat.columns else []
            tpiel   = sorted(cat['Tipo_Piel'].dropna().unique().tolist()) if 'Tipo_Piel' in cat.columns else []
            marca_sel  = colf1.multiselect("Marca", options=marcas, key="t1p_marca")
            canal_sel  = colf2.multiselect("Canal", options=canales, key="t1p_canal")
            tpiel_sel  = colf3.multiselect("Tipo de piel", options=tpiel, key="t1p_tpiel")

            if marca_sel and 'Marca' in cat.columns:
                cat = cat[cat['Marca'].isin(marca_sel)]
            if canal_sel and 'Canal' in cat.columns:
                cat = cat[cat['Canal'].isin(canal_sel)]
            if tpiel_sel and 'Tipo_Piel' in cat.columns:
                cat = cat[cat['Tipo_Piel'].isin(tpiel_sel)]

            # Selección de variable categórica para distribución
            vars_cat = [c for c in ['Marca','Canal','Tipo_Piel','Condicion'] if c in cat.columns]
            colg1, colg2 = st.columns(2)
            if vars_cat:
                vcat = colg1.selectbox("Distribución por", options=vars_cat, index=0, key="t1p_varcat")
                dist = cat[vcat].value_counts().reset_index()
                dist.columns = [vcat, 'Conteo']
                figc = px.bar(dist, x='Conteo', y=vcat, orientation='h', title=f"Distribución por {vcat}")
                figc.update_layout(yaxis={'categoryorder':'total ascending'})
                colg1.plotly_chart(figc, use_container_width=True, key="t1p_dist")

            # Exploración de precios (ignora 'No aplica')
            colg2.subheader("Precios (Médico vs Paciente)")
            cat_aux = cat.copy()
            # Asegurar auxiliares numéricas
            if '_Precio_Medico_num' not in cat_aux.columns or '_Precio_Paciente_num' not in cat_aux.columns:
                from math import isnan
                cat_aux = ensure_product_numeric_cols(cat_aux)
            pm = cat_aux['_Precio_Medico_num'].replace(0, np.nan)
            pp = cat_aux['_Precio_Paciente_num'].replace(0, np.nan)

            dfp = pd.DataFrame({
                "Precio": pd.concat([pm.dropna(), pp.dropna()], ignore_index=True),
                "Tipo":   ["Médico"]*pm.dropna().shape[0] + ["Paciente"]*pp.dropna().shape[0]
            })
            if not dfp.empty:
                figp2 = px.box(dfp, x="Tipo", y="Precio", points="suspectedoutliers", title="Distribución de precios")
                colg2.plotly_chart(figp2, use_container_width=True, key="t1p_price_box")
            st.dataframe(
                cat[[c for c in ['Producto_Nombre','Marca','Canal','Tipo_Piel','Condicion','Precio_Medico','Precio_Paciente'] if c in cat.columns]],
                use_container_width=True
            )

    # -------- Clientes (Top-N con selector y layout lado a lado) --------
    with tab_r4:
        if "Cliente/Empresa" in dv.columns:
            top_n_cli = st.selectbox("Top-N Clientes", options=[5,10,15,20,30], index=1, key="t1_top_cli")
            cli = dv.groupby("Cliente/Empresa", as_index=False)["Total"].sum().sort_values("Total", ascending=False)
            top_cli = cli.head(top_n_cli)
            cC1, cC2 = st.columns(2)
            with cC1:
                figc = px.bar(top_cli, x="Total", y="Cliente/Empresa", orientation="h", title=f"Top {top_n_cli} Clientes")
                figc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(figc, use_container_width=True, key="t1_cli_bar")
            with cC2:
                st.dataframe(top_cli, use_container_width=True)

    # -------- Mapa de calor (se mantiene) --------
    with tab_r6:
        if 'FECHA VENTA' in dv.columns:
            work = dv.copy()
            dt = work['FECHA VENTA']
            work["Mes"] = dt.dt.to_period("M").astype(str)
            work["DiaSemana"] = dt.dt.day_name()
            heat = work.groupby(["DiaSemana","Mes"], as_index=False)["Total"].sum()
            orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            heat["DiaSemana"] = pd.Categorical(heat["DiaSemana"], categories=orden_dias, ordered=True)
            heat_pivot = heat.pivot(index="DiaSemana", columns="Mes", values="Total").fillna(0)
            st.plotly_chart(px.imshow(heat_pivot, aspect="auto", title="Heatmap (Día x Mes)"),
                            use_container_width=True, key="t1_heatmap")
        else:
            st.info("No es posible generar el mapa de calor sin 'FECHA VENTA'.")
# ---------------------------------------------------------------------------------
# TAB 2: GESTIÓN DE CARTERA (lee Cartera y une COMERCIAL por factura)
# ---------------------------------------------------------------------------------
with tab2:
    st.header("Gestión de Cartera")

    dfc = df_cartera.copy()

    if 'Fecha de Vencimiento' in dfc.columns:
        dfc['Fecha de Vencimiento'] = pd.to_datetime(dfc['Fecha de Vencimiento'], errors='coerce')

    for col in ['Deuda por cobrar', 'Cantidad Abonada', 'Saldo pendiente']:
        if col in dfc.columns:
            dfc[col] = dfc[col].fillna(0).apply(limpiar_moneda)

    # Unir COMERCIAL desde ventas por NÚMERO DE FACTURA
    if 'NÚMERO DE FACTURA' in dfc.columns and 'NÚMERO DE FACTURA' in df_ventas.columns:
        dfc['NÚMERO DE FACTURA'] = dfc['NÚMERO DE FACTURA'].astype(str).str.strip()
        dfv_fact = df_ventas[['NÚMERO DE FACTURA', 'COMERCIAL']].dropna(subset=['NÚMERO DE FACTURA']).copy()
        dfv_fact['NÚMERO DE FACTURA'] = dfv_fact['NÚMERO DE FACTURA'].astype(str).str.strip()
        dfc = dfc.merge(dfv_fact.drop_duplicates(), on='NÚMERO DE FACTURA', how='left')
    else:
        dfc['COMERCIAL'] = "No disponible"

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
        if row['Estado'] == 'Vencida':
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
# TAB 3: ANÁLISIS RFM + Recomendador ML (segmentos & día)
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

                # Producto sugerido (más comprado históricamente)
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
                DS['Cliente/Empresa'] = DS.index.astype(str)  # etiqueta para exportar
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
# TAB 5: COTIZACIONES (usa hoja Productos; maneja "No aplica" y evita KeyError)
# ---------------------------------------------------------------------------------
with tab5:
    st.header("🧾 Cotizaciones")

    if df_productos is None or df_productos.empty or 'Producto_Nombre' not in df_productos.columns:
        st.warning("No se encontró la hoja 'Productos' con el formato esperado.")
        st.stop()

    # Copia + asegurar columnas requeridas (robusto a variaciones en el Excel)
    catalog = ensure_product_numeric_cols(df_productos.copy())

    # Relleno 0 SOLO para cálculo; la UI respeta 'No aplica'
    catalog['_Precio_Medico_num']   = catalog['_Precio_Medico_num'].fillna(0.0)
    catalog['_Precio_Paciente_num'] = catalog['_Precio_Paciente_num'].fillna(0.0)

    # Si por alguna razón no se crearon banderas, créalas ahora
    if 'NA_Medico' not in catalog.columns:
        catalog['NA_Medico'] = catalog['_Precio_Medico_num'].isna() | (catalog['_Precio_Medico_num'] <= 0)
    if 'NA_Paciente' not in catalog.columns:
        catalog['NA_Paciente'] = catalog['_Precio_Paciente_num'].isna() | (catalog['_Precio_Paciente_num'] <= 0)

    # Selector de productos (autocompletable)
    opciones = sorted(catalog['Producto_Nombre'].astype(str).unique().tolist())
    sel = st.multiselect(
        "Agrega productos a la cotización (escribe iniciales para buscar)",
        options=opciones, key="cot_sel"
    )

    # Estado de cotización en sesión
    if "cot_items" not in st.session_state:
        st.session_state["cot_items"] = {}  # { producto: {"qty": int, "price_type": "Medico"/"Paciente"} }

    # Sincronizar selección con sesión
    for p in sel:
        if p not in st.session_state["cot_items"]:
            row_p = catalog.loc[catalog['Producto_Nombre'] == p].iloc[0]
            # default según disponibilidad
            if not bool(row_p['NA_Medico']) and bool(row_p['NA_Paciente']):
                default_ptype = "Medico"
            elif bool(row_p['NA_Medico']) and not bool(row_p['NA_Paciente']):
                default_ptype = "Paciente"
            elif not bool(row_p['NA_Medico']) and not bool(row_p['NA_Paciente']):
                default_ptype = "Medico"
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

        # Encabezados
        hdr = st.columns([4, 2, 3, 3, 3])
        hdr[0].markdown("**Producto**")
        hdr[1].markdown("**Cantidad**")
        hdr[2].markdown("**Tipo de precio**")
        hdr[3].markdown("**Precio unitario**")
        hdr[4].markdown("**Subtotal**")

        for p in sel:
            row = catalog.loc[catalog['Producto_Nombre'] == p].iloc[0]
            na_med = bool(row['NA_Medico'])
            na_pac = bool(row['NA_Paciente'])

            if na_med and na_pac:
                valid_opts = []
            elif na_med and not na_pac:
                valid_opts = ["Paciente"]
            elif not na_med and na_pac:
                valid_opts = ["Medico"]
            else:
                valid_opts = ["Medico", "Paciente"]

            c0, c1, c2, c3, c4 = st.columns([4, 2, 3, 3, 3])
            c0.write(p)

            # cantidad
            qty_key = f"qty_{p}"
            qty_val = st.session_state["cot_items"][p]["qty"]
            qty = c1.number_input(" ", min_value=1, step=1, value=qty_val, key=qty_key)
            st.session_state["cot_items"][p]["qty"] = qty

            # tipo de precio
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

            # precio unitario numérico para cálculo (auxiliares robustas)
            punit = 0.0
            if st.session_state["cot_items"][p]["price_type"] == "Medico":
                punit = float(row.get('_Precio_Medico_num', 0.0))
            elif st.session_state["cot_items"][p]["price_type"] == "Paciente":
                punit = float(row.get('_Precio_Paciente_num', 0.0))

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
