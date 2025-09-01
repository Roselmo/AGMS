
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="AGMS Ventas Dashboard", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_excel(path_or_url: str) -> dict:
    # Allow both local path and raw GitHub URLs
    dfs = {}
    xls = pd.ExcelFile(path_or_url, engine="openpyxl")
    for sh in ["Ventas", "Lista Medicos", "Metadatos", "CarteraAgosto"]:
        if sh in xls.sheet_names:
            dfs[sh] = pd.read_excel(xls, sheet_name=sh)
    return dfs

def clean_ventas(df: pd.DataFrame) -> pd.DataFrame:
    # The first data row contains the actual headers
    if df.shape[0] == 0:
        return df
    header = df.iloc[0].tolist()
    df = df.iloc[1:].copy()
    df.columns = [str(x).strip() for x in header]
    # Standardize expected columns
    rename_map = {
        "FECHA VENTA": "FECHA_VENTA",
        "Hora": "HORA",
        "Cliente/Empresa": "CLIENTE",
        "Producto": "PRODUCTO",
        "Precio Unidad": "PRECIO_UNIDAD",
        "Cantidad": "CANTIDAD",
        "Total": "TOTAL",
        "NÚMERO DE FACTURA": "FACTURA",
        "COMERCIAL": "COMERCIAL",
    }
    for k,v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Parse dtypes
    if "FECHA_VENTA" in df.columns:
        df["FECHA_VENTA"] = pd.to_datetime(df["FECHA_VENTA"], errors="coerce")
        df["AÑO"] = df["FECHA_VENTA"].dt.year
        df["MES"] = df["FECHA_VENTA"].dt.to_period("M").astype(str)
        df["DIA"] = df["FECHA_VENTA"].dt.date
        df["SEMANA"] = df["FECHA_VENTA"].dt.to_period("W").apply(lambda p: str(p))
    if "CANTIDAD" in df.columns:
        df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce").fillna(0).astype(int)
    if "PRECIO_UNIDAD" in df.columns:
        df["PRECIO_UNIDAD"] = pd.to_numeric(df["PRECIO_UNIDAD"], errors="coerce")
    if "TOTAL" in df.columns:
        df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")

    # Drop rows without product or date
    if "PRODUCTO" in df.columns:
        df = df[df["PRODUCTO"].notna()]
    if "FECHA_VENTA" in df.columns:
        df = df[df["FECHA_VENTA"].notna()]

    # Normalize text columns
    for c in ["CLIENTE", "PRODUCTO", "COMERCIAL", "FACTURA"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)

def clean_lista(df: pd.DataFrame) -> pd.DataFrame:
    # Keep expected columns if present
    cols = df.columns.tolist()
    # Normalize column names common in the user's file
    rename_map = {
        "NOMBRE": "NOMBRE",
        "NOMBRE COMPLETO": "NOMBRE",
        "CEDULA": "CEDULA",
        "ESPECIALIDAD MEDICA": "ESPECIALIDAD_MEDICA",
        "TELEFONO": "TELEFONO",
        "DIRECCION": "DIRECCION",
        "EMAIL": "EMAIL",
        "CIUDAD": "CIUDAD",
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Drop obvious header/blank rows
    key_cols = [c for c in ["NOMBRE","CEDULA","ESPECIALIDAD_MEDICA","EMAIL"] if c in df.columns]
    if key_cols:
        df = df.dropna(how="all", subset=key_cols)
    # Normalize
    if "NOMBRE" in df.columns:
        df["NOMBRE_NORM"] = df["NOMBRE"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)

def clean_cartera(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize common columns
    rename_map = {
        "Nombre cliente": "CLIENTE",
        "Deuda por cobrar": "DEUDA",
        "NÚMERO DE FACTURA": "FACTURA",
        "Fecha de creación de la factura": "FECHA_CREACION",
        "Fecha de Vencimiento": "FECHA_VENCIMIENTO",
        "Cantidad Abonada": "ABONADO",
        "Saldo pendiente": "SALDO",
        "Fecha de pago": "FECHA_PAGO",
    }
    for k,v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    for c in ["FECHA_CREACION","FECHA_VENCIMIENTO","FECHA_PAGO"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["DEUDA","ABONADO","SALDO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "CLIENTE" in df.columns:
        df["CLIENTE"] = df["CLIENTE"].astype(str).str.strip()
    return df.reset_index(drop=True)

def rfm_score(df_ventas: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if df_ventas.empty:
        return pd.DataFrame()
    if ref_date is None:
        ref_date = df_ventas["FECHA_VENTA"].max() + pd.Timedelta(days=1)

    grp = df_ventas.groupby("CLIENTE").agg(
        Recency=("FECHA_VENTA", lambda s: (ref_date - s.max()).days),
        Frequency=("FACTURA", pd.Series.nunique) if "FACTURA" in df_ventas.columns else ("CLIENTE","size"),
        Monetary=("TOTAL", "sum")
    ).reset_index()

    # Quintile scoring
    grp["R_Score"] = pd.qcut(grp["Recency"].rank(method="first", ascending=True), 5, labels=[5,4,3,2,1]).astype(int)
    grp["F_Score"] = pd.qcut(grp["Frequency"].rank(method="first", ascending=False), 5, labels=[1,2,3,4,5]).astype(int)
    grp["M_Score"] = pd.qcut(grp["Monetary"].rank(method="first", ascending=False), 5, labels=[1,2,3,4,5]).astype(int)
    grp["RFM_Sum"] = grp["R_Score"] + grp["F_Score"] + grp["M_Score"]

    def rfm_segment(row):
        r,f,m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 3 and f >= 3 and m >= 3:
            return "Potential Loyalist"
        if r <= 2 and f >= 4:
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        if r >= 3 and f <= 2:
            return "New"
        return "Need Attention"

    grp["Segmento"] = grp.apply(rfm_segment, axis=1)
    return grp.sort_values(["RFM_Sum","Monetary"], ascending=[False, False]).reset_index(drop=True)

def norm_text(s):
    return str(s).strip().lower() if pd.notna(s) else ""

# ---------------------------
# Sidebar: data source
# ---------------------------
st.sidebar.header("Fuente de datos")
gh_raw = st.sidebar.text_input("URL RAW de GitHub (Excel)", value=os.getenv("AGMS_EXCEL_URL", ""),
                               help="Pegue aquí la URL RAW del archivo DB_AGMS.xlsx en su repositorio. "
                                    "Si se deja vacío, la app intentará usar un archivo local llamado DB_AGMS.xlsx.")

local_path = st.sidebar.text_input("Ruta local (opcional)", value="DB_AGMS.xlsx")
path = gh_raw.strip() or local_path

try:
    dfs = load_excel(path)
    ventas_raw = dfs.get("Ventas", pd.DataFrame())
    lista_raw = dfs.get("Lista Medicos", pd.DataFrame())
    cartera_raw = dfs.get("CarteraAgosto", pd.DataFrame())
except Exception as e:
    st.error(f"No pude cargar el Excel desde: {path}. Error: {e}")
    st.stop()

ventas = clean_ventas(ventas_raw.copy())
lista_med = clean_lista(lista_raw.copy())
cartera = clean_cartera(cartera_raw.copy())

if ventas.empty:
    st.warning("La hoja 'Ventas' no tiene datos procesables.")
    st.stop()

# ---------------------------
# Sidebar: filtros
# ---------------------------
st.sidebar.header("Filtros")
min_date = ventas["FECHA_VENTA"].min()
max_date = ventas["FECHA_VENTA"].max()
rango_fechas = st.sidebar.date_input("Rango de fechas", value=(min_date.date(), max_date.date()),
                                     min_value=min_date.date(), max_value=max_date.date())

clientes = sorted(ventas["CLIENTE"].dropna().unique().tolist())
sel_clientes = st.sidebar.multiselect("Clientes / Médicos", clientes)

comerciales = sorted(ventas["COMERCIAL"].dropna().unique().tolist()) if "COMERCIAL" in ventas.columns else []
sel_comerciales = st.sidebar.multiselect("Comerciales", comerciales) if comerciales else []

productos = sorted(ventas["PRODUCTO"].dropna().unique().tolist())
sel_productos = st.sidebar.multiselect("Productos", productos)

# Aplicar filtros
mask = (ventas["FECHA_VENTA"].dt.date >= rango_fechas[0]) & (ventas["FECHA_VENTA"].dt.date <= rango_fechas[1])
if sel_clientes:
    mask &= ventas["CLIENTE"].isin(sel_clientes)
if sel_comerciales:
    mask &= ventas["COMERCIAL"].isin(sel_comerciales)
if sel_productos:
    mask &= ventas["PRODUCTO"].isin(sel_productos)

v = ventas.loc[mask].copy()

# ---------------------------
# KPIs
# ---------------------------
st.title("AGMS • Dashboard de Ventas")
col1, col2, col3, col4 = st.columns(4)
total_ventas = v["TOTAL"].sum() if "TOTAL" in v.columns else 0
tickets = v["FACTURA"].nunique() if "FACTURA" in v.columns else v.shape[0]
clientes_unicos = v["CLIENTE"].nunique()
avg_ticket = (total_ventas / tickets) if tickets else 0
col1.metric("Ventas (COP)", f"${total_ventas:,.0f}")
col2.metric("Facturas", f"{tickets:,}")
col3.metric("Clientes únicos", f"{clientes_unicos:,}")
col4.metric("Ticket promedio", f"${avg_ticket:,.0f}")

st.markdown("---")

# ---------------------------
# Time series
# ---------------------------
st.subheader("Evolución temporal")
tab_dia, tab_sem, tab_mes = st.tabs(["Por día", "Por semana", "Por mes"])

with tab_dia:
    by_day = v.groupby("DIA", as_index=False).agg(Ventas=("TOTAL","sum"))
    fig = px.line(by_day, x="DIA", y="Ventas", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_sem:
    if "SEMANA" in v.columns:
        by_week = v.groupby("SEMANA", as_index=False).agg(Ventas=("TOTAL","sum"))
        fig = px.bar(by_week, x="SEMANA", y="Ventas")
        st.plotly_chart(fig, use_container_width=True)

with tab_mes:
    if "MES" in v.columns:
        by_month = v.groupby("MES", as_index=False).agg(Ventas=("TOTAL","sum"))
        fig = px.bar(by_month, x="MES", y="Ventas")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Productos y clientes
# ---------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Ventas por producto")
    top_prod = v.groupby("PRODUCTO", as_index=False).agg(Ventas=("TOTAL","sum"), Cantidad=("CANTIDAD","sum"))
    top_prod = top_prod.sort_values("Ventas", ascending=False)
    fig = px.bar(top_prod.head(30), x="PRODUCTO", y="Ventas", hover_data=["Cantidad"])
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_prod)

with c2:
    st.subheader("Top clientes / médicos")
    top_cli = v.groupby("CLIENTE", as_index=False).agg(Ventas=("TOTAL","sum"), Facturas=("FACTURA","nunique"))
    top_cli = top_cli.sort_values("Ventas", ascending=False)
    fig = px.bar(top_cli.head(30), x="CLIENTE", y="Ventas", hover_data=["Facturas"])
    fig.update_layout(xaxis=dict(tickangle=-45))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_cli)

st.markdown("---")

# ---------------------------
# RFM
# ---------------------------
st.subheader("Análisis RFM")
do_rfm = st.button("Calcular y visualizar RFM")
if do_rfm:
    rfm = rfm_score(v if not v.empty else ventas)
    st.success(f"RFM calculado para {len(rfm)} clientes.")
    segs = sorted(rfm["Segmento"].unique().tolist())
    seg_sel = st.multiselect("Segmentos", segs, default=segs)
    rfm_view = rfm[rfm["Segmento"].isin(seg_sel)].copy()

    colA, colB = st.columns([2,1])
    with colA:
        fig = px.scatter(rfm_view, x="Frequency", y="Monetary",
                         color="Segmento", size="Monetary", hover_name="CLIENTE",
                         title="RFM: Frequency vs Monetary")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        seg_summary = rfm_view.groupby("Segmento", as_index=False).agg(
            Clientes=("CLIENTE","count"),
            Ventas=("Monetary","sum"),
            FrecuenciaProm=("Frequency","mean"),
            RecencyMed=("Recency","median")
        ).sort_values("Ventas", ascending=False)
        st.dataframe(seg_summary, use_container_width=True)

    st.download_button("Descargar tabla RFM (CSV)", rfm.to_csv(index=False).encode("utf-8"),
                       file_name="rfm_agms.csv", mime="text/csv")

st.markdown("---")

# ---------------------------
# Prospectos por especialidad
# ---------------------------
st.subheader("Potenciales compradores por ESPECIALIDAD MÉDICA")
# Clientes que YA compraron (normalizados)
compradores = v["CLIENTE"].dropna().astype(str).str.strip().str.lower().unique().tolist()
compradores_set = set(compradores)

if not lista_med.empty and "NOMBRE_NORM" in lista_med.columns:
    lista_med["YA_COMPRO"] = lista_med["NOMBRE_NORM"].apply(lambda x: x in compradores_set)
    potenciales = lista_med[~lista_med["YA_COMPRO"]].copy()

    if "ESPECIALIDAD_MEDICA" in potenciales.columns:
        esp_options = ["(todas)"] + sorted([e for e in potenciales["ESPECIALIDAD_MEDICA"].dropna().unique().tolist() if str(e).strip()])
        esp_sel = st.selectbox("Filtrar por especialidad", esp_options, index=0)
        if esp_sel != "(todas)":
            potenciales = potenciales[potenciales["ESPECIALIDAD_MEDICA"] == esp_sel]

    st.info(f"Se identificaron {len(potenciales)} potenciales compradores que aún no han comprado.")
    st.dataframe(potenciales.drop(columns=["NOMBRE_NORM"], errors="ignore"), use_container_width=True)
    st.download_button("Descargar potenciales (CSV)", potenciales.to_csv(index=False).encode("utf-8"),
                       file_name="potenciales_por_especialidad.csv", mime="text/csv")
else:
    st.warning("No se pudo procesar 'Lista Medicos' para comparar compradores vs base de médicos.")

# ---------------------------
# Cartera (opcional si hay datos)
# ---------------------------
if not cartera.empty:
    st.markdown("---")
    st.subheader("Cartera y vencimientos")
    colx, coly = st.columns(2)
    with colx:
        by_cli = cartera.groupby("CLIENTE", as_index=False).agg(
            Deuda=("DEUDA","sum"),
            Abonado=("ABONADO","sum"),
            Saldo=("SALDO","sum"),
        ).sort_values("Saldo", ascending=False)
        fig = px.bar(by_cli.head(25), x="CLIENTE", y="Saldo", hover_data=["Deuda","Abonado"])
        fig.update_layout(xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_cli, use_container_width=True)

    with coly:
        if "FECHA_VENCIMIENTO" in cartera.columns:
            cartera["DIAS_VENCIDOS"] = (pd.Timestamp.today().normalize() - cartera["FECHA_VENCIMIENTO"]).dt.days
            buckets = pd.cut(cartera["DIAS_VENCIDOS"],
                             bins=[-1, 0, 30, 60, 90, 180, 365, 10_000],
                             labels=["Al día","1-30","31-60","61-90","91-180","181-365","+365"])
            venc = cartera.groupby(buckets, as_index=False).agg(Saldo=("SALDO","sum"))
            fig = px.bar(venc, x="DIAS_VENCIDOS", y="Saldo", title="Antigüedad de saldos")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cartera[["CLIENTE","FACTURA","FECHA_CREACION","FECHA_VENCIMIENTO","SALDO","DIAS_VENCIDOS"]]
                         .sort_values("DIAS_VENCIDOS", ascending=False), use_container_width=True)

st.caption("© AG Medical Solutions — Dashboard de ventas y clientes.")
