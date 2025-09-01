# AGMS Ventas Dashboard (Streamlit)

Panel interactivo para analizar ventas, clientes/médicos, productos y cartera desde un Excel (`DB_AGMS.xlsx`) alojado en GitHub o local. Incluye:
- KPIs (ventas, facturas, clientes únicos, ticket promedio)
- Series de tiempo por día/semana/mes
- Ventas por producto y top clientes
- **RFM** por botón (segmentos: Champions, Loyal, At Risk, etc.) con descarga CSV
- Detección de **potenciales compradores** por **ESPECIALIDAD_MEDICA** (comparando ventas vs `Lista Medicos`)
- (Opcional) **Cartera** y envejecimiento de saldos si existe la hoja `CarteraAgosto`

---

## 1) Estructura esperada

```
.
├── app.py
├── requirements.txt
└── DB_AGMS.xlsx          # opcional si no usas URL RAW
```

El Excel debe tener estas hojas: **Ventas**, **Lista Medicos**, **Metadatos**, **CarteraAgosto**.

> La app también admite un **URL RAW de GitHub** para `DB_AGMS.xlsx`.

---

## 2) Ejecutar en local

```bash
# 1) Crear y activar entorno (opcional)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Ejecutar
streamlit run app.py
```

En la barra lateral:
- Pega el **URL RAW** del Excel, o
- Deja vacío y usa `DB_AGMS.xlsx` local en el directorio del proyecto.

---

## 3) Despliegue en Streamlit Cloud

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io/) y **New app**.
3. Selecciona tu repo y rama; en **Main file path** coloca: `app.py`.
4. (Opcional) En **Advanced settings**, agrega una **Environment variable**:
   - `AGMS_EXCEL_URL` = `https://raw.githubusercontent.com/<usuario>/<repo>/<branch>/DB_AGMS.xlsx`
5. Deploy.

> También puedes pegar el URL RAW directamente en la barra lateral sin usar variables de entorno.

### ¿Cómo obtener el URL RAW?
En tu repositorio en GitHub:
1. Abre `DB_AGMS.xlsx`.
2. Click en botón **Raw**.
3. Copia la URL del navegador (comienza con `https://raw.githubusercontent.com/...`).

---

## 4) Formato y limpieza de datos (resumen)

### Hoja **Ventas**
- La primera fila se usa como encabezados reales (la app lo corrige).
- Columnas estándar que el script intenta estandarizar:  
  `FECHA_VENTA, HORA, CLIENTE, PRODUCTO, PRECIO_UNIDAD, CANTIDAD, TOTAL, FACTURA, COMERCIAL`
- Se calculan columnas auxiliares: `AÑO, MES, DIA, SEMANA`.

### Hoja **Lista Medicos**
- Normaliza `NOMBRE` → `NOMBRE_NORM` (lower/trim).
- Se usa para cruzar compradores vs. base y generar **potenciales** filtrables por `ESPECIALIDAD_MEDICA`.

### Hoja **CarteraAgosto** (opcional)
- Si existen fechas de vencimiento, calcula `DIAS_VENCIDOS` y agrupa por buckets (Al día, 1-30, 31-60, ...).

---

## 5) Troubleshooting

- **ModuleNotFoundError: No module named 'plotly'**  
  Asegúrate de que `requirements.txt` esté en la **raíz** del repo y contenga `plotly`. En local: `pip install -r requirements.txt`.

- **El Excel no carga (URL RAW)**  
  - Verifica que el archivo sea público o usa **Secrets**/variable de entorno si es privado.
  - Comprueba que la URL comience con `https://raw.githubusercontent.com/` y apunte a la rama correcta.

- **Datos vacíos o columnas distintas**  
  - Revisa que `Ventas` tenga fechas válidas en `FECHA VENTA` (o encabezado equivalente).
  - La primera fila debe ser el *header real*. La app ajusta esto automáticamente.

---

## 6) Personalización

- Ajusta los nombres de columnas en el diccionario `rename_map` dentro de `clean_ventas`, `clean_lista` y `clean_cartera` si tus encabezados cambian.
- Para matching más robusto entre compradores y médicos, puedes ampliar la normalización (remover tildes, prefijos “Dr.”/“Dra.”) o incorporar *fuzzy matching*.

---

## 7) Licencia y contacto

© AG Medical Solutions — Uso interno. Para soporte, abrir un **Issue** en el repo.
