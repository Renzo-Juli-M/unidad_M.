# =============================================================================
# SISTEMA DE SEGMENTACIÓN Y PREDICCIÓN DE CLIENTES — v3.0 PRODUCTION
# Robusto para cualquier CSV: binario, multiclase, miles de columnas/filas,
# tipos arrow, NaN en target, desbalance extremo, sin dependencias opcionales
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings, traceback, io
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, silhouette_score,
    classification_report
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree as sk_tree
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except Exception:
    SMOTE_OK = False

# =============================================================================
# PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Minería de Datos v3.0",
    page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>
.mc  {background:linear-gradient(135deg,#1e3a5f,#2d6a9f);padding:16px;border-radius:12px;
      color:white;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.2);margin:3px}
.mc h2{font-size:1.7rem;margin:0}.mc p{margin:0;font-size:.82rem;opacity:.9}
.mcg {background:linear-gradient(135deg,#1a5c38,#27ae60);padding:16px;border-radius:12px;
      color:white;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.2);margin:3px}
.mcg h2{font-size:1.7rem;margin:0}.mcg p{margin:0;font-size:.82rem;opacity:.9}
.card{
    background:white;
    color:#2c3e50 !important;
    padding:20px;
    border-radius:12px;
    box-shadow:0 2px 10px rgba(0,0,0,.08);
    margin-bottom:16px;
    border-left:5px solid #2d6a9f;
}

.card h3,
.card h4 {
    color:#1e3a5f !important;
    font-weight:700;
}

.card p,
.card li,
.card b,
.card small,
.card ol,
.card ul {
    color:#2c3e50 !important;
}
.ibox{background:linear-gradient(135deg,#e8f4fd,#d1ecf1);padding:13px 17px;border-radius:10px;
      border-left:4px solid #17a2b8;margin:8px 0;font-size:.91rem;color:#2c3e50}
.sbox{background:linear-gradient(135deg,#d4edda,#c3e6cb);padding:13px 17px;border-radius:10px;
      border-left:4px solid #28a745;margin:8px 0}
.wbox{background:#fff3cd;padding:13px 17px;border-radius:10px;
      border-left:4px solid #ffc107;margin:8px 0}
.dbox{background:#f8d7da;padding:13px 17px;border-radius:10px;
      border-left:4px solid #dc3545;margin:8px 0}
.banner{background:linear-gradient(135deg,#1e3a5f,#2d6a9f);color:white;padding:26px;
        border-radius:15px;text-align:center;margin-bottom:20px;
        box-shadow:0 6px 20px rgba(0,0,0,.25)}
.banner h1{font-size:1.75rem;margin-bottom:5px}
.banner p{font-size:.93rem;opacity:.9;margin:0}
hr{border:none;border-top:2px solid #e9ecef;margin:16px 0}
</style>""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE — fuente única de verdad
# =============================================================================
_DEFAULTS = dict(
    df_raw=None, df_proc=None,
    target_col=None, feature_cols=[], num_cols=[], cat_cols=[],
    X_train=None, X_val=None, X_test=None,
    y_train=None, y_val=None, y_test=None,
    scaler=None, le_target=None, class_names=[],
    is_binary=True, n_classes=2,
    results={}, best_k=3,
    df_clustered=None, cluster_col_ready=False,
    rf_importances=None, rf_names=None,
    dt_model=None, dt_names=None,
    models_trained=False, partition_done=False,
    feature_names_model=[],
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

PALETTE = ['#2d6a9f','#e74c3c','#27ae60','#f39c12',
           '#8e44ad','#1abc9c','#e67e22','#2ecc71',
           '#c0392b','#16a085','#d35400','#7f8c8d']

# =============================================================================
# HELPERS — usados en todos los módulos
# =============================================================================
def H(tag, txt): return f'<div class="{tag}">{txt}</div>'
def banner(t, s): st.markdown(f'<div class="banner"><h1>{t}</h1><p>{s}</p></div>', unsafe_allow_html=True)
def mc(v,l,g=False): tag="mcg" if g else "mc"; return f'<div class="{tag}"><h2>{v}</h2><p>{l}</p></div>'

def show_err(msg, exc=None):
    """Muestra error amigable con traza opcional."""
    st.markdown(H("dbox", f"❌ <b>Error:</b> {msg}"), unsafe_allow_html=True)
    if exc:
        with st.expander("Ver detalle técnico"):
            st.code(traceback.format_exc())

def to_float64(df, cols):
    """
    Convierte columnas a float64 de forma robusta.
    Maneja: arrow dtype, strings con comas, infinitos, etc.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        try:
            s = out[c]
            # Si es string numérico con coma decimal
            if s.dtype == object:
                s = s.astype(str).str.replace(',', '.', regex=False)
            s = pd.to_numeric(s, errors='coerce')
            # Reemplazar infinitos
            s = s.replace([np.inf, -np.inf], np.nan)
            out[c] = s.astype('float64')
        except Exception:
            try:
                out[c] = out[c].astype('float64')
            except Exception:
                pass  # columna se queda como está
    return out

def clean_df(df):
    """
    Limpieza profunda del DataFrame al cargar:
    - Nombres de columnas sin espacios extra
    - Tipos robustos
    - Eliminar columnas 100% vacías
    """
    df = df.copy()
    # Limpiar nombres de columnas
    df.columns = [str(c).strip().replace(' ', '_').replace('/', '_')
                  .replace('(', '').replace(')', '') for c in df.columns]
    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')
    # Intentar convertir columnas object que son numéricas
    for c in df.select_dtypes(include='object').columns:
        try:
            converted = pd.to_numeric(df[c].astype(str).str.replace(',','.'), errors='coerce')
            if converted.notna().sum() / max(len(df),1) > 0.80:
                df[c] = converted
        except Exception:
            pass
    return df

def identify_columns(df, target):
    """
    Identifica columnas numéricas y categóricas robustamente.
    Excluye: target, columnas con 1 valor único, columnas 100% nulas.
    """
    feat = [c for c in df.columns if c != target]
    # Excluir columna si tiene 0 varianza útil
    valid = []
    for c in feat:
        if df[c].nunique(dropna=True) < 2:
            continue  # columna constante → sin información
        if df[c].isnull().sum() / max(len(df),1) > 0.95:
            continue  # columna 95%+ nula → sin información
        valid.append(c)

    num_cols = [c for c in valid if df[c].dtype in
                ['int8','int16','int32','int64','float32','float64',
                 'Int8','Int16','Int32','Int64','Float32','Float64']]
    cat_cols = [c for c in valid if c not in num_cols and df[c].dtype == object]
    return num_cols, cat_cols

def safe_avg(is_binary):
    return 'binary' if is_binary else 'weighted'

def calc_metrics(y_true, y_pred, y_prob, is_binary, n_classes):
    """
    Calcula todas las métricas manejando binario, multiclase,
    y casos degenerados (una sola clase en y_test).
    """
    avg = safe_avg(is_binary)
    try: acc  = float(accuracy_score(y_true, y_pred))
    except Exception: acc = 0.0
    try: prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception: prec = 0.0
    try: rec  = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception: rec = 0.0
    try: f1   = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception: f1 = 0.0

    auc = None
    if y_prob is not None:
        try:
            classes_in_test = np.unique(y_true)
            if len(classes_in_test) < 2:
                auc = None  # imposible calcular AUC con 1 clase
            elif is_binary:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted',
                    labels=list(range(n_classes))
                ))
        except Exception:
            auc = None
    return acc, prec, rec, f1, auc

def subsample(X, max_n=50_000):
    """Submuestrea arrays grandes para operaciones lentas."""
    if len(X) <= max_n:
        return X, np.arange(len(X))
    idx = np.random.choice(len(X), max_n, replace=False)
    return X[idx], idx

def cap_cols_for_display(df, max_cols=50):
    """Limita columnas en DataFrames para visualización."""
    if df.shape[1] > max_cols:
        st.caption(f"ℹ️ Mostrando {max_cols} de {df.shape[1]} columnas.")
        return df.iloc[:, :max_cols]
    return df

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎯 Minería de Datos v3.0")
    st.markdown("---")
    menu = st.radio("Módulos", [
        "🏠 Inicio", "📁 Carga de Datos", "🧹 Limpieza de Datos",
        "⚙️ Preprocesamiento", "📊 Partición y Baseline", "🔵 Segmentación",
        "🤖 Clasificación", "📈 Evaluación", "📋 Comparativa",
        "🔮 Predicción Individual", "💼 Informe Gerencial"
    ], label_visibility="collapsed")
    st.markdown("---")
    s = st.session_state
    if s.df_raw is not None:
        st.success(f"✅ {s.df_raw.shape[0]:,} filas · {s.df_raw.shape[1]} cols")
        if s.target_col:
            tipo = "Binario" if s.is_binary else f"Multiclase ({s.n_classes} clases)"
            st.success(f"🎯 `{s.target_col}` · {tipo}")
        else:
            st.warning("⚠️ Sin variable objetivo")
        if s.partition_done: st.success("✂️ Partición OK")
        if s.models_trained:  st.success("🤖 Modelos OK")
    else:
        st.info("📤 Sin dataset")
    st.markdown("---")
    st.caption("Universidad Peruana Unión · 2025")

# =============================================================================
# MÓDULO 0: INICIO
# =============================================================================
if menu == "🏠 Inicio":
    banner("🎯 SISTEMA DE SEGMENTACIÓN Y PREDICCIÓN DE CLIENTES v3.0",
           "Robusto · Binario y Multiclase · Miles de columnas · Cero errores en producción")
    c1,c2,c3,c4 = st.columns(4)
    for col,v,l in zip([c1,c2,c3,c4],
        ["10","5","Binario+\nMulticlase","Arrow\nFix"],
        ["Módulos","Algoritmos ML","Tipos de problema","Pandas moderno"]):
        with col: st.markdown(mc(v,l), unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown("""<div class="card"><h3>🛡️ Robustez v3.0</h3><ul>
        <li>✅ Funciona con <b>cualquier CSV</b> — binario, multiclase, mixto</li>
        <li>✅ <b>Miles de columnas y filas</b> con submuestreo inteligente</li>
        <li>✅ <b>Fix ArrowDtype</b> de pandas moderno en todas las operaciones</li>
        <li>✅ Métricas <b>automáticas</b>: binary / weighted / OVR según el problema</li>
        <li>✅ <b>Columnas constantes</b> eliminadas automáticamente</li>
        <li>✅ <b>NaN en target</b> manejados con advertencia</li>
        <li>✅ <b>SMOTE</b> con validación de mínimo de muestras</li>
        <li>✅ <b>Cross-validation</b> seguro para datasets grandes</li>
        <li>✅ Visualizaciones <b>escalables</b> con cap de puntos/columnas</li>
        <li>✅ Todos los errores muestran <b>mensaje claro</b>, nunca crash</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card"><h3>📋 Flujo de uso</h3><ol>
        <li>📁 Cargar CSV → confirmar objetivo</li>
        <li>⚙️ Preprocesar (automático)</li>
        <li>📊 Particionar 70/15/15</li>
        <li>🔵 Segmentar (opcional)</li>
        <li>🤖 Entrenar modelos</li>
        <li>📈 Evaluar resultados</li>
        <li>📋 Comparar modelos</li>
        <li>🔮 Predecir cliente nuevo</li>
        <li>💼 Informe gerencial</li>
        </ol></div>""", unsafe_allow_html=True)
    st.markdown(H("ibox","📤 <b>Comienza en 📁 Carga de Datos.</b> El sistema se adapta automáticamente "
                  "al tipo de problema y características del dataset."), unsafe_allow_html=True)

# =============================================================================
# MÓDULO 1: CARGA DE DATOS
# =============================================================================
elif menu == "📁 Carga de Datos":
    banner("📁 Carga y Exploración de Datos",
           "Cualquier CSV · Detección automática · Limpieza inicial")

    up = st.file_uploader("Selecciona tu archivo CSV", type=['csv'])
    if up:
        try:
            # Intentar múltiples encodings
            raw = up.read()
            df = None
            detected_sep = ','
            for enc in ['utf-8','latin-1','cp1252','iso-8859-1']:
                for sep in [',', ';', '\t', '|']:
                    try:
                        candidate = pd.read_csv(io.BytesIO(raw), encoding=enc,
                                                sep=sep, low_memory=False)
                        # Descartar si produjo una sola columna (sep incorrecto)
                        if candidate.shape[1] >= 2:
                            df = candidate
                            detected_sep = sep
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is None:
                show_err("No se pudo leer el archivo con ningún separador ni encoding conocido.")
                st.stop()
            sep_names = {',':'coma (,)', ';':'punto y coma (;)', '\t':'tabulación', '|':'pipe (|)'}
            st.caption(f"🔍 Separador detectado automáticamente: **{sep_names.get(detected_sep, detected_sep)}**")

            df = clean_df(df)

            # Reset completo
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v if not isinstance(v, (dict, list)) else type(v)()
            st.session_state.df_raw = df
            st.success(f"✅ **{up.name}** — {df.shape[0]:,} filas · {df.shape[1]} columnas")

        except Exception as e:
            show_err("Error al leer el archivo.", e); st.stop()

    if st.session_state.df_raw is None:
        st.markdown(H("wbox","⚠️ Sube un archivo CSV para comenzar."), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df_raw

    # Métricas
    c1,c2,c3,c4 = st.columns(4)
    for col,v,l in zip([c1,c2,c3,c4],
        [f"{df.shape[0]:,}", df.shape[1],
         len(df.select_dtypes(include='number').columns),
         len(df.select_dtypes(include='object').columns)],
        ["Filas","Columnas","Numéricas","Categóricas"]):
        with col: st.markdown(mc(v,l), unsafe_allow_html=True)

    st.markdown("---")
    # ── Selector de variable objetivo ──
    st.markdown("### 🎯 Variable Objetivo")
    st.markdown(H("ibox","Elige la columna a predecir. Puede ser binaria (0/1, Sí/No) o multiclase."),
                unsafe_allow_html=True)

    cols_list = list(df.columns)
    def_idx = 0
    if st.session_state.target_col and st.session_state.target_col in cols_list:
        def_idx = cols_list.index(st.session_state.target_col)
    else:
        # Intentar detectar automáticamente columnas candidatas
        candidates = [c for c in cols_list if any(k in c.lower() for k in
                      ['target','label','class','respond','objetivo','output','y_','_y'])]
        if candidates:
            def_idx = cols_list.index(candidates[0])

    target_sel = st.selectbox("Columna objetivo:", cols_list, index=def_idx)

    if st.button("✅ Confirmar variable objetivo", type="primary"):
        col_data = df[target_sel].dropna()
        n_unique  = col_data.nunique()
        if n_unique < 2:
            st.error("❌ La columna tiene menos de 2 valores únicos. Elige otra."); st.stop()
        if n_unique > 50:
            st.error(f"❌ La columna tiene {n_unique} valores únicos. ¿Es continua? Elige una categórica."); st.stop()

        is_bin = n_unique == 2
        st.session_state.update(dict(
            target_col=target_sel, is_binary=is_bin, n_classes=n_unique,
            class_names=sorted([str(v) for v in col_data.unique()])
        ))
        st.success(f"✅ **{target_sel}** — {'Binario' if is_bin else f'Multiclase ({n_unique} clases)'} "
                   f"— Clases: {sorted(col_data.unique())}")

        # Nulos en target
        n_nan = df[target_sel].isnull().sum()
        if n_nan > 0:
            st.markdown(H("wbox", f"⚠️ La columna objetivo tiene <b>{n_nan} nulos</b>. "
                          "Se eliminarán esas filas en el preprocesamiento."), unsafe_allow_html=True)

    # Info del target si ya está configurado
    if st.session_state.target_col:
        target = st.session_state.target_col
        vc = df[target].value_counts()
        min_pct = vc.min()/len(df)*100
        c1,c2 = st.columns([2,3])
        with c1:
            if st.session_state.is_binary and min_pct < 20:
                st.markdown(H("dbox",f"⚠️ <b>Desbalance detectado</b><br>Clase minoritaria: <b>{min_pct:.1f}%</b>"),
                            unsafe_allow_html=True)
            elif not st.session_state.is_binary:
                st.markdown(H("ibox",f"📊 <b>Multiclase:</b> {st.session_state.n_classes} clases · "
                              f"Métricas: weighted"), unsafe_allow_html=True)
            else:
                st.markdown(H("sbox",f"✅ Balanceado · Clase minoritaria: {min_pct:.1f}%"),
                            unsafe_allow_html=True)
            st.dataframe(vc.rename_axis('Clase').reset_index(name='N'), use_container_width=True)
        with c2:
            fig = px.pie(df, names=target, title="Distribución de la Variable Objetivo",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs(["👁️ Vista Previa","🔍 Tipos","⚠️ Nulos","📊 Estadísticas"])
    with tab1:
        st.dataframe(cap_cols_for_display(df.head(100)), use_container_width=True)
    with tab2:
        tipos = pd.DataFrame({
            'Variable': df.columns,
            'Dtype': [str(df[c].dtype) for c in df.columns],
            'Únicos': [df[c].nunique() for c in df.columns],
            'Nulos': [int(df[c].isnull().sum()) for c in df.columns],
            '% Nulos': [(df[c].isnull().sum()/len(df)*100).round(1) for c in df.columns],
            'Ejemplo': [str(df[c].dropna().iloc[0]) if df[c].notna().any() else 'N/A'
                        for c in df.columns]
        })
        st.dataframe(tipos, use_container_width=True)
        if df.shape[1] > 50:
            st.caption(f"Dataset amplio: {df.shape[1]} columnas mostradas.")
    with tab3:
        nulos = pd.DataFrame({'Variable':df.columns,
                              'Nulos':df.isnull().sum().values,
                              '% Nulos':(df.isnull().sum().values/len(df)*100).round(2)})
        nulos = nulos.sort_values('Nulos', ascending=False)
        st.dataframe(nulos, use_container_width=True)
        total_n = df.isnull().sum().sum()
        msg = "✅ Sin valores nulos." if total_n==0 else f"⚠️ Total nulos: {total_n:,}"
        st.markdown(H("sbox" if total_n==0 else "wbox", msg), unsafe_allow_html=True)
    with tab4:
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 0:
            st.dataframe(cap_cols_for_display(num_df.describe().T.round(3)), use_container_width=True)
        else:
            st.info("Sin columnas numéricas.")

    # Distribuciones (máx 12 columnas para no saturar)
    st.markdown("### 📊 Distribuciones")
    num_c = [c for c in df.select_dtypes(include='number').columns
             if c != st.session_state.target_col][:12]
    if num_c:
        rows = [num_c[i:i+3] for i in range(0,len(num_c),3)]
        for row in rows:
            cols_ui = st.columns(len(row))
            for col_ui, c in zip(cols_ui, row):
                with col_ui:
                    fig = px.histogram(df, x=c, nbins=40, title=c,
                                       color_discrete_sequence=['#2d6a9f'])
                    fig.update_layout(margin=dict(t=35,b=5), height=230, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        if len([c for c in df.select_dtypes(include='number').columns
                if c != st.session_state.target_col]) > 12:
            st.caption("ℹ️ Mostrando primeras 12 columnas numéricas.")

# =============================================================================
# MÓDULO 1.5: LIMPIEZA DE DATOS
# =============================================================================
elif menu == "🧹 Limpieza de Datos":
    banner("🧹 Limpieza de Datos",
           "Detecta y corrige problemas · Duplicados · Outliers · Columnas problemáticas")

    if st.session_state.df_raw is None:
        st.markdown(H("wbox","⚠️ Primero carga un dataset."), unsafe_allow_html=True); st.stop()
    if not st.session_state.target_col:
        st.markdown(H("wbox","⚠️ Confirma la variable objetivo en Carga de Datos."),
                    unsafe_allow_html=True); st.stop()

    df_original = st.session_state.df_raw.copy()
    target      = st.session_state.target_col

    st.markdown(H("ibox","🧹 Este módulo te muestra todos los problemas del dataset y te deja decidir "
                  "qué hacer con cada uno. Los cambios se aplican al botón <b>'Aplicar limpieza'</b>."),
                unsafe_allow_html=True)

    # ── Diagnóstico completo ──
    st.markdown("### 🔍 Diagnóstico del Dataset")

    num_c_raw = df_original.select_dtypes(include='number').columns.tolist()
    cat_c_raw = df_original.select_dtypes(include='object').columns.tolist()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mc(f"{len(df_original):,}","Filas originales"), unsafe_allow_html=True)
    with c2: st.markdown(mc(int(df_original.duplicated().sum()),"Filas duplicadas",
                             g=df_original.duplicated().sum()==0), unsafe_allow_html=True)
    with c3: st.markdown(mc(int(df_original.isnull().sum().sum()),"Valores nulos totales",
                             g=df_original.isnull().sum().sum()==0), unsafe_allow_html=True)
    with c4:
        cols_constantes = [c for c in df_original.columns if df_original[c].nunique()<=1]
        st.markdown(mc(len(cols_constantes),"Columnas constantes",
                       g=len(cols_constantes)==0), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑️ Duplicados", "❓ Valores Nulos", "📉 Outliers",
        "🗂️ Columnas", "✂️ Aplicar Limpieza"
    ])

    # ── TAB 1: DUPLICADOS ──
    with tab1:
        st.markdown("### 🗑️ Filas Duplicadas")
        n_dup = df_original.duplicated().sum()
        if n_dup == 0:
            st.markdown(H("sbox","✅ No hay filas duplicadas en el dataset."), unsafe_allow_html=True)
        else:
            st.markdown(H("wbox",f"⚠️ Se encontraron <b>{n_dup:,} filas duplicadas</b> "
                          f"({n_dup/len(df_original)*100:.1f}% del total)."), unsafe_allow_html=True)
            st.dataframe(df_original[df_original.duplicated(keep=False)].head(50),
                         use_container_width=True)
            if n_dup > 50:
                st.caption(f"Mostrando 50 de {n_dup:,} duplicados.")

    # ── TAB 2: NULOS ──
    with tab2:
        st.markdown("### ❓ Análisis de Valores Nulos")
        nulos = pd.DataFrame({
            'Columna':  df_original.columns,
            'Nulos':    df_original.isnull().sum().values,
            '% Nulos':  (df_original.isnull().sum().values / len(df_original) * 100).round(2),
            'Tipo':     [str(df_original[c].dtype) for c in df_original.columns],
        }).sort_values('Nulos', ascending=False)

        nulos_reales = nulos[nulos['Nulos'] > 0]

        if len(nulos_reales) == 0:
            st.markdown(H("sbox","✅ Sin valores nulos."), unsafe_allow_html=True)
        else:
            st.dataframe(nulos_reales, use_container_width=True, hide_index=True)

            # Heatmap de nulos
            if df_original.isnull().any().any() and df_original.shape[1] <= 60:
                st.markdown("#### Mapa visual de nulos")
                fig, ax = plt.subplots(figsize=(min(14, df_original.shape[1]*0.5+2), 4))
                sns.heatmap(df_original.isnull().T, cbar=False, ax=ax,
                            cmap=['#d4edda','#f8d7da'], yticklabels=True)
                ax.set_title("Nulos por columna (rojo = nulo)", fontsize=11, fontweight='bold')
                ax.set_xlabel("Filas"); ax.set_ylabel("Columnas")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            # Estrategia de imputación por columna
            st.markdown("#### Estrategia de imputación")
            for _, row in nulos_reales.iterrows():
                col_n  = row['Columna']
                pct    = row['% Nulos']
                tipo   = row['Tipo']
                if pct >= 60:
                    recom = "🔴 Eliminar columna (>60% nulos)"
                elif pct >= 20:
                    recom = "🟡 Imputar con mediana/moda o eliminar"
                else:
                    recom = "🟢 Imputar con mediana (numéricas) o moda (categóricas)"
                st.markdown(f"- **`{col_n}`** ({pct}% nulos, {tipo}): {recom}")

    # ── TAB 3: OUTLIERS ──
    with tab3:
        st.markdown("### 📉 Detección de Outliers (Método IQR)")
        st.markdown(H("ibox","Un <b>outlier</b> es un valor extremo que se aleja significativamente "
                      "del resto. Se detectan con el método IQR: valores fuera de "
                      "[Q1 - 1.5×IQR, Q3 + 1.5×IQR]."), unsafe_allow_html=True)

        feat_num = [c for c in num_c_raw if c != target]

        if not feat_num:
            st.info("Sin columnas numéricas para analizar.")
        else:
            resumen_out = []
            for c in feat_num:
                col_s = pd.to_numeric(df_original[c], errors='coerce').dropna()
                if len(col_s) < 4: continue
                Q1, Q3 = col_s.quantile(0.25), col_s.quantile(0.75)
                IQR    = Q3 - Q1
                if IQR == 0: continue
                lim_inf, lim_sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                n_out  = ((col_s < lim_inf) | (col_s > lim_sup)).sum()
                if n_out > 0:
                    resumen_out.append({
                        'Columna':    c,
                        'Outliers':   int(n_out),
                        '% Outliers': round(n_out/len(col_s)*100, 2),
                        'Límite inf': round(float(lim_inf), 3),
                        'Límite sup': round(float(lim_sup), 3),
                        'Min real':   round(float(col_s.min()), 3),
                        'Max real':   round(float(col_s.max()), 3),
                    })

            if not resumen_out:
                st.markdown(H("sbox","✅ No se detectaron outliers significativos."), unsafe_allow_html=True)
            else:
                df_out = pd.DataFrame(resumen_out).sort_values('% Outliers', ascending=False)
                st.dataframe(df_out, use_container_width=True, hide_index=True)

                # Boxplots de las columnas con más outliers
                top_out = df_out.head(6)['Columna'].tolist()
                st.markdown("#### Boxplots de columnas con más outliers")
                cols_box = st.columns(min(3, len(top_out)))
                for i, c in enumerate(top_out):
                    with cols_box[i % 3]:
                        col_clean = pd.to_numeric(df_original[c], errors='coerce').dropna()
                        fig = px.box(y=col_clean, title=c,
                                     color_discrete_sequence=['#2d6a9f'])
                        fig.update_layout(height=260, margin=dict(t=35,b=5))
                        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: COLUMNAS ──
    with tab4:
        st.markdown("### 🗂️ Análisis de Columnas Problemáticas")

        problemas_col = []
        for c in df_original.columns:
            if c == target: continue
            n_uniq = df_original[c].nunique()
            pct_nul = df_original[c].isnull().sum() / len(df_original) * 100
            problema = None
            if n_uniq <= 1:
                problema = "🔴 Constante (sin varianza) — no aporta información"
            elif pct_nul >= 60:
                problema = f"🔴 {pct_nul:.0f}% nulos — muy poco dato útil"
            elif n_uniq == len(df_original) and df_original[c].dtype == object:
                problema = "🟡 ID o texto único — probablemente no es útil como feature"
            elif df_original[c].dtype == object and n_uniq > 50:
                problema = f"🟡 Categórica con {n_uniq} valores únicos — alta cardinalidad"
            if problema:
                problemas_col.append({'Columna': c, 'Problema': problema,
                                       'Únicos': n_uniq, '% Nulos': round(pct_nul,1)})

        if not problemas_col:
            st.markdown(H("sbox","✅ No se detectaron columnas problemáticas."), unsafe_allow_html=True)
        else:
            st.dataframe(pd.DataFrame(problemas_col), use_container_width=True, hide_index=True)

        # Resumen de tipos
        st.markdown("#### Tipos de columnas")
        tipos_resumen = df_original.dtypes.value_counts().reset_index()
        tipos_resumen.columns = ['Tipo','Cantidad']
        tipos_resumen['Tipo'] = tipos_resumen['Tipo'].astype(str)
        fig = px.bar(tipos_resumen, x='Tipo', y='Cantidad', title='Tipos de Datos',
                     color='Tipo', color_discrete_sequence=PALETTE, text='Cantidad')
        fig.update_traces(textposition='auto')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=280)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: APLICAR LIMPIEZA ──
    with tab5:
        st.markdown("### ✂️ Configurar y Aplicar Limpieza")
        st.markdown(H("ibox","Elige qué operaciones aplicar. El dataset limpio reemplazará al original "
                      "y podrás continuar con Preprocesamiento."), unsafe_allow_html=True)

        st.markdown("#### 1️⃣ Duplicados")
        elim_dup = st.checkbox("Eliminar filas duplicadas",
                               value=df_original.duplicated().sum() > 0,
                               help=f"Hay {df_original.duplicated().sum():,} duplicados.")

        st.markdown("#### 2️⃣ Valores Nulos")
        estrategia_nulos = st.radio(
            "¿Cómo tratar los nulos en columnas numéricas?",
            ["Imputar con mediana (recomendado)",
             "Imputar con media",
             "Eliminar filas con nulos"],
            index=0
        )
        estrategia_cat = st.radio(
            "¿Cómo tratar los nulos en columnas categóricas?",
            ["Imputar con moda (recomendado)",
             "Rellenar con 'Desconocido'",
             "Eliminar filas con nulos"],
            index=0
        )

        st.markdown("#### 3️⃣ Outliers")
        tratar_outliers = st.checkbox(
            "Aplicar capping de outliers (reemplazar extremos con límites IQR)",
            value=False,
            help="Reemplaza valores fuera de [Q1-1.5×IQR, Q3+1.5×IQR] con ese límite."
        )

        st.markdown("#### 4️⃣ Columnas a eliminar")
        feat_cols_all = [c for c in df_original.columns if c != target]
        cols_sugeridas_eliminar = []
        for c in feat_cols_all:
            pct_nul = df_original[c].isnull().sum() / len(df_original) * 100
            if df_original[c].nunique() <= 1 or pct_nul >= 60:
                cols_sugeridas_eliminar.append(c)

        cols_a_eliminar = st.multiselect(
            "Columnas a eliminar del dataset:",
            options=feat_cols_all,
            default=cols_sugeridas_eliminar,
            help="Se sugieren automáticamente columnas constantes o con >60% nulos."
        )

        st.markdown("---")

        if st.button("🧹 Aplicar Limpieza", type="primary", use_container_width=True):
            try:
                df_clean = df_original.copy()
                log = []

                # 1. Eliminar columnas
                if cols_a_eliminar:
                    df_clean = df_clean.drop(columns=cols_a_eliminar, errors='ignore')
                    log.append(f"🗑️ Eliminadas {len(cols_a_eliminar)} columnas: {cols_a_eliminar}")

                # 2. Duplicados
                if elim_dup:
                    antes_d = len(df_clean)
                    df_clean = df_clean.drop_duplicates()
                    elim_d = antes_d - len(df_clean)
                    log.append(f"🗑️ Eliminadas {elim_d:,} filas duplicadas.")

                # 3. Filas con target nulo
                antes_t = len(df_clean)
                df_clean = df_clean.dropna(subset=[target])
                elim_t = antes_t - len(df_clean)
                if elim_t > 0:
                    log.append(f"🗑️ Eliminadas {elim_t:,} filas con target nulo.")

                # 4. Nulos numéricos
                cols_num_c = [c for c in df_clean.select_dtypes(include='number').columns
                              if c != target]
                cols_cat_c = [c for c in df_clean.select_dtypes(include='object').columns
                              if c != target]

                if estrategia_nulos == "Imputar con mediana (recomendado)":
                    for c in cols_num_c:
                        med = df_clean[c].median()
                        df_clean[c] = df_clean[c].fillna(med if pd.notna(med) else 0)
                    log.append(f"✅ Nulos numéricos imputados con mediana.")
                elif estrategia_nulos == "Imputar con media":
                    for c in cols_num_c:
                        mn = df_clean[c].mean()
                        df_clean[c] = df_clean[c].fillna(mn if pd.notna(mn) else 0)
                    log.append(f"✅ Nulos numéricos imputados con media.")
                else:
                    antes_fn = len(df_clean)
                    df_clean = df_clean.dropna(subset=cols_num_c)
                    log.append(f"🗑️ Eliminadas {antes_fn - len(df_clean):,} filas con nulos numéricos.")

                # 5. Nulos categóricos
                if estrategia_cat == "Imputar con moda (recomendado)":
                    for c in cols_cat_c:
                        moda = df_clean[c].mode()
                        df_clean[c] = df_clean[c].fillna(moda.iloc[0] if len(moda)>0 else 'Desconocido')
                    log.append(f"✅ Nulos categóricos imputados con moda.")
                elif estrategia_cat == "Rellenar con 'Desconocido'":
                    for c in cols_cat_c:
                        df_clean[c] = df_clean[c].fillna('Desconocido')
                    log.append(f"✅ Nulos categóricos rellenados con 'Desconocido'.")
                else:
                    antes_fc = len(df_clean)
                    df_clean = df_clean.dropna(subset=cols_cat_c)
                    log.append(f"🗑️ Eliminadas {antes_fc - len(df_clean):,} filas con nulos categóricos.")

                # 6. Capping de outliers
                if tratar_outliers:
                    n_caps = 0
                    for c in cols_num_c:
                        col_s = pd.to_numeric(df_clean[c], errors='coerce')
                        Q1, Q3 = col_s.quantile(0.25), col_s.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR == 0: continue
                        lim_inf, lim_sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                        antes_cap = ((col_s < lim_inf) | (col_s > lim_sup)).sum()
                        df_clean[c] = col_s.clip(lower=lim_inf, upper=lim_sup)
                        n_caps += antes_cap
                    log.append(f"✅ Capping aplicado: {n_caps:,} valores recortados a límites IQR.")

                # Guardar
                st.session_state.df_raw = df_clean

                # Mostrar resumen
                st.markdown("### ✅ Limpieza Completada")
                c1,c2,c3 = st.columns(3)
                c1.metric("Filas antes", f"{len(df_original):,}")
                c2.metric("Filas después", f"{len(df_clean):,}",
                          delta=f"{len(df_clean)-len(df_original):,}")
                c3.metric("Nulos restantes", int(df_clean.isnull().sum().sum()))

                st.markdown("#### 📋 Log de operaciones")
                for entry in log:
                    st.markdown(H("sbox", entry), unsafe_allow_html=True)

                # Descarga del CSV limpio
                csv_limpio = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar dataset limpio",
                                   data=csv_limpio, file_name="dataset_limpio.csv",
                                   mime="text/csv")

                st.markdown(H("sbox","✅ Dataset limpio guardado. Continúa con <b>⚙️ Preprocesamiento</b>."),
                            unsafe_allow_html=True)

            except Exception as e:
                show_err("Error durante la limpieza.", e)

# =============================================================================
# MÓDULO 2: PREPROCESAMIENTO
# =============================================================================
elif menu == "⚙️ Preprocesamiento":
    banner("⚙️ Preprocesamiento", "Robusto · Automático · Sin Data Leakage")

    if st.session_state.df_raw is None:
        st.markdown(H("wbox","⚠️ Primero carga un dataset."), unsafe_allow_html=True); st.stop()
    if not st.session_state.target_col:
        st.markdown(H("wbox","⚠️ Confirma la variable objetivo en Carga de Datos."),
                    unsafe_allow_html=True); st.stop()

    st.markdown(H("ibox","🔒 <b>Sin Data Leakage:</b> El StandardScaler se ajusta <b>solo con datos de "
                  "entrenamiento</b> en el módulo de Partición, nunca antes."), unsafe_allow_html=True)

    df = st.session_state.df_raw.copy()
    target = st.session_state.target_col

    try:
        with st.expander("1️⃣ Eliminar filas con target nulo", expanded=True):
            antes = len(df)
            df = df.dropna(subset=[target])
            despues = len(df)
            eliminadas = antes - despues
            msg = f"✅ Sin nulos en target." if eliminadas == 0 else \
                  f"⚠️ Se eliminaron <b>{eliminadas}</b> filas con target nulo."
            st.markdown(H("sbox" if eliminadas==0 else "wbox", msg), unsafe_allow_html=True)

        with st.expander("2️⃣ Identificar variables", expanded=True):
            num_cols, cat_cols = identify_columns(df, target)
            total_feat = len(num_cols) + len(cat_cols)
            c1,c2,c3 = st.columns(3)
            c1.metric("Numéricas", len(num_cols))
            c2.metric("Categóricas", len(cat_cols))
            c3.metric("Total features", total_feat)
            if total_feat == 0:
                show_err("No se encontraron features válidas. Revisa tu dataset."); st.stop()
            if len(num_cols) > 0:
                with st.expander(f"Ver {len(num_cols)} columnas numéricas"):
                    st.write(num_cols)
            if len(cat_cols) > 0:
                with st.expander(f"Ver {len(cat_cols)} columnas categóricas"):
                    st.write(cat_cols)

        with st.expander("3️⃣ Convertir tipos y reparar datos", expanded=True):
            # Forzar float64 en numéricas (fix ArrowDtype, strings numéricos, etc.)
            df = to_float64(df, num_cols)
            # Imputar nulos numéricos con mediana
            for c in num_cols:
                if df[c].isnull().any():
                    med = df[c].median()
                    df[c] = df[c].fillna(med if pd.notna(med) else 0.0)
            # Imputar nulos categóricos con moda
            for c in cat_cols:
                if df[c].isnull().any():
                    moda = df[c].mode()
                    df[c] = df[c].fillna(moda.iloc[0] if len(moda)>0 else 'desconocido')
            nulos_rest = df[num_cols + cat_cols].isnull().sum().sum()
            st.markdown(H("sbox", f"✅ Tipos corregidos. Nulos restantes en features: {nulos_rest}"),
                        unsafe_allow_html=True)

        with st.expander("4️⃣ Codificar categóricas", expanded=True):
            le_map = {}
            for c in cat_cols:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                le_map[c] = le
                df[c] = df[c].astype('float64')
            if cat_cols:
                st.markdown(H("sbox",f"✅ LabelEncoder aplicado a {len(cat_cols)} columnas."),
                            unsafe_allow_html=True)
            else:
                st.info("ℹ️ Sin categóricas.")

        with st.expander("5️⃣ Correlación (top columnas)", expanded=False):
            all_feat = num_cols + cat_cols
            # Limitar a 30 columnas para heatmap legible
            feat_for_corr = all_feat[:30]
            corr_df = df[feat_for_corr].copy()
            if corr_df.shape[1] >= 2:
                corr = corr_df.corr()
                sz = min(14, max(6, corr_df.shape[1]))
                fig, ax = plt.subplots(figsize=(sz, sz*0.75))
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, annot=corr.shape[1]<=15, fmt='.1f',
                            cmap='RdBu_r', center=0, mask=mask, ax=ax,
                            linewidths=0.3, cbar_kws={"shrink":.6})
                ax.set_title("Correlación de Features" +
                             (" (primeras 30)" if len(all_feat)>30 else ""),
                             fontsize=12, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # Actualizar num_cols post-codificación
        all_feat_final = [c for c in (num_cols + cat_cols) if c in df.columns]
        st.session_state.num_cols     = all_feat_final
        st.session_state.cat_cols     = []
        st.session_state.feature_cols = all_feat_final
        st.session_state.df_proc      = df

        c1,c2,c3 = st.columns(3)
        c1.metric("Filas finales", f"{len(df):,}")
        c2.metric("Features finales", len(all_feat_final))
        c3.metric("Nulos totales", int(df.isnull().sum().sum()))
        st.markdown(H("sbox","✅ <b>Preprocesamiento completado.</b> Continúa con Partición y Baseline."),
                    unsafe_allow_html=True)

    except Exception as e:
        show_err("Error en preprocesamiento.", e)

# =============================================================================
# MÓDULO 3: PARTICIÓN Y BASELINE
# =============================================================================
elif menu == "📊 Partición y Baseline":
    banner("📊 Partición y Baseline", "70/15/15 · SMOTE opcional · Baseline adaptativo")

    if st.session_state.df_proc is None:
        st.markdown(H("wbox","⚠️ Ejecuta Preprocesamiento primero."), unsafe_allow_html=True); st.stop()

    df_proc  = st.session_state.df_proc
    target   = st.session_state.target_col
    num_cols = st.session_state.num_cols
    is_bin   = st.session_state.is_binary
    n_cls    = st.session_state.n_classes

    if not num_cols:
        show_err("Sin features numéricas disponibles."); st.stop()

    st.markdown(H("ibox","📌 <b>Sin Data Leakage:</b> StandardScaler ajustado <b>solo con el 70% "
                  "de entrenamiento</b>. Validación y prueba se transforman con ese mismo scaler."),
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        smote_ok_cond = SMOTE_OK and is_bin
        usar_smote = st.checkbox("🔁 SMOTE (balancear clases)",
                                 disabled=not smote_ok_cond,
                                 help="Solo para clasificación binaria. "
                                      "Requiere imbalanced-learn instalado.")
        if not SMOTE_OK: st.caption("pip install imbalanced-learn")
        if not is_bin:   st.caption("SMOTE solo disponible para binario.")

        usar_cluster = st.checkbox("🔵 Incluir clúster K-Means como feature",
                                   disabled=st.session_state.df_clustered is None)
        if st.session_state.df_clustered is None:
            st.caption("Ejecuta Segmentación primero para activar.")
    with c2:
        cv_folds = st.slider("Folds para Cross-Validation", 3, 10, 5)
        test_size = st.slider("Tamaño de Test+Val (%)", 20, 40, 30) / 100

    if st.button("✂️ Ejecutar Partición", type="primary", use_container_width=True):
        try:
            # Construir X
            if usar_cluster and st.session_state.df_clustered is not None:
                df_use = st.session_state.df_clustered.copy()
                df_use = to_float64(df_use, num_cols + ['cluster_kmeans'])
                cols_X = [c for c in num_cols + ['cluster_kmeans'] if c in df_use.columns]
            else:
                df_use = df_proc.copy()
                cols_X = num_cols

            df_use = to_float64(df_use, cols_X)
            X = df_use[cols_X].copy().astype('float64')
            y_raw = df_use[target].copy()

            # Codificar target
            le_y = LabelEncoder()
            y = pd.Series(le_y.fit_transform(y_raw.astype(str)), index=y_raw.index, dtype='int64')
            st.session_state.le_target  = le_y
            st.session_state.class_names = [str(c) for c in le_y.classes_]

            # Verificar que hay suficientes muestras por clase
            min_class_count = y.value_counts().min()
            if min_class_count < 3:
                show_err(f"La clase con menos muestras tiene solo {min_class_count} registros. "
                         "Necesitas al menos 3 por clase para hacer la partición estratificada.")
                st.stop()

            # Split 70 / 15 / 15 (o según slider)
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

            # Scaler — ajustado SOLO con train
            scaler = StandardScaler()
            X_tr_sc  = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            X_te_sc  = scaler.transform(X_te)

            # SMOTE (solo binario, solo si hay suficientes muestras)
            if usar_smote and SMOTE_OK and is_bin:
                min_after = min(pd.Series(y_tr).value_counts())
                if min_after >= 6:
                    sm = SMOTE(random_state=42, k_neighbors=min(5, min_after-1))
                    X_tr_sc, y_tr = sm.fit_resample(X_tr_sc, y_tr)
                    vc_after = pd.Series(y_tr).value_counts().to_dict()
                    st.markdown(H("sbox",f"✅ SMOTE aplicado. Train balanceado: {vc_after}"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(H("wbox","⚠️ SMOTE omitido: clase minoritaria con menos de 6 muestras."),
                                unsafe_allow_html=True)

            # Guardar
            st.session_state.update(dict(
                X_train=X_tr_sc, X_val=X_val_sc, X_test=X_te_sc,
                y_train=y_tr, y_val=y_val, y_test=y_te,
                scaler=scaler, feature_names_model=list(cols_X),
                partition_done=True, models_trained=False, results={}
            ))

            # Mostrar distribución
            c1,c2,c3,c4 = st.columns(4)
            for col,(l,n) in zip([c1,c2,c3,c4],[
                ("📦 Total",len(X)),("🎓 Train",len(X_tr)),
                ("🔍 Val",len(X_val)),("🧪 Test",len(X_te))]):
                with col: st.markdown(mc(f"{n:,}",l), unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=['Entrenamiento','Validación','Prueba'],
                y=[len(X_tr),len(X_val),len(X_te)],
                marker_color=['#2d6a9f','#27ae60','#e74c3c'],
                text=[f'{v:,}' for v in [len(X_tr),len(X_val),len(X_te)]],
                textposition='auto'))
            fig.update_layout(title="Distribución de Conjuntos",
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            # ── BASELINE ──
            st.markdown("---")
            st.markdown("### 🎲 Modelo Baseline")
            st.markdown(H("ibox","📌 El baseline predice la clase más frecuente. "
                          "Un modelo real debe superarlo claramente."), unsafe_allow_html=True)

            baselines = {
                'Baseline (Dummy)':   DummyClassifier(strategy='most_frequent', random_state=42),
                'Baseline (Log.Reg)': LogisticRegression(max_iter=2000, random_state=42,
                                                          class_weight='balanced', solver='saga')
            }
            b_cols = st.columns(len(baselines))
            for (nom,mod), col in zip(baselines.items(), b_cols):
                try:
                    mod.fit(X_tr_sc, y_tr)
                    y_pred_b = mod.predict(X_te_sc)
                    y_prob_b = mod.predict_proba(X_te_sc)
                    acc_b, prec_b, rec_b, f1_b, auc_b = calc_metrics(y_te, y_pred_b, y_prob_b, is_bin, n_cls)
                    st.session_state.results[nom] = dict(
                        Accuracy=acc_b, Precision=prec_b, Recall=rec_b,
                        F1=f1_b, AUC=auc_b, CV_mean=None, CV_std=None,
                        Interpretación='Modelo de referencia', y_pred=y_pred_b, y_prob=y_prob_b
                    )
                    with col:
                        st.markdown(f"**{nom}**")
                        mc1,mc2,mc3 = st.columns(3)
                        mc1.metric("Acc",  f"{acc_b:.3f}")
                        mc2.metric("F1",   f"{f1_b:.3f}")
                        mc3.metric("AUC",  f"{auc_b:.3f}" if auc_b else "N/A")
                except Exception as eb:
                    with col:
                        st.markdown(H("wbox",f"⚠️ {nom} falló: {eb}"), unsafe_allow_html=True)

            st.markdown(H("sbox","✅ Partición y baseline completados."), unsafe_allow_html=True)

        except Exception as e:
            show_err("Error en la partición.", e)

    elif st.session_state.partition_done:
        st.markdown(H("sbox","✅ Partición ya ejecutada. Continúa con los demás módulos."),
                    unsafe_allow_html=True)

# =============================================================================
# MÓDULO 4: SEGMENTACIÓN
# =============================================================================
elif menu == "🔵 Segmentación":
    banner("🔵 Segmentación de Clientes",
           "K-Means · Clustering Jerárquico · Silhouette · Perfiles automáticos")

    if st.session_state.df_proc is None:
        st.markdown(H("wbox","⚠️ Ejecuta Preprocesamiento primero."), unsafe_allow_html=True); st.stop()

    df      = st.session_state.df_proc.copy()
    target  = st.session_state.target_col
    num_c   = st.session_state.num_cols

    feat_c  = [c for c in num_c if c in df.columns]
    if not feat_c:
        show_err("Sin features numéricas para clustering."); st.stop()

    df = to_float64(df, feat_c)
    df[feat_c] = df[feat_c].fillna(df[feat_c].median())

    X_raw = df[feat_c].values.astype('float64')

    # Submuestrear para clustering si dataset es muy grande
    MAX_CLUSTER = 20_000
    if len(X_raw) > MAX_CLUSTER:
        idx_sub = np.random.choice(len(X_raw), MAX_CLUSTER, replace=False)
        X_sub   = X_raw[idx_sub]
        st.markdown(H("wbox",f"⚠️ Dataset grande: clustering sobre muestra de {MAX_CLUSTER:,} filas."),
                    unsafe_allow_html=True)
    else:
        X_sub = X_raw
        idx_sub = np.arange(len(X_raw))

    sc_c = StandardScaler()
    X_sc = sc_c.fit_transform(X_sub)

    # PCA para visualización (max 2 componentes, mínimo 1)
    n_pca = min(2, X_sc.shape[1])
    pca   = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    tab1,tab2,tab3 = st.tabs(["🔵 K-Means","🌳 Jerárquico","👥 Perfiles"])

    # ── K-MEANS ──
    with tab1:
        st.markdown("### 🔵 K-Means")
        st.markdown(H("ibox","📌 <b>Silhouette:</b> de -1 a 1. Cercano a 1 = clústeres bien separados. "
                      "El sistema elige el k con mayor score automáticamente."), unsafe_allow_html=True)

        k_range = range(2, min(9, len(X_sc)//10+2))  # evitar k inválido
        sil_scores = []
        prog = st.progress(0)
        for i,k in enumerate(k_range):
            try:
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                lbl_tmp = km_tmp.fit_predict(X_sc)
                if len(np.unique(lbl_tmp)) < 2:
                    sil_scores.append(-1)
                else:
                    sil_scores.append(float(silhouette_score(X_sc, lbl_tmp, sample_size=min(5000,len(X_sc)))))
            except Exception:
                sil_scores.append(-1)
            prog.progress((i+1)/len(k_range))
        prog.empty()

        best_k = list(k_range)[int(np.argmax(sil_scores))]
        st.session_state.best_k = best_k

        c1,c2 = st.columns([3,2])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode='lines+markers',
                                     line=dict(width=3,color='#2d6a9f'),
                                     marker=dict(size=10,color='#2d6a9f')))
            fig.add_vline(x=best_k, line_dash='dash', line_color='red',
                          annotation_text=f'Mejor k={best_k}')
            fig.update_layout(title='Silhouette Score por k', xaxis_title='k', yaxis_title='Score',
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sil_df = pd.DataFrame({'k':list(k_range), 'Silhouette':[round(s,4) for s in sil_scores]})
            sil_df[''] = sil_df['k'].apply(lambda x: '⭐ Mejor' if x==best_k else '')
            st.dataframe(sil_df, use_container_width=True, hide_index=True)

        n_k = st.slider("Clústeres K-Means", 2, max(2,len(list(k_range))+1), best_k)

        km_final = KMeans(n_clusters=n_k, random_state=42, n_init=10, max_iter=300)
        labels_km = km_final.fit_predict(X_sc)

        # Asignar clústeres al DF completo usando los índices de submuestreo
        df['cluster_kmeans'] = -1
        df.iloc[idx_sub, df.columns.get_loc('cluster_kmeans')] = labels_km
        # Para filas no muestreadas, asignar con predict
        if len(X_raw) > MAX_CLUSTER:
            idx_rest = np.setdiff1d(np.arange(len(X_raw)), idx_sub)
            X_rest_sc = sc_c.transform(X_raw[idx_rest].astype('float64'))
            df.iloc[idx_rest, df.columns.get_loc('cluster_kmeans')] = km_final.predict(X_rest_sc)
        df['cluster_kmeans'] = df['cluster_kmeans'].astype(int)

        # Scatter
        df_pca_plot = pd.DataFrame()
        if n_pca >= 2:
            df_pca_plot['PC1'] = X_pca[:,0]
            df_pca_plot['PC2'] = X_pca[:,1]
        else:
            df_pca_plot['PC1'] = X_pca[:,0]
            df_pca_plot['PC2'] = np.zeros(len(X_pca))
        df_pca_plot['Clúster'] = labels_km.astype(str)
        var_exp = pca.explained_variance_ratio_ * 100
        titulo_pca = (f'Clústeres K-Means (k={n_k}) — '
                      f'PC1={var_exp[0]:.1f}%{"  ·  PC2="+str(round(var_exp[1],1))+"%" if n_pca>=2 else ""}')
        # Submuestrear puntos para scatter si son muchos
        if len(df_pca_plot) > 10000:
            df_pca_plot = df_pca_plot.sample(10000, random_state=42)
        fig = px.scatter(df_pca_plot, x='PC1', y='PC2', color='Clúster',
                         title=titulo_pca, color_discrete_sequence=PALETTE, opacity=0.6)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.df_clustered = df
        st.markdown(H("sbox",f"✅ Clústeres asignados a {len(df):,} registros (k={n_k})."),
                    unsafe_allow_html=True)

    # ── JERÁRQUICO ──
    with tab2:
        st.markdown("### 🌳 Clustering Jerárquico")
        n_h = st.slider("Clústeres jerárquicos", 2, 8, min(best_k,4))

        # Dendrograma sobre muestra pequeña siempre
        dendo_n = min(500, len(X_sc))
        idx_d   = np.random.choice(len(X_sc), dendo_n, replace=False)
        try:
            Z = linkage(X_sc[idx_d], method='ward')
            fig, ax = plt.subplots(figsize=(13,5))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=40,
                       leaf_rotation=90, leaf_font_size=8, show_contracted=True)
            if n_h > 1 and len(Z) >= n_h:
                corte = Z[-(n_h-1),2]
                ax.axhline(y=corte, color='red', linestyle='--', linewidth=1.5,
                           label=f'Corte k={n_h}')
                ax.legend()
            ax.set_title(f'Dendrograma (muestra {dendo_n:,} de {len(X_sc):,})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Muestra'); ax.set_ylabel('Distancia Ward')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception as ed:
            st.warning(f"⚠️ No se pudo generar el dendrograma: {ed}")

        try:
            agg = AgglomerativeClustering(n_clusters=n_h, linkage='ward')
            hier_labels = agg.fit_predict(X_sc)
            dist = pd.Series(hier_labels).value_counts().sort_index().reset_index()
            dist.columns = ['Clúster','N']
            dist['Clúster'] = dist['Clúster'].astype(str)
            fig = px.bar(dist, x='Clúster', y='N', color='Clúster',
                         title='Clientes por Clúster Jerárquico',
                         color_discrete_sequence=PALETTE, text='N')
            fig.update_traces(textposition='auto')
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as eh:
            st.warning(f"⚠️ Clustering jerárquico falló: {eh}")

    # ── PERFILES ──
    with tab3:
        st.markdown("### 👥 Perfiles Automáticos")
        df_cp = st.session_state.df_clustered
        if df_cp is None or 'cluster_kmeans' not in df_cp.columns:
            st.warning("Ejecuta K-Means primero."); st.stop()

        cols_p = [c for c in feat_c if c in df_cp.columns]
        df_cp  = to_float64(df_cp, cols_p)

        try:
            # Usar máx 10 columnas para el resumen
            cols_resumen = cols_p[:10]
            summary = df_cp.groupby('cluster_kmeans')[cols_resumen].mean().round(2)
            summary['N_clientes'] = df_cp.groupby('cluster_kmeans').size()
            if target in df_cp.columns:
                target_num = pd.to_numeric(df_cp[target], errors='coerce')
                if target_num.notna().sum() > 0:
                    summary['Tasa_resp(%)'] = (
                        df_cp.groupby('cluster_kmeans').apply(
                            lambda g: pd.to_numeric(g[target], errors='coerce').mean()*100
                        ).round(1)
                    )

            scores = summary[cols_resumen].mean(axis=1)
            rank   = scores.rank(method='first')
            n_cl   = len(rank)
            pmap   = {}
            for idx in rank.index:
                r = rank[idx]
                if   r >= n_cl*0.75: pmap[idx] = ("🌟 Premium",   "Beneficios exclusivos y atención VIP.")
                elif r >= n_cl*0.50: pmap[idx] = ("🔄 Frecuente", "Programas de fidelización.")
                elif r >= n_cl*0.25: pmap[idx] = ("💰 Económico", "Descuentos y promociones.")
                else:                pmap[idx] = ("💤 Inactivo",  "Campañas de reactivación.")

            st.dataframe(summary.reset_index(), use_container_width=True)

            # Cards
            card_cols = st.columns(min(n_cl,4))
            for i,(idx,row) in enumerate(summary.iterrows()):
                p,r = pmap[idx]
                with card_cols[i%len(card_cols)]:
                    tasa = f" · Tasa: {row.get('Tasa_resp(%)','-')}%" if 'Tasa_resp(%)' in row.index else ""
                    st.markdown(f"""<div class="card" style="min-height:140px">
                        <h4>Clúster {idx}</h4><b>{p}</b><br>
                        👥 {int(row['N_clientes']):,} clientes{tasa}<br>
                        <small>💡 {r}</small></div>""", unsafe_allow_html=True)

            # Heatmap (máx 10 cols)
            fig = px.imshow(summary[cols_resumen].T, title="Heatmap Clústeres",
                            color_continuous_scale='RdBu_r', text_auto='.1f', aspect='auto')
            fig.update_layout(height=max(300, len(cols_resumen)*28))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as ep:
            show_err("Error generando perfiles.", ep)

        csv_cl = df_cp.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar datos con clústeres",
                           data=csv_cl, file_name="clientes_segmentados.csv", mime="text/csv")

# =============================================================================
# MÓDULO 5: CLASIFICACIÓN
# =============================================================================
elif menu == "🤖 Clasificación":
    banner("🤖 Clasificación", "5 modelos · Cross-Validation · Árbol visual · Importancia de variables")

    if not st.session_state.partition_done:
        st.markdown(H("wbox","⚠️ Ejecuta Partición y Baseline primero."), unsafe_allow_html=True); st.stop()

    X_tr    = st.session_state.X_train
    X_val   = st.session_state.X_val
    X_te    = st.session_state.X_test
    y_tr    = st.session_state.y_train
    y_val   = st.session_state.y_val
    y_te    = st.session_state.y_test
    fnams   = st.session_state.feature_names_model
    is_bin  = st.session_state.is_binary
    n_cls   = st.session_state.n_classes
    avg     = safe_avg(is_bin)

    tipo_txt = "Binario" if is_bin else f"Multiclase ({n_cls} clases)"
    st.markdown(H("ibox",f"📌 <b>Problema: {tipo_txt}</b> · "
                  f"Métricas: {'binary' if is_bin else 'weighted'} · "
                  f"AUC: {'binario' if is_bin else 'OVR weighted'}"), unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        max_depth = st.slider("Profundidad máx. Árbol", 2, 20, 5)
        n_est     = st.slider("N° árboles (RF/GBM)", 50, 500, 100, 50)
    with c2:
        usar_cv   = st.checkbox("✅ Cross-Validation", value=True)
        cv_folds  = st.slider("Folds CV", 3, 10, 5, disabled=not usar_cv)
        cw_opt    = st.selectbox("Peso clases", ['balanced','none'])
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Entrenar todos los modelos", type="primary", use_container_width=True)

    if run_btn:
        try:
            cw = 'balanced' if cw_opt=='balanced' else None

            # GradientBoosting no acepta class_weight → siempre None
            modelos = {
                '🌲 Árbol de Decisión':  DecisionTreeClassifier(max_depth=max_depth,
                                           random_state=42, class_weight=cw),
                '🌲🌲 Random Forest':    RandomForestClassifier(n_estimators=n_est,
                                           random_state=42, class_weight=cw, n_jobs=-1),
                '⚡ Gradient Boosting':  GradientBoostingClassifier(n_estimators=n_est,
                                           random_state=42),
                '📉 Reg. Logística':     LogisticRegression(max_iter=2000, random_state=42,
                                           class_weight=cw, solver='saga'),
            }
            if XGBOOST_OK:
                modelos['🚀 XGBoost'] = XGBClassifier(
                    n_estimators=n_est, random_state=42, verbosity=0,
                    eval_metric='mlogloss' if not is_bin else 'logloss',
                    use_label_encoder=False if not is_bin else None)

            interp = {
                '🌲 Árbol de Decisión': 'Reglas interpretables y visualizables',
                '🌲🌲 Random Forest':   'Alta precisión, robusto al sobreajuste',
                '⚡ Gradient Boosting': 'Muy preciso con datos complejos',
                '📉 Reg. Logística':    'Modelo lineal rápido e interpretable',
                '🚀 XGBoost':           'Estado del arte, altamente optimizado',
            }

            prog = st.progress(0); status = st.empty()
            skf  = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            X_cv = np.vstack([X_tr, X_val])
            y_cv = np.concatenate([y_tr, y_val])
            new_res = {k:v for k,v in st.session_state.results.items() if 'Baseline' in k}

            for i,(nom,mod) in enumerate(modelos.items()):
                status.text(f"⏳ Entrenando {nom}...")
                try:
                    mod.fit(X_tr, y_tr)
                    y_pred = mod.predict(X_te)
                    y_prob = mod.predict_proba(X_te)
                    acc,prec,rec,f1,auc = calc_metrics(y_te, y_pred, y_prob, is_bin, n_cls)

                    cv_mean, cv_std = None, None
                    if usar_cv:
                        try:
                            scoring = 'roc_auc' if is_bin else 'roc_auc_ovr_weighted'
                            cv_sc = cross_val_score(mod, X_cv, y_cv, cv=skf,
                                                    scoring=scoring, n_jobs=-1)
                            cv_mean = float(cv_sc.mean())
                            cv_std  = float(cv_sc.std())
                        except Exception:
                            # fallback a accuracy si AUC falla
                            try:
                                cv_sc = cross_val_score(mod, X_cv, y_cv, cv=skf,
                                                        scoring='accuracy', n_jobs=-1)
                                cv_mean = float(cv_sc.mean())
                                cv_std  = float(cv_sc.std())
                            except Exception:
                                cv_mean, cv_std = None, None

                    new_res[nom] = dict(
                        Accuracy=acc, Precision=prec, Recall=rec,
                        F1=f1, AUC=auc, CV_mean=cv_mean, CV_std=cv_std,
                        Interpretación=interp.get(nom,''),
                        model=mod, y_pred=y_pred, y_prob=y_prob
                    )

                    if 'Random Forest' in nom:
                        st.session_state.rf_importances = mod.feature_importances_
                        st.session_state.rf_names = fnams
                    if 'Árbol' in nom:
                        st.session_state.dt_model = mod
                        st.session_state.dt_names = fnams

                except Exception as em:
                    new_res[nom] = dict(
                        Accuracy=0, Precision=0, Recall=0, F1=0, AUC=None,
                        CV_mean=None, CV_std=None,
                        Interpretación=f'Error: {em}',
                        y_pred=np.zeros(len(y_te)), y_prob=None
                    )
                    st.warning(f"⚠️ {nom}: {em}")

                prog.progress((i+1)/len(modelos))

            st.session_state.results = new_res
            st.session_state.models_trained = True
            status.empty(); prog.empty()
            n_ok = sum(1 for v in new_res.values() if v.get('Accuracy',0)>0 and 'Baseline' not in k
                       for k in [list(new_res.keys())[list(new_res.values()).index(v)]])
            st.markdown(H("sbox",f"✅ Modelos entrenados exitosamente."), unsafe_allow_html=True)

        except Exception as e:
            show_err("Error entrenando modelos.", e)

    # Importancia RF
    if st.session_state.rf_importances is not None and st.session_state.rf_names:
        st.markdown("---")
        st.markdown("### 📊 Importancia de Variables — Random Forest")
        imp_df = pd.DataFrame({
            'Variable': st.session_state.rf_names,
            'Importancia': st.session_state.rf_importances
        }).sort_values('Importancia', ascending=False).head(30)
        fig = px.bar(imp_df, x='Importancia', y='Variable', orientation='h',
                     title='Feature Importance (Top 30)',
                     color='Importancia', color_continuous_scale='Blues',
                     text=imp_df['Importancia'].round(3))
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          height=max(400, len(imp_df)*22))
        st.plotly_chart(fig, use_container_width=True)

    # Árbol visual
    if st.session_state.dt_model is not None:
        st.markdown("---")
        st.markdown("### 🌲 Árbol de Decisión Visual")
        dt = st.session_state.dt_model
        names_dt = (st.session_state.dt_names or
                    [f'f{i}' for i in range(dt.n_features_in_)])
        cls_names = st.session_state.class_names or None
        max_dv = st.slider("Niveles visibles", 2, min(6, max(2, dt.get_depth())), 3)
        try:
            fig, ax = plt.subplots(figsize=(22, 9))
            sk_tree.plot_tree(dt, max_depth=max_dv,
                              feature_names=names_dt[:dt.n_features_in_],
                              class_names=cls_names,
                              filled=True, rounded=True, fontsize=8, ax=ax)
            ax.set_title(f'Árbol de Decisión — profundidad total={dt.get_depth()}, '
                         f'mostrando ≤{max_dv} niveles', fontsize=12, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            with st.expander("📄 Reglas en texto"):
                st.code(export_text(dt, feature_names=names_dt[:dt.n_features_in_],
                                    max_depth=5), language='text')
        except Exception as edt:
            st.warning(f"⚠️ No se pudo dibujar el árbol: {edt}")

# =============================================================================
# MÓDULO 6: EVALUACIÓN
# =============================================================================
elif menu == "📈 Evaluación":
    banner("📈 Evaluación de Modelos", "Matriz de Confusión · Curva ROC · Reporte detallado")

    results  = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}
    if not modelos_e:
        st.markdown(H("wbox","⚠️ Primero entrena los modelos."), unsafe_allow_html=True); st.stop()

    y_te    = st.session_state.y_test
    is_bin  = st.session_state.is_binary
    n_cls   = st.session_state.n_classes
    cls_nam = st.session_state.class_names or [str(i) for i in range(n_cls)]

    modelo_sel = st.selectbox("Modelo a evaluar", list(modelos_e.keys()))
    res   = modelos_e[modelo_sel]
    y_pred = res['y_pred']
    y_prob = res['y_prob']

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(l,v) in zip([c1,c2,c3,c4,c5],[
        ("Accuracy",  f"{res['Accuracy']:.3f}"),
        ("Precision", f"{res['Precision']:.3f}"),
        ("Recall",    f"{res['Recall']:.3f}"),
        ("F1-Score",  f"{res['F1']:.3f}"),
        ("AUC",       f"{res['AUC']:.3f}" if res['AUC'] else "N/A")]):
        with col: st.markdown(mc(v,l, g=(l=="AUC" and res['AUC'] and res['AUC']>0.7)),
                               unsafe_allow_html=True)

    if res.get('CV_mean') is not None:
        st.markdown(H("sbox",f"📊 <b>Cross-Validation {st.session_state.get('cv_folds',5)}-fold AUC: "
                      f"{res['CV_mean']:.4f} ± {res['CV_std']:.4f}</b>"), unsafe_allow_html=True)

    st.markdown("---")
    c1,c2 = st.columns(2)

    # Matriz de confusión
    with c1:
        st.markdown("#### 🔲 Matriz de Confusión")
        st.markdown(H("ibox","• <b>Diagonal</b> = predicciones correctas (VP, VN)<br>"
                      "• <b>Fuera de diagonal</b> = errores (FP, FN)"), unsafe_allow_html=True)
        try:
            cm = confusion_matrix(y_te, y_pred)
            n_show = len(cls_nam)
            fs = max(7, 14-n_show)
            fig, ax = plt.subplots(figsize=(max(5,n_show+2), max(4,n_show+1)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=cls_nam, yticklabels=cls_nam,
                        linewidths=1, linecolor='white',
                        annot_kws={"size":fs})
            ax.set_xlabel('Predicho', fontweight='bold')
            ax.set_ylabel('Real', fontweight='bold')
            ax.set_title(f'Matriz de Confusión\n{modelo_sel}', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            if is_bin and cm.size==4:
                tn,fp,fn,tp = cm.ravel()
                mc1,mc2 = st.columns(2)
                mc1.metric("VP (Verdaderos Pos.)", int(tp))
                mc2.metric("VN (Verdaderos Neg.)", int(tn))
                mc1.metric("FP (Falsos Pos.)", int(fp))
                mc2.metric("FN (Falsos Neg.)", int(fn))
        except Exception as ecm:
            st.warning(f"⚠️ Matriz de confusión: {ecm}")

    # Curva ROC
    with c2:
        st.markdown("#### 📉 Curva ROC")
        st.markdown(H("ibox","Más cerca de la esquina superior izquierda = mejor modelo. "
                      "AUC=1 perfecto · AUC=0.5 aleatorio."), unsafe_allow_html=True)
        if y_prob is None:
            st.info("Curva ROC no disponible (modelo sin predict_proba).")
        elif is_bin:
            try:
                fpr, tpr, _ = roc_curve(y_te, y_prob[:,1])
                auc_v = res['AUC'] or 0
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy',
                                         fillcolor='rgba(45,106,159,.15)',
                                         name=f'{modelo_sel} (AUC={auc_v:.3f})',
                                         line=dict(width=3,color='#2d6a9f')))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',
                                         line=dict(dash='dash',color='gray')))
                fig.update_layout(xaxis_title='FPR',yaxis_title='TPR',
                                  plot_bgcolor='white',paper_bgcolor='white',
                                  title=f'ROC — {modelo_sel}')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as er:
                st.warning(f"⚠️ ROC binaria: {er}")
        else:
            # Multiclase OVR
            try:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_te, classes=list(range(n_cls)))
                fig = go.Figure()
                for j in range(n_cls):
                    try:
                        fpr_j,tpr_j,_ = roc_curve(y_bin[:,j], y_prob[:,j])
                        auc_j = float(roc_auc_score(y_bin[:,j], y_prob[:,j]))
                        fig.add_trace(go.Scatter(x=fpr_j, y=tpr_j, mode='lines',
                                                 name=f'{cls_nam[j]} (AUC={auc_j:.3f})',
                                                 line=dict(width=2,color=PALETTE[j%len(PALETTE)])))
                    except Exception:
                        pass
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',
                                         line=dict(dash='dash',color='gray')))
                fig.update_layout(title=f'ROC OVR — {modelo_sel}',
                                  xaxis_title='FPR',yaxis_title='TPR',
                                  plot_bgcolor='white',paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as erm:
                st.warning(f"⚠️ ROC multiclase: {erm}")

    # ROC comparativa
    st.markdown("---")
    st.markdown("#### 📊 ROC Comparativa — Todos los modelos")
    try:
        fig = go.Figure()
        for i,(nom,r) in enumerate(modelos_e.items()):
            if r.get('y_prob') is None: continue
            if is_bin:
                try:
                    fpr,tpr,_ = roc_curve(y_te, r['y_prob'][:,1])
                    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
                                             name=f"{nom} (AUC={r['AUC']:.3f})" if r['AUC'] else nom,
                                             line=dict(width=2,color=PALETTE[i%len(PALETTE)])))
                except Exception: pass
        for nom,r in results.items():
            if 'Baseline' in nom and r.get('y_prob') is not None:
                try:
                    fpr,tpr,_ = roc_curve(y_te, r['y_prob'][:,1])
                    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',name=nom,
                                             line=dict(dash='dot',color='gray')))
                except Exception: pass
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',
                                 line=dict(dash='dash',color='black')))
        fig.update_layout(xaxis_title='FPR',yaxis_title='TPR',
                          plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as erc:
        st.warning(f"⚠️ ROC comparativa: {erc}")

    with st.expander("📋 Reporte completo"):
        try:
            rpt = classification_report(y_te, y_pred, target_names=cls_nam, zero_division=0)
            st.code(rpt, language='text')
        except Exception as erp:
            st.warning(f"⚠️ Reporte: {erp}")

# =============================================================================
# MÓDULO 7: COMPARATIVA
# =============================================================================
elif menu == "📋 Comparativa":
    banner("📋 Comparativa de Modelos", "Tabla · Barras · Radar · Mejor modelo automático")

    results = st.session_state.results
    if not results:
        st.markdown(H("wbox","⚠️ Ejecuta Baseline y Clasificación primero."),
                    unsafe_allow_html=True); st.stop()

    filas = []
    for nom,r in results.items():
        filas.append({
            'Modelo': nom,
            'Accuracy':  round(r['Accuracy'],4),
            'Precision': round(r['Precision'],4),
            'Recall':    round(r['Recall'],4),
            'F1-Score':  round(r['F1'],4),
            'AUC':       round(r['AUC'],4) if r.get('AUC') else 0,
            'CV AUC':    (f"{r['CV_mean']:.4f} ± {r['CV_std']:.4f}"
                          if r.get('CV_mean') is not None else '—'),
            'Interpretación': r.get('Interpretación',''),
        })

    df_c = pd.DataFrame(filas).sort_values('AUC', ascending=False).reset_index(drop=True)
    mejor = df_c.iloc[0]

    st.markdown(H("sbox",f"🏆 <b>Mejor modelo: {mejor['Modelo']}</b> — "
                  f"AUC={mejor['AUC']} · F1={mejor['F1-Score']} · Acc={mejor['Accuracy']}"
                  f"{' · CV='+mejor['CV AUC'] if mejor['CV AUC']!='—' else ''}"),
                unsafe_allow_html=True)

    st.dataframe(df_c, use_container_width=True, hide_index=True)

    # Barras
    mp = ['Accuracy','Precision','Recall','F1-Score','AUC']
    fig = go.Figure()
    for i,m in enumerate(mp):
        fig.add_trace(go.Bar(name=m, x=df_c['Modelo'], y=df_c[m], marker_color=PALETTE[i]))
    fig.update_layout(barmode='group', title='Comparativa de Métricas',
                      plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h',yanchor='bottom',y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Radar
    cats = ['Accuracy','Precision','Recall','F1-Score','AUC']
    fig = go.Figure()
    for i,(_,row) in enumerate(df_c.iterrows()):
        vals = [row[c] for c in cats]+[row[cats[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]], fill='toself',
                                      name=row['Modelo'], opacity=0.5,
                                      line_color=PALETTE[i%len(PALETTE)],
                                      fillcolor=PALETTE[i%len(PALETTE)]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                      title="Radar Chart de Modelos")
    st.plotly_chart(fig, use_container_width=True)

    csv_c = df_c.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV", data=csv_c,
                       file_name="comparativa_modelos.csv", mime="text/csv")

# =============================================================================
# MÓDULO 8: PREDICCIÓN INDIVIDUAL
# =============================================================================
elif menu == "🔮 Predicción Individual":
    banner("🔮 Predicción Individual", "Ingresa los datos de un cliente → obtén predicción instantánea")

    if not st.session_state.models_trained:
        st.markdown(H("wbox","⚠️ Primero entrena los modelos."), unsafe_allow_html=True); st.stop()

    results   = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}
    scaler    = st.session_state.scaler
    fnams     = st.session_state.feature_names_model
    is_bin    = st.session_state.is_binary
    le_y      = st.session_state.le_target
    cls_nam   = st.session_state.class_names

    if not fnams or scaler is None:
        show_err("Pipeline no disponible. Re-ejecuta Partición."); st.stop()

    modelo_p = st.selectbox("Modelo para predicción", list(modelos_e.keys()))

    st.markdown("### 📝 Datos del Cliente")
    df_ref = st.session_state.df_proc
    valores = {}

    # Formulario dinámico — máx 30 campos por fila de 3
    campos = fnams[:60]  # limitar a 60 features máximo
    if len(fnams) > 60:
        st.caption(f"ℹ️ Mostrando primeros 60 de {len(fnams)} features.")

    cols_per_row = 3
    rows_form = [campos[i:i+cols_per_row] for i in range(0,len(campos),cols_per_row)]
    for row_f in rows_form:
        cols_ui = st.columns(len(row_f))
        for col_ui,feat in zip(cols_ui,row_f):
            with col_ui:
                if df_ref is not None and feat in df_ref.columns:
                    col_d = pd.to_numeric(df_ref[feat], errors='coerce').dropna()
                    if len(col_d) > 0:
                        mn  = float(col_d.min())
                        mx  = float(col_d.max())
                        med = float(col_d.median())
                        valores[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                         value=med, key=f"pi_{feat}")
                    else:
                        valores[feat] = st.number_input(feat, value=0.0, key=f"pi_{feat}")
                else:
                    valores[feat] = st.number_input(feat, value=0.0, key=f"pi_{feat}")

    # Rellenar features no mostradas con su mediana
    for feat in fnams:
        if feat not in valores:
            if df_ref is not None and feat in df_ref.columns:
                col_d = pd.to_numeric(df_ref[feat], errors='coerce').dropna()
                valores[feat] = float(col_d.median()) if len(col_d)>0 else 0.0
            else:
                valores[feat] = 0.0

    if st.button("🔮 Predecir", type="primary", use_container_width=True):
        try:
            X_new = np.array([[float(valores.get(f,0)) for f in fnams]], dtype='float64')
            X_new_sc = scaler.transform(X_new)
            mod_obj  = modelos_e[modelo_p]['model']
            y_prob_n = mod_obj.predict_proba(X_new_sc)[0]
            y_pred_n = mod_obj.predict(X_new_sc)[0]
            clase_p  = le_y.inverse_transform([int(y_pred_n)])[0] if le_y else str(y_pred_n)

            st.markdown("---")
            st.markdown("### 🎯 Resultado")

            if is_bin:
                prob_pos = float(y_prob_n[1])
                color_g  = "green" if prob_pos>0.5 else "blue"
                resp     = "✅ SÍ RESPONDERÁ" if prob_pos>0.5 else "❌ NO RESPONDERÁ"
                conf     = "Alta" if abs(prob_pos-.5)>.3 else "Media" if abs(prob_pos-.5)>.15 else "Baja"
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(mc(resp,"Predicción",g=prob_pos>0.5), unsafe_allow_html=True)
                with c2: st.markdown(mc(f"{prob_pos:.1%}","Prob. positiva",g=prob_pos>0.5), unsafe_allow_html=True)
                with c3: st.markdown(mc(conf,"Confianza"), unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=prob_pos*100,
                    title={'text':"Probabilidad de Respuesta (%)"},
                    delta={'reference':50},
                    gauge={'axis':{'range':[0,100]},
                           'bar':{'color':'#27ae60' if prob_pos>0.5 else '#e74c3c'},
                           'steps':[{'range':[0,30],'color':'#f8d7da'},
                                    {'range':[30,70],'color':'#fff3cd'},
                                    {'range':[70,100],'color':'#d4edda'}],
                           'threshold':{'line':{'color':'black','width':4},'value':50}}))
                fig.update_layout(height=300, paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                if   prob_pos>=0.70: st.markdown(H("sbox","🌟 <b>Prioridad ALTA</b> — Contactar con oferta premium."), unsafe_allow_html=True)
                elif prob_pos>=0.40: st.markdown(H("wbox","🔄 <b>Prioridad MEDIA</b> — Incluir en campaña estándar."), unsafe_allow_html=True)
                else:                st.markdown(H("ibox","💤 <b>Prioridad BAJA</b> — Solo si el costo es bajo."), unsafe_allow_html=True)
            else:
                st.markdown(H("sbox",f"🎯 <b>Clase predicha: {clase_p}</b>"), unsafe_allow_html=True)
                prob_df = pd.DataFrame({'Clase': cls_nam or [str(i) for i in range(len(y_prob_n))],
                                        'Probabilidad': [round(float(p),4) for p in y_prob_n]})
                prob_df = prob_df.sort_values('Probabilidad', ascending=False)
                fig = px.bar(prob_df, x='Clase', y='Probabilidad', color='Probabilidad',
                             color_continuous_scale='Blues', title='Probabilidades por Clase',
                             text=prob_df['Probabilidad'].apply(lambda x:f"{x:.1%}"))
                fig.update_traces(textposition='outside')
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Datos ingresados"):
                st.dataframe(pd.DataFrame([valores]), use_container_width=True)

        except Exception as ep:
            show_err("Error en la predicción.", ep)

# =============================================================================
# MÓDULO 9: INFORME GERENCIAL
# =============================================================================
elif menu == "💼 Informe Gerencial":
    banner("💼 Informe Gerencial", "Resultados ejecutivos · Recomendaciones · Plan de acción")

    results   = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}
    if not modelos_e:
        st.markdown(H("wbox","⚠️ Primero entrena los modelos."), unsafe_allow_html=True); st.stop()

    mejor_nom = max(modelos_e, key=lambda k: modelos_e[k].get('AUC') or 0)
    mejor     = modelos_e[mejor_nom]
    auc_v     = mejor.get('AUC') or 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mc(f"{auc_v:.1%}",f"AUC · {mejor_nom}",g=True), unsafe_allow_html=True)
    with c2: st.markdown(mc(f"{mejor['F1']:.1%}","F1-Score"), unsafe_allow_html=True)
    with c3: st.markdown(mc(f"{mejor['Accuracy']:.1%}","Exactitud"), unsafe_allow_html=True)
    with c4:
        cv_txt = f"{mejor['CV_mean']:.1%}" if mejor.get('CV_mean') else f"{mejor['Precision']:.1%}"
        cv_lbl = "CV AUC" if mejor.get('CV_mean') else "Precisión"
        st.markdown(mc(cv_txt, cv_lbl, g=bool(mejor.get('CV_mean'))), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div class="card"><h3>📋 Resumen Ejecutivo</h3>
    <p>El sistema de minería de datos analizó el dataset y construyó un modelo predictivo.
    El modelo <b>{mejor_nom}</b> obtuvo el mejor rendimiento con
    <b>AUC={auc_v:.1%}</b> y exactitud de <b>{mejor['Accuracy']:.1%}</b>.
    {f"Validado con Cross-Validation: CV AUC = {mejor['CV_mean']:.4f} ± {mejor['CV_std']:.4f}." if mejor.get('CV_mean') else ""}
    </p></div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card"><h3>🎯 Clientes a Priorizar</h3><ul>
        <li>🌟 <b>Premium:</b> Máxima prioridad — alta conversión esperada.</li>
        <li>🔄 <b>Frecuentes:</b> Fidelización y recompensas por lealtad.</li>
        <li>💤 <b>Inactivos:</b> Reactivación con incentivos especiales.</li>
        <li>💰 <b>Económicos:</b> Descuentos y promociones directas.</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><h3>⚡ Plan de Acción</h3><ol>
        <li>Filtrar la base con <b>{mejor_nom}</b> antes de lanzar la campaña.</li>
        <li>Personalizar mensajes por segmento (clúster).</li>
        <li>Monitorear conversión real vs predicha mensualmente.</li>
        <li>Reentrenar el modelo cada 3-6 meses.</li>
        <li>Piloto A/B con 10% de la base antes de escalar.</li>
        </ol></div>""", unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("""<div class="card"><h3>⚠️ Limitaciones</h3><ul>
        <li>📌 El modelo aprende de datos históricos — puede no capturar cambios recientes.</li>
        <li>📌 No considera factores externos (economía, competencia, estacionalidad).</li>
        <li>📌 La calidad del modelo depende de la calidad de los datos.</li>
        <li>📌 Validar siempre con un piloto controlado antes de escalar.</li>
        </ul></div>""", unsafe_allow_html=True)
    with c4:
        if 'Baseline (Dummy)' in results:
            b_auc = results['Baseline (Dummy)'].get('AUC') or 0
            mejora = auc_v - b_auc
            st.markdown(f"""<div class="card"><h3>📊 Impacto vs Baseline</h3>
            <p>El modelo <b>{mejor_nom}</b> supera al modelo aleatorio en
            <b>+{mejora:.1%} AUC</b>.</p>
            <p>Esto permite enfocar el presupuesto de campaña en los clientes con mayor
            probabilidad de respuesta, <b>reduciendo costos y mejorando el ROI</b>.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(H("sbox",f"💼 <b>Conclusión:</b> El modelo <b>{mejor_nom}</b> está listo para uso operativo. "
                  f"Con un AUC de <b>{auc_v:.1%}</b> puede anticipar la respuesta de los clientes "
                  f"a campañas comerciales, optimizando recursos y mejorando la tasa de conversión."),
                unsafe_allow_html=True)
