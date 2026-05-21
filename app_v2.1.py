# =============================================================================
# SISTEMA DE SEGMENTACIÓN Y PREDICCIÓN DE CLIENTES — v2.1
# Fix: soporte multiclase completo, fix TypeError pandas arrow,
#      métricas automáticas según tipo de problema (binario vs multiclase),
#      ROC multiclase con OVR, SMOTE solo para binario
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
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, silhouette_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree as sk_tree
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Sistema Minería de Datos v2.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 18px; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 4px;
    }
    .metric-card h2 { font-size: 1.8rem; margin: 0; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card-green {
        background: linear-gradient(135deg, #1a5c38 0%, #27ae60 100%);
        padding: 18px; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 4px;
    }
    .metric-card-green h2 { font-size: 1.8rem; margin: 0; }
    .metric-card-green p  { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .section-card {
        background: white; padding: 22px; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 18px; border-left: 5px solid #2d6a9f;
    }
    .info-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        padding: 14px 18px; border-radius: 10px;
        border-left: 4px solid #17a2b8; margin: 10px 0; font-size: 0.92rem; color: #2c3e50;
    }
    .warning-box {
        background: #fff3cd; padding: 14px 18px; border-radius: 10px;
        border-left: 4px solid #ffc107; margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 14px 18px; border-radius: 10px;
        border-left: 4px solid #28a745; margin: 10px 0;
    }
    .danger-box {
        background: #f8d7da; padding: 14px 18px; border-radius: 10px;
        border-left: 4px solid #dc3545; margin: 10px 0;
    }
    .main-title {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        color: white; padding: 28px; border-radius: 15px;
        text-align: center; margin-bottom: 22px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .main-title h1 { font-size: 1.8rem; margin-bottom: 6px; }
    .main-title p  { font-size: 0.95rem; opacity: 0.9; margin: 0; }
    hr { border: none; border-top: 2px solid #e9ecef; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
defaults = {
    'df_raw': None, 'df_proc': None,
    'target_col': None, 'feature_cols': [], 'num_cols': [], 'cat_cols': [],
    'X_train': None, 'X_val': None, 'X_test': None,
    'y_train': None, 'y_val': None, 'y_test': None,
    'scaler': None, 'le_target': None,
    'results': {}, 'best_k': 3, 'df_clustered': None,
    'rf_importances': None, 'rf_names': None,
    'models_trained': False, 'partition_done': False,
    'is_binary': True, 'n_classes': 2, 'class_names': [],
    'dt_model': None, 'dt_names': None,
    'feature_names_model': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

PALETTE = ['#2d6a9f','#e74c3c','#27ae60','#f39c12','#8e44ad','#1abc9c','#e67e22','#2ecc71']

# =============================================================================
# HELPERS GLOBALES
# =============================================================================
def mcard(val, label, color="blue"):
    css = "metric-card-green" if color == "green" else "metric-card"
    return f'<div class="{css}"><h2>{val}</h2><p>{label}</p></div>'

def ibox(txt):  return f'<div class="info-box">{txt}</div>'
def sbox(txt):  return f'<div class="success-box">{txt}</div>'
def wbox(txt):  return f'<div class="warning-box">{txt}</div>'
def dbox(txt):  return f'<div class="danger-box">{txt}</div>'

def banner(title, subtitle):
    st.markdown(f'<div class="main-title"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def get_avg():
    """Retorna el parámetro average correcto según tipo de problema."""
    return 'binary' if st.session_state.is_binary else 'weighted'

def calc_metrics(y_true, y_pred, y_prob=None):
    """Calcula métricas compatibles con binario y multiclase."""
    avg = get_avg()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)

    auc = None
    if y_prob is not None:
        try:
            if st.session_state.is_binary:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception:
            auc = None

    return acc, prec, rec, f1, auc

def safe_float_cols(df, cols):
    """Convierte columnas a float64 de forma segura (fix arrow dtype)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            try:
                out[c] = out[c].astype('float64')
            except Exception:
                try:
                    out[c] = pd.to_numeric(out[c], errors='coerce').astype('float64')
                except Exception:
                    pass
    return out

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🎯 Minería de Datos v2.1")
    st.markdown("---")
    menu = st.radio("Módulos", options=[
        "🏠 Inicio", "📁 Carga de Datos", "⚙️ Preprocesamiento",
        "📊 Partición y Baseline", "🔵 Segmentación", "🤖 Clasificación",
        "📈 Evaluación de Modelos", "📋 Comparativa de Modelos",
        "🔮 Predicción Individual", "💼 Interpretación Gerencial"
    ], label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.df_raw is not None:
        df_s = st.session_state.df_raw
        st.success(f"✅ Dataset: {df_s.shape[0]:,} filas · {df_s.shape[1]} cols")
        if st.session_state.target_col:
            tipo = "Binario" if st.session_state.is_binary else f"Multiclase ({st.session_state.n_classes} clases)"
            st.success(f"🎯 Objetivo: `{st.session_state.target_col}`\n\n📌 {tipo}")
        else:
            st.warning("⚠️ Variable objetivo no definida")
        if st.session_state.partition_done:
            st.success("✂️ Partición completada")
        if st.session_state.models_trained:
            st.success("🤖 Modelos entrenados")
    else:
        st.info("📤 Sin dataset cargado")
    st.markdown("---")
    st.caption("Universidad Peruana Unión · Minería de Datos · 2025")

# =============================================================================
# MÓDULO 0: INICIO
# =============================================================================
if menu == "🏠 Inicio":
    banner("🎯 SISTEMA DE SEGMENTACIÓN Y PREDICCIÓN DE CLIENTES v2.1",
           "Soporte Binario y Multiclase · Pipeline completo sin Data Leakage")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mcard("10","Módulos"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("5","Algoritmos ML"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("Binario\n+\nMulticlase","Tipos de problema"), unsafe_allow_html=True)
    with c4: st.markdown(mcard("CV 5-fold","Cross-Validation"), unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns([3,2])
    with c1:
        st.markdown("""<div class="section-card"><h3>🆕 Novedades v2.1</h3><ul>
        <li>✅ <b>Soporte multiclase completo</b> — métricas weighted automáticas</li>
        <li>✅ <b>ROC multiclase OVR</b> — curva por cada clase</li>
        <li>✅ <b>Fix TypeError pandas arrow</b> — compatible con cualquier CSV</li>
        <li>✅ <b>Selector dinámico</b> de variable objetivo</li>
        <li>✅ <b>Cross-Validation</b> estratificada 5-fold</li>
        <li>✅ <b>SMOTE</b> para datasets binarios desbalanceados</li>
        <li>✅ <b>Árbol de Decisión visual</b> + reglas en texto</li>
        <li>✅ <b>Predicción individual</b> con gauge de probabilidad</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="section-card"><h3>📋 Flujo recomendado</h3><ol>
        <li>📁 Cargar CSV → elegir objetivo</li>
        <li>⚙️ Preprocesar</li>
        <li>📊 Particionar + Baseline</li>
        <li>🔵 Segmentar (opcional)</li>
        <li>🤖 Entrenar modelos</li>
        <li>📈 Evaluar</li>
        <li>📋 Comparar</li>
        <li>🔮 Predecir cliente nuevo</li>
        <li>💼 Informe gerencial</li>
        </ol></div>""", unsafe_allow_html=True)
    st.markdown(ibox("📤 <b>Empieza en 📁 Carga de Datos.</b> El sistema detecta automáticamente si tu "
                     "problema es <b>binario</b> (0/1) o <b>multiclase</b> y ajusta todas las métricas."),
                unsafe_allow_html=True)

# =============================================================================
# MÓDULO 1: CARGA DE DATOS
# =============================================================================
elif menu == "📁 Carga de Datos":
    banner("📁 Carga y Exploración de Datos",
           "Sube tu CSV · Selecciona la variable objetivo · Detección automática del tipo de problema")

    uploaded = st.file_uploader("Selecciona tu archivo CSV", type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df_raw = df
            for k in ['df_proc','target_col','feature_cols','num_cols','cat_cols',
                      'X_train','X_val','X_test','y_train','y_val','y_test',
                      'results','df_clustered','rf_importances','rf_names',
                      'models_trained','partition_done','preprocessor','scaler',
                      'le_target','dt_model','dt_names','feature_names_model']:
                st.session_state[k] = defaults.get(k, None)
            st.session_state['results'] = {}
            st.session_state['feature_names_model'] = []
            st.success(f"✅ Cargado: **{uploaded.name}** — {df.shape[0]:,} filas · {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"❌ Error: {e}"); st.stop()

    if st.session_state.df_raw is None:
        st.markdown(wbox("⚠️ Sube un archivo CSV para comenzar."), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df_raw

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mcard(f"{df.shape[0]:,}","Filas"), unsafe_allow_html=True)
    with c2: st.markdown(mcard(df.shape[1],"Columnas"), unsafe_allow_html=True)
    with c3: st.markdown(mcard(len(df.select_dtypes('number').columns),"Numéricas"), unsafe_allow_html=True)
    with c4: st.markdown(mcard(len(df.select_dtypes('object').columns),"Categóricas"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Selecciona la Variable Objetivo")
    st.markdown(ibox("Elige la columna que contiene las clases a predecir. Puede ser binaria (0/1) o multiclase."),
                unsafe_allow_html=True)

    col_options = list(df.columns)
    default_idx = 0
    if st.session_state.target_col and st.session_state.target_col in col_options:
        default_idx = col_options.index(st.session_state.target_col)
    elif 'respondio_campana' in col_options:
        default_idx = col_options.index('respondio_campana')

    target_sel = st.selectbox("Variable objetivo:", col_options, index=default_idx)

    if st.button("✅ Confirmar variable objetivo", type="primary"):
        unique_vals = df[target_sel].dropna().unique()
        n_unique = len(unique_vals)
        if n_unique > 20:
            st.error(f"❌ '{target_sel}' tiene {n_unique} valores únicos. Elige una columna categórica o binaria.")
        elif n_unique < 2:
            st.error(f"❌ '{target_sel}' solo tiene 1 valor único. Necesita al menos 2 clases.")
        else:
            is_bin = n_unique == 2
            st.session_state.target_col = target_sel
            st.session_state.is_binary  = is_bin
            st.session_state.n_classes  = n_unique
            st.session_state.class_names = sorted([str(v) for v in unique_vals])
            tipo = "✅ Binario (2 clases)" if is_bin else f"📊 Multiclase ({n_unique} clases)"
            st.success(f"Variable objetivo: **{target_sel}** — {tipo} — Clases: {sorted(unique_vals)}")

    if st.session_state.target_col:
        target = st.session_state.target_col
        vc = df[target].value_counts()
        vc_pct = df[target].value_counts(normalize=True)
        minority_pct = vc_pct.min() * 100

        c1,c2 = st.columns([2,3])
        with c1:
            if st.session_state.is_binary and minority_pct < 20:
                st.markdown(dbox(f"⚠️ <b>Dataset DESBALANCEADO</b><br>Clase minoritaria: <b>{minority_pct:.1f}%</b>"),
                            unsafe_allow_html=True)
            elif not st.session_state.is_binary:
                st.markdown(ibox(f"📊 <b>Problema MULTICLASE</b><br>{st.session_state.n_classes} clases detectadas.<br>"
                                 f"Se usarán métricas <b>weighted</b>."), unsafe_allow_html=True)
            else:
                st.markdown(sbox(f"✅ Dataset balanceado — Clase minoritaria: {minority_pct:.1f}%"),
                            unsafe_allow_html=True)
            st.dataframe(vc.rename_axis('Clase').reset_index(name='Cantidad'), use_container_width=True)
        with c2:
            fig = px.pie(df, names=target, title="Distribución Variable Objetivo",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs(["👁️ Vista Previa","🔍 Tipos","⚠️ Nulos","📊 Estadísticas"])
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
    with tab2:
        tipos = pd.DataFrame({'Variable':df.columns,'Tipo':df.dtypes.values,
                              'Únicos':[df[c].nunique() for c in df.columns],
                              'Nulos':[df[c].isnull().sum() for c in df.columns],
                              'Ejemplo':[str(df[c].iloc[0]) if len(df)>0 else '' for c in df.columns]})
        st.dataframe(tipos, use_container_width=True)
    with tab3:
        nulos = pd.DataFrame({'Variable':df.columns,
                              'Nulos':df.isnull().sum().values,
                              '% Nulos':(df.isnull().sum().values/len(df)*100).round(2)})
        st.dataframe(nulos, use_container_width=True)
        total_n = df.isnull().sum().sum()
        st.markdown((sbox("✅ Sin valores nulos.") if total_n==0
                     else wbox(f"⚠️ {total_n} nulos — se imputarán en Preprocesamiento.")),
                    unsafe_allow_html=True)
    with tab4:
        st.dataframe(df.describe(include='all').T.round(3), use_container_width=True)

    # Distribuciones numéricas
    st.markdown("### 📊 Distribuciones de Variables Numéricas")
    num_c = df.select_dtypes(include='number').columns.tolist()
    if st.session_state.target_col in num_c:
        num_c.remove(st.session_state.target_col)
    if num_c:
        rows = [num_c[i:i+3] for i in range(0,len(num_c),3)]
        for row in rows:
            cols = st.columns(len(row))
            for col,c in zip(cols,row):
                with col:
                    fig = px.histogram(df, x=c, nbins=30, title=c,
                                       color_discrete_sequence=['#2d6a9f'])
                    fig.update_layout(margin=dict(t=35,b=10), height=250)
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MÓDULO 2: PREPROCESAMIENTO
# =============================================================================
elif menu == "⚙️ Preprocesamiento":
    banner("⚙️ Preprocesamiento de Datos",
           "Imputación · Codificación · Correlación · Sin Data Leakage")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Primero carga un dataset."); st.stop()
    if not st.session_state.target_col:
        st.warning("⚠️ Primero confirma la variable objetivo en Carga de Datos."); st.stop()

    df = st.session_state.df_raw.copy()
    target = st.session_state.target_col

    st.markdown(ibox("🔒 <b>Data Leakage evitado:</b> El ColumnTransformer se ajusta <i>solo con datos de "
                     "entrenamiento</i>. El escalado nunca se aplica antes de partir el dataset."),
                unsafe_allow_html=True)

    feature_cols = [c for c in df.columns if c != target]
    num_cols = df[feature_cols].select_dtypes(include='number').columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include='object').columns.tolist()

    with st.expander("1️⃣ Variables detectadas", expanded=True):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Numéricas:**"); st.write(num_cols if num_cols else ["Ninguna"])
        with c2:
            st.markdown("**Categóricas:**"); st.write(cat_cols if cat_cols else ["Ninguna"])
        st.markdown(f"**Variable objetivo:** `{target}` — "
                    f"{'Binaria' if st.session_state.is_binary else f'Multiclase ({st.session_state.n_classes} clases)'}")

    with st.expander("2️⃣ Imputación de nulos", expanded=True):
        antes = df.isnull().sum().sum()
        # FIX: convertir a float64 antes de imputar (fix arrow dtype)
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode())>0 else 'desconocido')
        despues = df.isnull().sum().sum()
        c1,c2 = st.columns(2)
        c1.metric("Nulos antes", int(antes))
        c2.metric("Nulos después", int(despues), delta=int(despues-antes))
        st.markdown(sbox("✅ Numéricas → mediana · Categóricas → moda"), unsafe_allow_html=True)

    with st.expander("3️⃣ Codificación de categóricas", expanded=True):
        if cat_cols:
            for c in cat_cols:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
            st.markdown(sbox(f"✅ LabelEncoder aplicado: {cat_cols}"), unsafe_allow_html=True)
        else:
            st.info("ℹ️ Sin categóricas.")

    with st.expander("4️⃣ Mapa de correlación", expanded=True):
        corr_df = df[[c for c in df.columns if c != target and df[c].dtype in ['float64','int64','int32']]]
        if corr_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(min(14, corr_df.shape[1]+2), min(10, corr_df.shape[1]+1)))
            mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
            sns.heatmap(corr_df.corr(), annot=corr_df.shape[1]<=12,
                        fmt='.2f', cmap='RdBu_r', center=0,
                        mask=mask, ax=ax, linewidths=0.5)
            ax.set_title("Matriz de Correlación", fontsize=13, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Actualizar columnas numéricas tras codificación
    num_cols_upd = [c for c in feature_cols if c in df.columns and
                    df[c].dtype in ['float64','int64','int32','float32']]
    st.session_state.num_cols     = num_cols_upd
    st.session_state.cat_cols     = []
    st.session_state.feature_cols = feature_cols
    st.session_state.df_proc      = df

    st.markdown(sbox("✅ <b>Preprocesamiento completado.</b> Continúa con Partición y Baseline."),
                unsafe_allow_html=True)

# =============================================================================
# MÓDULO 3: PARTICIÓN Y BASELINE
# =============================================================================
elif menu == "📊 Partición y Baseline":
    banner("📊 Partición de Datos y Modelo Baseline",
           "Split 70/15/15 · SMOTE (solo binario) · Baseline adaptativo")

    if st.session_state.df_proc is None:
        st.warning("⚠️ Primero ejecuta Preprocesamiento."); st.stop()

    df_proc = st.session_state.df_proc
    target   = st.session_state.target_col
    num_cols = st.session_state.num_cols
    is_bin   = st.session_state.is_binary

    if not num_cols:
        st.error("❌ No hay variables numéricas."); st.stop()

    st.markdown(ibox("📌 <b>Conjuntos:</b> Entrenamiento 70% · Validación 15% · Prueba 15%<br>"
                     "El StandardScaler se ajusta <b>solo con el 70% de entrenamiento</b>."),
                unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        usar_smote = st.checkbox(
            "🔁 Usar SMOTE (solo para binario desbalanceado)",
            value=False,
            disabled=(not SMOTE_AVAILABLE or not is_bin),
            help="SMOTE genera muestras sintéticas de la clase minoritaria."
        )
        if not SMOTE_AVAILABLE: st.caption("pip install imbalanced-learn")
        if not is_bin: st.caption("SMOTE solo disponible para clasificación binaria.")
    with c2:
        usar_cluster = st.checkbox(
            "🔵 Incluir clúster K-Means como feature",
            value=False,
            disabled=st.session_state.df_clustered is None,
        )
        if st.session_state.df_clustered is None:
            st.caption("Ejecuta Segmentación primero.")

    if st.button("✂️ Ejecutar Partición", type="primary", use_container_width=True):
        try:
            # Construir X
            if usar_cluster and st.session_state.df_clustered is not None:
                df_use = st.session_state.df_clustered.copy()
                extra = [c for c in ['cluster_kmeans'] if c in df_use.columns]
                cols_X = num_cols + extra
            else:
                df_use = df_proc.copy()
                cols_X = num_cols

            # FIX: asegurar float64 en todas las features
            df_use = safe_float_cols(df_use, cols_X)

            X = df_use[cols_X].copy()
            y_raw = df_use[target].copy()

            # Codificar y
            le_y = LabelEncoder()
            y = pd.Series(le_y.fit_transform(y_raw.astype(str)), index=y_raw.index)
            st.session_state.le_target = le_y

            # Split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

            # Escalar solo con train
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc   = scaler.transform(X_val)
            X_test_sc  = scaler.transform(X_test)

            # SMOTE (solo binario)
            if usar_smote and SMOTE_AVAILABLE and is_bin:
                sm = SMOTE(random_state=42)
                X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)
                st.markdown(sbox(f"✅ SMOTE: train balanceado → {pd.Series(y_train).value_counts().to_dict()}"),
                            unsafe_allow_html=True)

            # Guardar
            st.session_state.X_train = X_train_sc
            st.session_state.X_val   = X_val_sc
            st.session_state.X_test  = X_test_sc
            st.session_state.y_train = y_train
            st.session_state.y_val   = y_val
            st.session_state.y_test  = y_test
            st.session_state.scaler  = scaler
            st.session_state.feature_names_model = list(cols_X)
            st.session_state.partition_done = True
            st.session_state.models_trained = False
            st.session_state.results = {}

            # Métricas de partición
            c1,c2,c3,c4 = st.columns(4)
            for col,(lbl,n,pct) in zip([c1,c2,c3,c4],[
                ("📦 Total",len(X),100),("🎓 Train",len(X_train),70),
                ("🔍 Val",len(X_val),15),("🧪 Test",len(X_test),15)]):
                with col: st.markdown(mcard(f"{n:,}",f"{lbl} ({pct}%)"), unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=['Entrenamiento (70%)','Validación (15%)','Prueba (15%)'],
                y=[len(X_train),len(X_val),len(X_test)],
                marker_color=['#2d6a9f','#27ae60','#e74c3c'],
                text=[f'{v:,}' for v in [len(X_train),len(X_val),len(X_test)]],
                textposition='auto'))
            fig.update_layout(title="Distribución de Conjuntos",
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

            # ── BASELINE ──
            st.markdown("---")
            st.markdown("### 🎲 Modelo Baseline")
            st.markdown(ibox("📌 Si los modelos avanzados no superan al baseline, no aportan valor real."),
                        unsafe_allow_html=True)

            avg = get_avg()
            baselines = {
                'Baseline (Dummy)': DummyClassifier(strategy='most_frequent', random_state=42),
                'Baseline (Log.Reg)': LogisticRegression(max_iter=1000, random_state=42,
                                                          class_weight='balanced')
            }
            for nombre, modelo in baselines.items():
                modelo.fit(X_train_sc, y_train)
                y_pred_b = modelo.predict(X_test_sc)
                y_prob_b = modelo.predict_proba(X_test_sc)
                acc_b, prec_b, rec_b, f1_b, auc_b = calc_metrics(y_test, y_pred_b, y_prob_b)

                st.session_state.results[nombre] = {
                    'Accuracy': acc_b, 'Precision': prec_b, 'Recall': rec_b,
                    'F1-Score': f1_b, 'AUC': auc_b, 'CV_mean': None, 'CV_std': None,
                    'Interpretación': 'Modelo de referencia',
                    'y_pred': y_pred_b, 'y_prob': y_prob_b
                }

            c1,c2 = st.columns(2)
            for col, nombre in zip([c1,c2], list(baselines.keys())):
                with col:
                    r = st.session_state.results[nombre]
                    st.markdown(f"**{nombre}**")
                    mc1,mc2,mc3 = st.columns(3)
                    mc1.metric("Accuracy", f"{r['Accuracy']:.3f}")
                    mc2.metric("F1-Score", f"{r['F1-Score']:.3f}")
                    mc3.metric("AUC", f"{r['AUC']:.3f}" if r['AUC'] else "N/A")

            st.markdown(sbox("✅ Partición y baseline completados."), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error en la partición: {e}")
            import traceback; st.code(traceback.format_exc())

    elif st.session_state.partition_done:
        st.markdown(sbox("✅ Partición ya ejecutada. Puedes continuar con los demás módulos."),
                    unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,(lbl,n) in zip([c1,c2,c3,c4],[
            ("🎓 Train",len(st.session_state.X_train)),
            ("🔍 Val",len(st.session_state.X_val)),
            ("🧪 Test",len(st.session_state.X_test)),
            ("📐 Features",st.session_state.X_train.shape[1])]):
            with col: st.markdown(mcard(f"{n:,}",lbl), unsafe_allow_html=True)

# =============================================================================
# MÓDULO 4: SEGMENTACIÓN
# =============================================================================
elif menu == "🔵 Segmentación":
    banner("🔵 Segmentación de Clientes",
           "K-Means · Clustering Jerárquico · Silhouette · Perfiles automáticos")

    if st.session_state.df_proc is None:
        st.warning("⚠️ Primero ejecuta Preprocesamiento."); st.stop()

    df = st.session_state.df_proc.copy()
    target   = st.session_state.target_col
    num_cols = st.session_state.num_cols
    feat_cluster = [c for c in num_cols if c in df.columns]

    # FIX: convertir a float64 antes de cualquier operación numpy
    df = safe_float_cols(df, feat_cluster)
    X_c  = df[feat_cluster].dropna()
    X_sc = StandardScaler().fit_transform(X_c.astype('float64'))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    tab1,tab2,tab3 = st.tabs(["🔵 K-Means","🌳 Jerárquico","👥 Perfiles"])

    with tab1:
        st.markdown("### 🔵 K-Means")
        st.markdown(ibox("📌 <b>Silhouette:</b> de -1 a 1. Cercano a 1 = clústeres bien separados."),
                    unsafe_allow_html=True)

        k_range = range(2,9)
        sil_scores = []
        prog = st.progress(0)
        for i,k in enumerate(k_range):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_sc)
            sil_scores.append(silhouette_score(X_sc, lbl))
            prog.progress((i+1)/len(k_range))
        prog.empty()

        best_k = list(k_range)[np.argmax(sil_scores)]
        st.session_state.best_k = best_k

        c1,c2 = st.columns([3,2])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode='lines+markers',
                                     marker=dict(size=10,color='#2d6a9f'),
                                     line=dict(width=3,color='#2d6a9f')))
            fig.add_vline(x=best_k, line_dash='dash', line_color='red',
                          annotation_text=f'Mejor k={best_k}', annotation_position='top right')
            fig.update_layout(title='Silhouette por k', xaxis_title='k',
                              yaxis_title='Score', plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sil_df = pd.DataFrame({'k':list(k_range),'Silhouette':[round(s,4) for s in sil_scores]})
            sil_df[''] = sil_df['k'].apply(lambda x: '⭐ Mejor' if x==best_k else '')
            st.dataframe(sil_df, use_container_width=True, hide_index=True)

        n_k = st.slider("Número de clústeres K-Means", 2, 8, best_k)
        km_final = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        cluster_labels = km_final.fit_predict(X_sc)

        # FIX: asignar preservando índice correcto
        df_cl = df.copy()
        df_cl.loc[X_c.index, 'cluster_kmeans'] = cluster_labels
        df_cl['cluster_kmeans'] = df_cl['cluster_kmeans'].fillna(0).astype(int)

        df_pca = pd.DataFrame(X_pca, columns=['PC1','PC2'])
        df_pca['Clúster'] = cluster_labels.astype(str)
        var_exp = pca.explained_variance_ratio_ * 100
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Clúster',
                         title=f'Clústeres K-Means (k={n_k}) — PC1={var_exp[0]:.1f}% · PC2={var_exp[1]:.1f}%',
                         color_discrete_sequence=PALETTE, opacity=0.7)
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.df_clustered = df_cl
        st.markdown(sbox(f"✅ Clústeres K-Means guardados (k={n_k})."), unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🌳 Clustering Jerárquico")
        n_h = st.slider("Clústeres jerárquicos", 2, 8, min(st.session_state.best_k,4))
        sample_n = min(300, len(X_sc))
        idx_s = np.random.choice(len(X_sc), sample_n, replace=False)
        Z = linkage(X_sc[idx_s], method='ward')
        fig, ax = plt.subplots(figsize=(13,5))
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=40,
                   leaf_rotation=90, leaf_font_size=9, show_contracted=True)
        if n_h > 1 and len(Z) >= n_h:
            ax.axhline(y=Z[-(n_h-1),2], color='red', linestyle='--', linewidth=1.5,
                       label=f'Corte k={n_h}')
            ax.legend()
        ax.set_title(f'Dendrograma (muestra {sample_n})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Muestra'); ax.set_ylabel('Distancia Ward')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        agg = AgglomerativeClustering(n_clusters=n_h, linkage='ward')
        hier_labels = agg.fit_predict(X_sc)
        df_h = df.copy()
        df_h.loc[X_c.index,'cluster_hier'] = hier_labels.astype(int)
        dist = pd.Series(hier_labels).value_counts().reset_index()
        dist.columns = ['Clúster','Cantidad']
        dist['Clúster'] = dist['Clúster'].astype(str)
        fig = px.bar(dist, x='Clúster', y='Cantidad', color='Clúster',
                     title='Clientes por Clúster Jerárquico',
                     color_discrete_sequence=PALETTE, text='Cantidad')
        fig.update_traces(textposition='auto')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 👥 Perfiles de Clientes")
        df_cl_prof = st.session_state.df_clustered
        if df_cl_prof is None:
            st.warning("Ejecuta K-Means primero."); st.stop()

        cl_cols = [c for c in feat_cluster if c in df_cl_prof.columns]
        # FIX: convertir a float antes del groupby
        df_cl_prof = safe_float_cols(df_cl_prof, cl_cols)

        summary = df_cl_prof.groupby('cluster_kmeans')[cl_cols].mean().round(2)
        summary['N_clientes'] = df_cl_prof.groupby('cluster_kmeans').size()
        if target in df_cl_prof.columns:
            try:
                tasa = df_cl_prof.groupby('cluster_kmeans')[target].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').mean() * 100
                ).round(1)
                summary['Tasa_resp(%)'] = tasa
            except Exception:
                pass

        scores = summary[cl_cols].mean(axis=1)
        rank = scores.rank()
        n_cl = len(rank)
        perfiles_map = {}
        for idx in rank.index:
            r = rank[idx]
            if   r >= n_cl*0.75: perfiles_map[idx] = ("🌟 Premium",   "Beneficios exclusivos y atención VIP.")
            elif r >= n_cl*0.50: perfiles_map[idx] = ("🔄 Frecuente", "Programas de fidelización y puntos.")
            elif r >= n_cl*0.25: perfiles_map[idx] = ("💰 Económico", "Descuentos y promociones de precio.")
            else:                perfiles_map[idx] = ("💤 Inactivo",  "Campañas de reactivación.")

        st.dataframe(summary.reset_index(), use_container_width=True)

        cols_cards = st.columns(min(n_cl,4))
        for i,(idx,row) in enumerate(summary.iterrows()):
            perfil, recomend = perfiles_map[idx]
            with cols_cards[i % len(cols_cards)]:
                tasa_txt = f" · Tasa: {row.get('Tasa_resp(%)','-')}%" if 'Tasa_resp(%)' in row.index else ""
                st.markdown(f"""<div class="section-card" style="min-height:150px">
                    <h4>Clúster {idx}</h4><b>{perfil}</b><br>
                    👥 {int(row['N_clientes'])} clientes{tasa_txt}<br>
                    <small>💡 {recomend}</small></div>""", unsafe_allow_html=True)

        # Heatmap
        if len(cl_cols) > 0:
            fig = px.imshow(summary[cl_cols].T, title="Heatmap por Clúster",
                            color_continuous_scale='RdBu_r', text_auto='.1f', aspect='auto')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        csv_cl = df_cl_prof.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar datos con clústeres",
                           data=csv_cl, file_name="clientes_segmentados.csv", mime="text/csv")

# =============================================================================
# MÓDULO 5: CLASIFICACIÓN
# =============================================================================
elif menu == "🤖 Clasificación":
    banner("🤖 Clasificación de Clientes",
           "Árbol · Random Forest · Gradient Boosting · XGBoost · Cross-Validation")

    if not st.session_state.partition_done:
        st.warning("⚠️ Primero ejecuta Partición y Baseline."); st.stop()

    X_train = st.session_state.X_train
    X_val   = st.session_state.X_val
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val   = st.session_state.y_val
    y_test  = st.session_state.y_test
    feat_names = st.session_state.feature_names_model
    is_bin  = st.session_state.is_binary
    n_cls   = st.session_state.n_classes

    tipo_txt = "Binario (2 clases)" if is_bin else f"Multiclase ({n_cls} clases)"
    st.markdown(ibox(f"📌 <b>Tipo de problema detectado: {tipo_txt}</b><br>"
                     f"Métricas usadas: {'binary' if is_bin else 'weighted'}. "
                     f"{'ROC AUC binario.' if is_bin else 'ROC AUC OVR weighted para multiclase.'}"),
                unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        max_depth_dt = st.slider("Profundidad máx. Árbol", 2, 15, 5)
        n_estimators = st.slider("N° árboles (RF/GBM)", 50, 500, 100, 50)
    with c2:
        usar_cv   = st.checkbox("✅ Cross-Validation 5-fold", value=True)
        cw_opt    = st.selectbox("Peso de clases", ['balanced','none'])
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Entrenar todos los modelos", type="primary", use_container_width=True)

    if run_btn:
        cw = 'balanced' if cw_opt == 'balanced' else None

        modelos = {
            '🌲 Árbol de Decisión': DecisionTreeClassifier(max_depth=max_depth_dt,
                                      random_state=42, class_weight=cw),
            '🌲🌲 Random Forest':   RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=42, class_weight=cw, n_jobs=-1),
            '⚡ Gradient Boosting': GradientBoostingClassifier(n_estimators=n_estimators,
                                      random_state=42),
            '📉 Reg. Logística':    LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight=cw),
        }
        if XGBOOST_AVAILABLE:
            modelos['🚀 XGBoost'] = XGBClassifier(
                n_estimators=n_estimators, random_state=42,
                eval_metric='mlogloss' if not is_bin else 'logloss', verbosity=0)

        interp_map = {
            '🌲 Árbol de Decisión':  'Reglas interpretables, fácil de explicar',
            '🌲🌲 Random Forest':    'Alta precisión, robusto al sobreajuste',
            '⚡ Gradient Boosting':  'Muy preciso, bueno con datos complejos',
            '📉 Reg. Logística':     'Modelo lineal, rápido e interpretable',
            '🚀 XGBoost':            'Excelente rendimiento, altamente optimizado',
        }

        prog = st.progress(0); status = st.empty()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        new_results = {k:v for k,v in st.session_state.results.items() if 'Baseline' in k}

        for i,(nombre,modelo) in enumerate(modelos.items()):
            status.text(f"⏳ Entrenando {nombre}...")
            try:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                y_prob = modelo.predict_proba(X_test)

                acc, prec, rec, f1, auc = calc_metrics(y_test, y_pred, y_prob)

                cv_mean, cv_std = None, None
                if usar_cv:
                    X_cv = np.vstack([X_train, X_val])
                    y_cv = np.concatenate([y_train, y_val])
                    scoring = 'roc_auc' if is_bin else 'roc_auc_ovr_weighted'
                    try:
                        cv_sc = cross_val_score(modelo, X_cv, y_cv, cv=skf,
                                                scoring=scoring, n_jobs=-1)
                        cv_mean = cv_sc.mean(); cv_std = cv_sc.std()
                    except Exception:
                        cv_mean, cv_std = None, None

                new_results[nombre] = {
                    'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                    'F1-Score': f1, 'AUC': auc, 'CV_mean': cv_mean, 'CV_std': cv_std,
                    'Interpretación': interp_map.get(nombre,''),
                    'model': modelo, 'y_pred': y_pred, 'y_prob': y_prob
                }

                if 'Random Forest' in nombre:
                    st.session_state.rf_importances = modelo.feature_importances_
                    st.session_state.rf_names = feat_names
                if 'Árbol' in nombre:
                    st.session_state.dt_model = modelo
                    st.session_state.dt_names = feat_names

            except Exception as e:
                st.warning(f"⚠️ Error entrenando {nombre}: {e}")
                new_results[nombre] = {
                    'Accuracy':0,'Precision':0,'Recall':0,'F1-Score':0,
                    'AUC':None,'CV_mean':None,'CV_std':None,
                    'Interpretación': f'Error: {e}',
                    'y_pred': np.zeros(len(y_test)), 'y_prob': None
                }

            prog.progress((i+1)/len(modelos))

        st.session_state.results = new_results
        st.session_state.models_trained = True
        status.empty(); prog.empty()
        st.markdown(sbox(f"✅ {len(modelos)} modelos entrenados."), unsafe_allow_html=True)

    # Importancia RF
    if st.session_state.rf_importances is not None and st.session_state.rf_names:
        st.markdown("---")
        st.markdown("### 📊 Importancia de Variables — Random Forest")
        imp_df = pd.DataFrame({
            'Variable': st.session_state.rf_names,
            'Importancia': st.session_state.rf_importances
        }).sort_values('Importancia', ascending=False)
        fig = px.bar(imp_df, x='Importancia', y='Variable', orientation='h',
                     title='Feature Importance', color='Importancia',
                     color_continuous_scale='Blues', text=imp_df['Importancia'].round(3))
        fig.update_traces(textposition='outside')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Árbol visual
    if st.session_state.dt_model is not None:
        st.markdown("---")
        st.markdown("### 🌲 Árbol de Decisión Visual")
        dt = st.session_state.dt_model
        names_dt = st.session_state.dt_names or [f'f{i}' for i in range(dt.n_features_in_)]
        class_names_dt = [str(c) for c in st.session_state.class_names] if st.session_state.class_names else None
        max_dv = st.slider("Niveles visibles del árbol", 2, min(5, dt.get_depth()), 3)
        fig, ax = plt.subplots(figsize=(20, 8))
        sk_tree.plot_tree(dt, max_depth=max_dv, feature_names=names_dt,
                          class_names=class_names_dt, filled=True, rounded=True,
                          fontsize=8, ax=ax)
        ax.set_title(f'Árbol de Decisión (profundidad total={dt.get_depth()}, mostrando ≤{max_dv} niveles)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        with st.expander("📄 Reglas en texto"):
            st.code(export_text(dt, feature_names=names_dt, max_depth=4), language='text')

# =============================================================================
# MÓDULO 6: EVALUACIÓN
# =============================================================================
elif menu == "📈 Evaluación de Modelos":
    banner("📈 Evaluación de Modelos",
           "Matriz de Confusión · Curva ROC · Métricas detalladas")

    results  = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}

    if not modelos_e:
        st.warning("⚠️ Primero entrena los modelos."); st.stop()

    y_test  = st.session_state.y_test
    is_bin  = st.session_state.is_binary
    n_cls   = st.session_state.n_classes
    le_y    = st.session_state.le_target

    modelo_sel = st.selectbox("Selecciona modelo", list(modelos_e.keys()))
    res = modelos_e[modelo_sel]
    y_pred = res['y_pred']
    y_prob = res['y_prob']

    # Métricas
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5],[
        ("Accuracy",  f"{res['Accuracy']:.3f}"),
        ("Precision", f"{res['Precision']:.3f}"),
        ("Recall",    f"{res['Recall']:.3f}"),
        ("F1-Score",  f"{res['F1-Score']:.3f}"),
        ("AUC",       f"{res['AUC']:.3f}" if res['AUC'] else "N/A")]):
        with col: st.markdown(mcard(val,lbl), unsafe_allow_html=True)

    if res.get('CV_mean') is not None:
        st.markdown(sbox(f"📊 <b>Cross-Validation 5-fold AUC:</b> "
                         f"{res['CV_mean']:.4f} ± {res['CV_std']:.4f}"), unsafe_allow_html=True)

    st.markdown("---")
    c1,c2 = st.columns(2)

    with c1:
        st.markdown("#### 🔲 Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        labels = st.session_state.class_names or [str(i) for i in range(n_cls)]
        fig, ax = plt.subplots(figsize=(max(5, n_cls+2), max(4, n_cls+1)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=1, linecolor='white',
                    annot_kws={"size": max(8, 16-n_cls)})
        ax.set_xlabel('Predicho', fontweight='bold')
        ax.set_ylabel('Real', fontweight='bold')
        ax.set_title(f'Matriz de Confusión\n{modelo_sel}', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if is_bin and cm.size == 4:
            tn,fp,fn,tp = cm.ravel()
            mc1,mc2 = st.columns(2)
            mc1.metric("VP (Verdaderos Pos.)", int(tp))
            mc2.metric("VN (Verdaderos Neg.)", int(tn))
            mc1.metric("FP (Falsos Pos.)", int(fp))
            mc2.metric("FN (Falsos Neg.)", int(fn))

    with c2:
        st.markdown("#### 📉 Curva ROC")
        if is_bin and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
            auc_val = res['AUC'] or 0
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy',
                                     fillcolor='rgba(45,106,159,0.15)',
                                     name=f'{modelo_sel} (AUC={auc_val:.3f})',
                                     line=dict(width=3,color='#2d6a9f')))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',
                                     line=dict(dash='dash',color='gray')))
            fig.update_layout(title=f'ROC — {modelo_sel}',
                              xaxis_title='FPR',yaxis_title='TPR',
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        elif not is_bin and y_prob is not None:
            # ROC por clase (OVR)
            from sklearn.preprocessing import label_binarize
            labels_bin = st.session_state.class_names or [str(i) for i in range(n_cls)]
            y_bin = label_binarize(y_test, classes=list(range(n_cls)))
            fig = go.Figure()
            for j in range(n_cls):
                try:
                    fpr_j, tpr_j, _ = roc_curve(y_bin[:,j], y_prob[:,j])
                    auc_j = roc_auc_score(y_bin[:,j], y_prob[:,j])
                    fig.add_trace(go.Scatter(x=fpr_j, y=tpr_j, mode='lines',
                                             name=f'Clase {labels_bin[j]} (AUC={auc_j:.3f})',
                                             line=dict(width=2,color=PALETTE[j%len(PALETTE)])))
                except Exception:
                    pass
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Aleatorio',
                                     line=dict(dash='dash',color='gray')))
            fig.update_layout(title=f'ROC OVR — {modelo_sel}',
                              xaxis_title='FPR',yaxis_title='TPR',
                              plot_bgcolor='white',paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Curva ROC no disponible para este modelo.")

    # Reporte
    with st.expander("📋 Reporte completo de clasificación"):
        labels_rep = st.session_state.class_names or None
        st.code(classification_report(y_test, y_pred, target_names=labels_rep), language='text')

# =============================================================================
# MÓDULO 7: COMPARATIVA
# =============================================================================
elif menu == "📋 Comparativa de Modelos":
    banner("📋 Comparativa de Modelos", "Tabla · Barras · Radar · Mejor modelo automático")

    results = st.session_state.results
    if not results:
        st.warning("⚠️ Primero ejecuta Baseline y Clasificación."); st.stop()

    filas = []
    for nombre, res in results.items():
        filas.append({
            'Modelo': nombre,
            'Accuracy':  round(res['Accuracy'],4),
            'Precision': round(res['Precision'],4),
            'Recall':    round(res['Recall'],4),
            'F1-Score':  round(res['F1-Score'],4),
            'AUC':       round(res['AUC'],4) if res['AUC'] else 0,
            'CV AUC':    f"{res['CV_mean']:.4f} ± {res['CV_std']:.4f}" if res.get('CV_mean') else '—',
            'Interpretación': res.get('Interpretación',''),
        })

    df_comp = pd.DataFrame(filas).sort_values('AUC', ascending=False).reset_index(drop=True)
    mejor = df_comp.iloc[0]

    st.markdown(sbox(f"🏆 <b>Mejor modelo: {mejor['Modelo']}</b> — "
                     f"AUC={mejor['AUC']} · F1={mejor['F1-Score']} · Acc={mejor['Accuracy']} "
                     f"{'· CV=' + mejor['CV AUC'] if mejor['CV AUC'] != '—' else ''}"),
                unsafe_allow_html=True)

    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    metrics_plot = ['Accuracy','Precision','Recall','F1-Score','AUC']
    fig = go.Figure()
    for i,m in enumerate(metrics_plot):
        fig.add_trace(go.Bar(name=m, x=df_comp['Modelo'], y=df_comp[m],
                             marker_color=PALETTE[i]))
    fig.update_layout(barmode='group', title='Comparativa de Métricas',
                      plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation='h',yanchor='bottom',y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Radar
    st.markdown("#### 🕸️ Radar Chart")
    cats = ['Accuracy','Precision','Recall','F1-Score','AUC']
    fig = go.Figure()
    for i,(_,row) in enumerate(df_comp.iterrows()):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill='toself',
            name=row['Modelo'], opacity=0.5,
            line_color=PALETTE[i%len(PALETTE)],
            fillcolor=PALETTE[i%len(PALETTE)]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])))
    st.plotly_chart(fig, use_container_width=True)

    csv_comp = df_comp.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar tabla CSV",
                       data=csv_comp, file_name="comparativa_modelos.csv", mime="text/csv")

# =============================================================================
# MÓDULO 8: PREDICCIÓN INDIVIDUAL
# =============================================================================
elif menu == "🔮 Predicción Individual":
    banner("🔮 Predicción Individual",
           "Ingresa datos de un cliente nuevo y obtén la probabilidad de respuesta")

    if not st.session_state.models_trained:
        st.warning("⚠️ Primero entrena los modelos."); st.stop()

    results   = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}
    scaler    = st.session_state.scaler
    feat_names = st.session_state.feature_names_model
    is_bin    = st.session_state.is_binary

    modelo_pred = st.selectbox("Modelo para predicción", list(modelos_e.keys()))

    if not feat_names:
        st.error("❌ Sin features. Re-ejecuta Partición."); st.stop()

    st.markdown("### 📝 Datos del Cliente")
    df_ref = st.session_state.df_proc
    valores = {}
    cols_f = st.columns(min(3, len(feat_names)))
    for i,feat in enumerate(feat_names):
        col = cols_f[i % len(cols_f)]
        with col:
            if df_ref is not None and feat in df_ref.columns:
                col_data = pd.to_numeric(df_ref[feat], errors='coerce')
                mn  = float(col_data.min())
                mx  = float(col_data.max())
                med = float(col_data.median())
                valores[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                 value=med, key=f"pi_{feat}")
            else:
                valores[feat] = st.number_input(feat, value=0.0, key=f"pi_{feat}")

    if st.button("🔮 Predecir", type="primary", use_container_width=True):
        try:
            X_nuevo = np.array([[valores[f] for f in feat_names]], dtype='float64')
            X_sc    = scaler.transform(X_nuevo)
            modelo_obj = modelos_e[modelo_pred]['model']
            y_prob_n   = modelo_obj.predict_proba(X_sc)[0]
            y_pred_n   = modelo_obj.predict(X_sc)[0]

            le_y  = st.session_state.le_target
            clase_pred = le_y.inverse_transform([y_pred_n])[0] if le_y else y_pred_n

            st.markdown("---")
            st.markdown("### 🎯 Resultado")

            if is_bin:
                prob_pos = float(y_prob_n[1])
                color = "green" if prob_pos > 0.5 else "blue"
                resp  = "✅ SÍ RESPONDERÁ" if prob_pos > 0.5 else "❌ NO RESPONDERÁ"
                c1,c2,c3 = st.columns(3)
                with c1: st.markdown(mcard(resp, "Predicción", color), unsafe_allow_html=True)
                with c2: st.markdown(mcard(f"{prob_pos:.1%}", "Probabilidad positiva",
                                           "green" if prob_pos>0.5 else "blue"), unsafe_allow_html=True)
                confianza = "Alta" if abs(prob_pos-0.5)>0.3 else "Media" if abs(prob_pos-0.5)>0.15 else "Baja"
                with c3: st.markdown(mcard(confianza,"Confianza"), unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=prob_pos*100,
                    title={'text':"Probabilidad de Respuesta Positiva (%)"},
                    delta={'reference':50},
                    gauge={'axis':{'range':[0,100]},
                           'bar':{'color':'#27ae60' if prob_pos>0.5 else '#e74c3c'},
                           'steps':[{'range':[0,30],'color':'#f8d7da'},
                                    {'range':[30,70],'color':'#fff3cd'},
                                    {'range':[70,100],'color':'#d4edda'}],
                           'threshold':{'line':{'color':'black','width':4},'value':50}}
                ))
                fig.update_layout(height=300, paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

                if prob_pos >= 0.70:
                    st.markdown(sbox("🌟 <b>Prioridad ALTA</b> — Contactar con oferta premium."),
                                unsafe_allow_html=True)
                elif prob_pos >= 0.40:
                    st.markdown(wbox("🔄 <b>Prioridad MEDIA</b> — Incluir en campaña estándar."),
                                unsafe_allow_html=True)
                else:
                    st.markdown(ibox("💤 <b>Prioridad BAJA</b> — Solo si el costo de contacto es bajo."),
                                unsafe_allow_html=True)
            else:
                # Multiclase: mostrar probabilidades por clase
                class_names = st.session_state.class_names
                st.markdown(f"**Clase predicha:** `{clase_pred}`")
                prob_df = pd.DataFrame({'Clase': class_names or [str(i) for i in range(len(y_prob_n))],
                                        'Probabilidad': y_prob_n})
                fig = px.bar(prob_df, x='Clase', y='Probabilidad',
                             title='Probabilidades por Clase',
                             color='Probabilidad', color_continuous_scale='Blues',
                             text=prob_df['Probabilidad'].apply(lambda x: f"{x:.1%}"))
                fig.update_traces(textposition='outside')
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Datos ingresados"):
                st.dataframe(pd.DataFrame([valores]), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error en predicción: {e}")
            import traceback; st.code(traceback.format_exc())

# =============================================================================
# MÓDULO 9: INTERPRETACIÓN GERENCIAL
# =============================================================================
elif menu == "💼 Interpretación Gerencial":
    banner("💼 Informe Gerencial",
           "Resultados ejecutivos · Recomendaciones · Limitaciones")

    results   = st.session_state.results
    modelos_e = {k:v for k,v in results.items() if 'model' in v}

    if not modelos_e:
        st.warning("⚠️ Primero entrena los modelos."); st.stop()

    mejor_nombre = max(modelos_e, key=lambda k: modelos_e[k]['AUC'] or 0)
    mejor_res    = modelos_e[mejor_nombre]

    c1,c2,c3,c4 = st.columns(4)
    auc_val = mejor_res['AUC'] or 0
    with c1: st.markdown(mcard(f"{auc_val:.1%}", f"AUC — {mejor_nombre}", "green"), unsafe_allow_html=True)
    with c2: st.markdown(mcard(f"{mejor_res['F1-Score']:.1%}", "F1-Score"), unsafe_allow_html=True)
    with c3: st.markdown(mcard(f"{mejor_res['Accuracy']:.1%}", "Exactitud"), unsafe_allow_html=True)
    with c4:
        if mejor_res.get('CV_mean'):
            st.markdown(mcard(f"{mejor_res['CV_mean']:.1%}", "CV AUC (5-fold)", "green"),
                        unsafe_allow_html=True)
        else:
            st.markdown(mcard(f"{mejor_res['Precision']:.1%}", "Precisión"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div class="section-card">
        <h3>📋 Resumen Ejecutivo</h3>
        <p>El modelo <b>{mejor_nombre}</b> obtuvo el mejor rendimiento con AUC de
        <b>{auc_val:.1%}</b> y exactitud de <b>{mejor_res['Accuracy']:.1%}</b>.
        {"Validado con cross-validation 5-fold (CV AUC = " + str(round(mejor_res['CV_mean'],4)) + "), lo que confirma que generaliza bien a datos nuevos." if mejor_res.get('CV_mean') else ""}
        </p></div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="section-card"><h3>🎯 Clientes a Priorizar</h3><ul>
        <li>🌟 <b>Premium:</b> máxima prioridad de campaña.</li>
        <li>🔄 <b>Frecuentes:</b> fidelización y recompensas.</li>
        <li>💤 <b>Inactivos:</b> reactivación con incentivos.</li>
        <li>💰 <b>Económicos:</b> descuentos y promociones.</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="section-card"><h3>⚡ Acciones Recomendadas</h3><ol>
        <li>Filtrar base con el modelo <b>{mejor_nombre}</b> antes de la campaña.</li>
        <li>Personalizar mensajes según clúster de cliente.</li>
        <li>Monitorear conversión real vs predicha.</li>
        <li>Reentrenar cada 3-6 meses con datos nuevos.</li>
        <li>Piloto A/B antes de escalar.</li>
        </ol></div>""", unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown("""<div class="section-card"><h3>⚠️ Limitaciones</h3><ul>
        <li>📌 Basado en patrones históricos; el comportamiento puede cambiar.</li>
        <li>📌 No considera factores externos (economía, competencia).</li>
        <li>📌 Requiere datos de calidad para predicciones confiables.</li>
        <li>📌 Validar con piloto antes de escalar.</li>
        </ul></div>""", unsafe_allow_html=True)
    with c4:
        if 'Baseline (Dummy)' in results:
            base_auc = results['Baseline (Dummy)']['AUC'] or 0
            mejora   = auc_val - base_auc
            st.markdown(f"""<div class="section-card"><h3>📊 Impacto vs Baseline</h3>
            <p>El modelo <b>{mejor_nombre}</b> supera al baseline aleatorio en
            <b>+{mejora:.1%} AUC</b>.<br><br>
            Esto permite enfocar el presupuesto de campaña en clientes con mayor
            probabilidad de conversión, reduciendo costos y mejorando el ROI.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(sbox(f"💼 <b>Conclusión:</b> El modelo <b>{mejor_nombre}</b> está listo para uso operativo "
                     f"con AUC de <b>{auc_val:.1%}</b>. Permite anticipar qué clientes responderán a la campaña "
                     f"y enfocar los recursos comerciales de forma eficiente."), unsafe_allow_html=True)
