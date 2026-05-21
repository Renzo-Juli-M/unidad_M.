# 🎯 Sistema de Segmentación y Predicción de Clientes v2.0

## ¿Qué hay de nuevo en v2.0?
- ✅ Selector dinámico de variable objetivo (funciona con cualquier CSV)
- ✅ Pipeline sin data leakage garantizado
- ✅ Cross-Validation 5-fold estratificada
- ✅ SMOTE para datasets desbalanceados
- ✅ Visualización gráfica del Árbol de Decisión
- ✅ Módulo de Predicción Individual (cliente nuevo)
- ✅ Clúster K-Means como feature adicional
- ✅ Detección automática de desbalance de clases

## Estructura
```
sistema_mineria_clientes/
├── app.py                      ← Aplicación principal v2.0
├── requirements.txt            ← Dependencias
├── datos_clientes_ejemplo.csv  ← Dataset de prueba (100 clientes)
└── README.md                   ← Este archivo
```

## Módulos
| # | Módulo | Descripción |
|---|--------|-------------|
| 1 | 🏠 Inicio | Presentación y novedades v2.0 |
| 2 | 📁 Carga de Datos | Upload CSV + selector de variable objetivo |
| 3 | ⚙️ Preprocesamiento | Nulos, codificación, correlación |
| 4 | 📊 Partición y Baseline | Split 70/15/15 + SMOTE + DummyClassifier |
| 5 | 🔵 Segmentación | K-Means + Jerárquico + Perfiles |
| 6 | 🤖 Clasificación | 5 modelos + Cross-Validation + árbol visual |
| 7 | 📈 Evaluación | Matriz de confusión + Curva ROC |
| 8 | 📋 Comparativa | Tabla + Radar Chart |
| 9 | 🔮 Predicción Individual | Formulario dinámico + Gauge |
| 10 | 💼 Gerencial | Informe ejecutivo + Recomendaciones |

---

## Instalación — Visual Studio Code

```bash
# 1. Entrar a la carpeta del proyecto
cd sistema_mineria_clientes

# 2. Crear entorno virtual
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
streamlit run app.py
```
Se abre en: http://localhost:8501

---

## Instalación — Google Colab

```python
# Celda 1: instalar
!pip install streamlit pyngrok imbalanced-learn xgboost scikit-learn plotly seaborn -q

# Celda 2: subir el archivo
from google.colab import files
files.upload()   # selecciona app.py

# Celda 3: lanzar
from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen([
    'streamlit', 'run', 'app.py',
    '--server.port', '8501',
    '--server.headless', 'true',
    '--server.enableCORS', 'false'
])
time.sleep(4)
url = ngrok.connect(8501)
print("🌐 Accede aquí:", url)
```

---

## Columna objetivo
El CSV puede tener cualquier nombre de columna objetivo. Al cargar el archivo, el sistema te muestra un **selector** para elegirla. La columna debe ser binaria (0/1 o dos categorías).

## Autor
Universidad Peruana Unión · Minería de Datos · 2025
