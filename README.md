# Trabajo Práctico 2 - Aprendizaje Automático

## Descripción General

Este proyecto aborda dos problemas de clasificación utilizando implementaciones propias (exclusivamente con NumPy) de algoritmos de Machine Learning. El objetivo es resolver tareas reales de diagnóstico biomédico y predicción deportiva, siguiendo las mejores prácticas de ingeniería de software y ciencia de datos.

- **Problema 1:** Diagnóstico de Cáncer de Mama (Clasificación Binaria)
- **Problema 2:** Predicción de Rendimiento de Jugadores de Basketball (Clasificación Multiclase)

El código es modular, escalable y está organizado para facilitar la reutilización, el análisis y la comparación de modelos.

---

## Estructura del Proyecto

```
.
├── data/
│   ├── p1/                  # Datos y descripciones para Problema 1
│   │   ├── raw/             # Datos originales (csv)
│   │   └── cell_diagnosis_description.md
│   └── p2/                  # Datos y descripciones para Problema 2
│       ├── raw/             # Datos originales (csv)
│       └── WAR_class.md
├── src/
│   ├── models/              # Implementaciones de modelos ML (NumPy puro)
│   ├── preprocessing/       # Preprocesamiento, rebalanceo, outliers, etc.
│   ├── evaluation/          # Métricas y validación cruzada
│   ├── utils/               # Utilidades y visualizaciones
│   └── Lebrero_Juan_Francisco_TP2.ipynb  # Notebook principal
├── figures/                 # Gráficos y visualizaciones generadas
├── requirements.txt         # Dependencias
├── README.md                # Este archivo
└── ...
```

---

## Datasets y Variables

### Problema 1: Diagnóstico de Cáncer de Mama
- **Archivos:**
  - `cell_diagnosis_balanced_dev.csv`, `cell_diagnosis_balanced_test.csv`
  - `cell_diagnosis_imbalanced_dev.csv`, `cell_diagnosis_imbalanced_test.csv`
- **Descripción de variables:** (ver `data/p1/cell_diagnosis_description.md`)
  - **Numéricas:** CellSize, CellShape, NucleusDensity, ChromatinTexture, CytoplasmSize, CellAdhesion, MitosisRate, NuclearMembrane, GrowthFactor, OxygenSaturation, Vascularization, InflammationMarkers
  - **Categóricas:** CellType (`Epithelial`, `Mesenchymal`, `Unknown`), GeneticMutation (`Present`, `Absent`, `Unknown`)
  - **Objetivo:** Diagnosis (0: normal, 1: anómala)

### Problema 2: Rendimiento de Jugadores de Basketball
- **Archivos:**
  - `WAR_class_dev.csv`, `WAR_class_test.csv`
- **Descripción de variables:** (ver `data/p2/WAR_class.md`)
  - poss: Posesiones
  - mp: Minutos jugados
  - off_def: Impacto ofensivo/defensivo
  - pace_impact: Impacto en el ritmo de juego
  - **Objetivo:** WAR_class (1: Negative, 2: Null, 3: Positive)

---

## Implementaciones y Funcionalidades

- **Modelos desde cero (NumPy):**
  - Regresión Logística (binaria y multiclase, regularización L2)
  - Análisis Discriminante Lineal (LDA)
  - Bosque Aleatorio (Random Forest, entropía)
- **Preprocesamiento:**
  - Normalización/estandarización
  - Imputación de valores faltantes
  - Codificación de variables categóricas
  - Detección y tratamiento de outliers
  - Técnicas de rebalanceo: Undersampling, Oversampling, SMOTE, Cost Reweighting
- **Evaluación y Visualización:**
  - Métricas: Accuracy, Precision, Recall, F1, Matriz de confusión, Curvas ROC y PR, AUC-ROC, AUC-PR
  - Validación cruzada y búsqueda de hiperparámetros
  - Gráficos comparativos y reportes automáticos

---

## Instalación y Requisitos

- **Python 3.8+**
- **Dependencias:**
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - jupyter (opcional para notebooks)

Instalación recomendada:

```bash
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Ejecución y Uso

### 1. Notebook principal

Ejecuta y explora el flujo completo desde el notebook:

```bash
jupyter notebook src/Lebrero_Juan_Francisco_TP2.ipynb
```

### 2. Uso de los modelos (ejemplo)

```python
from src.models.logistic_regression import LogisticRegression, LogisticRegressionConfig

config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    lambda_reg=0.1,
    random_state=42
)
model = LogisticRegression(config)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_prob(X_test)
```

### 3. Evaluación y métricas

```python
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.utils import evaluate_model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = evaluate_model(model, X_test, y_test, show_plots=True)
```

---

## Créditos y Licencia

- Autor: Juan Francisco Lebrero
- Licencia: MIT (ver LICENSE)

---

## Referencias
- Enunciado y datasets provistos por la cátedra de Aprendizaje Automático, UdeSA.
- Ver `Consigna_TP2_ML.pdf` para detalles completos del trabajo práctico.