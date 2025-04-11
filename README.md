# Proyecto de Aprendizaje Automático - TP2

## Descripción

Este proyecto implementa varios algoritmos de aprendizaje automático desde cero utilizando NumPy, para resolver dos problemas de clasificación:

1. **Diagnóstico de Cáncer de Mama**: Clasificación binaria para determinar si una célula presenta características compatibles con cáncer, basándose en mediciones morfológicas y bioquímicas.

2. **Predicción de Rendimiento de Jugadores de Baloncesto**: Clasificación multiclase para predecir la métrica WAR (Wins Above Replacement) de jugadores de baloncesto, categorizados en: Negative WAR, Null WAR y Positive WAR.

Este proyecto ha sido desarrollado como parte del Trabajo Práctico 2 para la materia de Aprendizaje Automático.

## Características

- Implementación desde cero (utilizando exclusivamente NumPy) de:
  - Regresión Logística con regularización L2
  - Análisis Discriminante Lineal (LDA)
  - Bosque Aleatorio (Random Forest)
  - Técnicas de rebalanceo: SMOTE, Undersampling, Oversampling
  - Métricas de evaluación: Accuracy, Precision, Recall, F1, ROC, AUC, PR-Curve

- Módulos para preprocesamiento de datos:
  - Normalización y estandarización
  - Manejo de valores faltantes
  - Codificación de variables categóricas
  - Detección y tratamiento de outliers

- Sistema completo de evaluación y visualización:
  - Matrices de confusión
  - Curvas ROC y PR
  - Comparativas visuales de modelos
  - Validación cruzada

## Estructura del Proyecto

```
.
├── data/                          # Datos de entrada
│   ├── p1/                        # Datos para el problema 1 (Cáncer de Mama)
│   └── p2/                        # Datos para el problema 2 (Rendimiento Baloncesto)
├── src/                           # Código fuente
│   ├── config/                    # Configuraciones
│   │   ├── __init__.py
│   │   └── plot_config.py         # Configuraciones para gráficos
│   ├── evaluation/                # Módulos de evaluación
│   │   ├── cross_validation.py    # Implementación de validación cruzada
│   │   └── metrics.py             # Implementación de métricas de evaluación
│   ├── models/                    # Implementación de modelos
│   │   ├── base.py                # Clase base para todos los modelos
│   │   ├── lda.py                 # Análisis Discriminante Lineal
│   │   ├── logistic_regression.py # Regresión Logística
│   │   └── random_forest.py       # Bosque Aleatorio
│   ├── preprocessing/             # Módulos de preprocesamiento
│   │   ├── categorical_encoder.py # Codificación de variables categóricas
│   │   ├── data_loader.py         # Cargador de datos
│   │   ├── imputation.py          # Manejo de valores faltantes
│   │   ├── outliers.py            # Detección y tratamiento de outliers
│   │   └── rebalancing.py         # Técnicas de rebalanceo de clases
│   ├── utils/                     # Utilidades
│   │   ├── utils.py               # Funciones auxiliares generales
│   │   └── visuals.py             # Funciones para visualización
│   └── Lebrero_Juan_Francisco_TP2.ipynb  # Notebook principal con el desarrollo
├── figures/                       # Gráficos y visualizaciones generadas
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## Requisitos

- Python 3.8+
- Dependencias:
  - numpy
  - pandas
  - matplotlib
  - seaborn

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/yourusername/tp2-machine-learning.git
   cd tp2-machine-learning
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

### Ejecutar el Notebook
El proyecto se puede ejecutar y explorar mediante el notebook principal:

```bash
jupyter notebook src/Lebrero_Juan_Francisco_TP2.ipynb
```

### Usar los modelos implementados

Los modelos están diseñados para usarse de forma similar a scikit-learn:

```python
# Ejemplo de uso de la Regresión Logística
from src.models.logistic_regression import LogisticRegression, LogisticRegressionConfig

# Configurar el modelo
config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    lambda_reg=0.1,  # Regularización L2
    random_state=42
)

# Crear y entrenar el modelo
model = LogisticRegression(config)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)
y_prob = model.predict_prob(X_test)
```

### Evaluación de modelos

```python
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.utils import evaluate_model

# Evaluación manual
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# O usando la función de evaluación completa
metrics = evaluate_model(model, X_test, y_test, show_plots=True)
```

## Modelos Implementados

### 1. Regresión Logística
- Implementación completa con regularización L2
- Soporte para problemas binarios y multiclase
- Optimización mediante descenso de gradiente
- Configuración de hiperparámetros como tasa de aprendizaje y fuerza de regularización

### 2. Análisis Discriminante Lineal (LDA)
- Implementación para problemas multiclase
- Estimación de matrices de covarianza y probabilidades a priori
- Proyección a espacio discriminante de menor dimensión

### 3. Bosque Aleatorio (Random Forest)
- Implementación desde cero de árboles de decisión y ensemble
- Criterio de división basado en entropía
- Parámetros configurables: número de árboles, profundidad máxima, número de características a considerar

## Técnicas de Rebalanceo

Se implementaron las siguientes técnicas para manejar conjuntos de datos desbalanceados:

1. **Undersampling**: Reducción aleatoria de la clase mayoritaria
2. **Oversampling**: Duplicación aleatoria de la clase minoritaria
3. **SMOTE**: Generación sintética de ejemplos de la clase minoritaria
4. **Cost Reweighting**: Ajuste de pesos en la función de pérdida


## Licencia

Este proyecto está licenciado bajo la licencia MIT - ver el archivo LICENSE para más detalles.

## Contacto

Juan Francisco Lebrero - [GitHub](https://github.com/frizynn)