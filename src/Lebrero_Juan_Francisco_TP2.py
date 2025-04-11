#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


# In[2]:


from config import plot_config
from preprocessing.data_loader import DataLoader, DatasetConfig, SplitConfig
from preprocessing.outliers import replace_outliers_iqr
from utils.utils import (print_numerical_features_range, evaluate_model, evaluate_all_models,
                         analyze_null_values, impute_missing_values, train_valid_split, remove_negative_values,
                         apply_feature_engineering, save_processed_data, calculate_class_weights, normalize_data)

from utils.visuals import (analyze_categorical_variables, plot_numerical_distributions, plot_correlation_heatmap, 
                           plot_outliers_analysis, plot_lambda_tuning, plot_roc_curve, plot_precision_recall_curve)
from models.logistic_regression import LogisticRegression, LogisticRegressionConfig
from evaluation.cross_validation import cross_validate_lambda
from evaluation.metrics import f1_score, recall_score
from evaluation.cross_validation import stratified_cross_validate_lambda
from models.logistic_regression import LogisticRegression, LogisticRegressionConfig
from models.logistic_regression_with_cost import CostSensitiveLogisticRegression, CostSensitiveLogisticRegressionConfig
from preprocessing.data_loader import DataLoader, DatasetConfig, SplitConfig
from preprocessing.rebalancing import (
    RandomUnderSampler, 
    RandomOverSampler, 
    SMOTE, 
    SMOTEConfig,
    RebalancingConfig
)
from utils.utils import evaluate_model
from utils.visuals import plot_roc_curve, plot_precision_recall_curve


# In[3]:


def process_data(train_df, valid_df, target_column="Diagnosis", encode_categorical=True):
    # Resetear índices
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Separar características y variable objetivo
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column].to_numpy()

    X_val = valid_df.drop(columns=[target_column])
    y_val = valid_df[target_column].to_numpy()

    # Codificar variables categóricas
    if encode_categorical:
        X_train_encoded = DataLoader.encode_categorical(X_train).to_numpy()
        X_val_encoded = DataLoader.encode_categorical(X_val).to_numpy()
    else:
        X_train_encoded = X_train.to_numpy()
        X_val_encoded = X_val.to_numpy()

    return X_train_encoded, y_train, X_val_encoded, y_val


# In[4]:


def plot_data_analysis_all_visualizations(df, numerical_cols, target_column, features_to_plot, output_dir, fig_output_dir_p1, fig_params={}):

    heatmap_params = fig_params.get("heatmap_params", {
        "label_fontsize": 18,
        "title_fontsize": 18,
        "tick_fontsize": 18,
        "cbar_fontsize": 18,
        "annot_fontsize": 18,
        "figsize": (18, 12),
        "filename": "correlation_heatmap_numerical_features_outliers.png"
    })
    dist_params = fig_params.get("dist_params", {
        "filename": "numerical_distributions_outliers.png",
        "features_to_plot":features_to_plot,

        "figsize": (16, 5)
    })
    outlier_params = fig_params.get("outlier_params", {
        "filename": "boxplots_outliers_analysis.png",
        "features_to_plot":features_to_plot,
        "figsize": (16, 5)
    })

    # Plot numerical distributions
    fig1 = plot_numerical_distributions(
        df,
        numerical_cols,
        target_column,
        output_dir=output_dir,
        **dist_params
    )
    plt.show()

    # Plot correlation heatmap
    fig2 = plot_correlation_heatmap(
        df,
        numerical_cols + [target_column],
        output_dir=fig_output_dir_p1,
        **heatmap_params
    )
    plt.show()

    # Plot outliers analysis
    fig3 = plot_outliers_analysis(
        df=df,
        save_dir=fig_output_dir_p1,
        **outlier_params
    )
    return fig1, fig2, fig3


# %% Configuración de Directorios y Variables Globales

# In[5]:


fig_output_dir_p1 = "./figures/p1"
os.makedirs(fig_output_dir_p1, exist_ok=True)
current_dir = Path.cwd()
data_dir = current_dir.parent / "data"
data_dir = data_dir.resolve()
data_dir_p1 = data_dir / "p1"


# # Punto 1.1: Inicialización del Dataset

# In[6]:


config_cell_diagnosis = DatasetConfig(
    data_dir=data_dir_p1,
    target_column="Diagnosis",
    split_config=SplitConfig(test_size=0.2, shuffle=True, random_state=42)
)
loader_cell_diagnosis = DataLoader(config_cell_diagnosis)
loader_cell_diagnosis.read_data(
    dev_file="raw/cell_diagnosis_balanced_dev.csv",
    test_file="raw/cell_diagnosis_balanced_test.csv",
    splitted=False
)
cell_diagnosis_balanced_dev_outliers, cell_diagnosis_balanced_test_outliers = loader_cell_diagnosis.get_pandas_data(splitted=False)


# %% Análisis de Valores Nulos y División Train/Valid

# In[7]:


cell_diagnosis_balanced_for_train_outliers, cell_diagnosis_balanced_for_valid_outliers = train_valid_split(
    cell_diagnosis_balanced_dev_outliers, test_size=0.2, random_state=12
)
analyze_null_values(
    [cell_diagnosis_balanced_for_train_outliers, cell_diagnosis_balanced_test_outliers],
    ["conjunto de train", "conjunto de prueba"]
);


# %% Imputación de Valores Faltantes

# In[8]:


cell_diagnosis_balanced_for_train_outliers, cell_diagnosis_balanced_for_valid_outliers, cell_diagnosis_balanced_test_outliers = impute_missing_values(
    cell_diagnosis_balanced_for_train_outliers,
    cell_diagnosis_balanced_for_valid_outliers,
    cell_diagnosis_balanced_test_outliers,
    knn_neighbors=8,
    knn_weights="distance"
)


# %% Actualización y Guardado de Datos Preprocesados

# In[9]:


# Uso de la función
data_dict = {
    'df_train': cell_diagnosis_balanced_for_train_outliers,
    'df_valid': cell_diagnosis_balanced_for_valid_outliers,
    'df_test': cell_diagnosis_balanced_test_outliers
}

save_processed_data(
    loader=loader_cell_diagnosis,
    data_dict=data_dict,
    data_dir=data_dir_p1,
    dataset_name="cell_diagnosis_balanced",
    processing_type="preprocessed"
)


# %% Recarga de Datos Preprocesados

# In[10]:


loader_cell_diagnosis.read_data(
    train_file="preprocessed/cell_diagnosis_balanced_train.csv",
    valid_file="preprocessed/cell_diagnosis_balanced_valid.csv",
    test_file="preprocessed/cell_diagnosis_balanced_test.csv",
    splitted=True
)
cell_diagnosis_balanced_for_train_outliers, cell_diagnosis_balanced_for_valid_outliers, cell_diagnosis_balanced_test_outliers = loader_cell_diagnosis.get_pandas_data(splitted=True)


# %% Análisis de Rango de Features

# In[11]:


print_numerical_features_range(cell_diagnosis_balanced_for_train_outliers)


# %% Análisis Exploratorio y Visualizaciones

# In[12]:


analyze_categorical_variables(cell_diagnosis_balanced_for_train_outliers, ["CellType", "GeneticMutation", "Diagnosis"])


# In[13]:


exclude_columns = ["Diagnosis"]
numerical_cols_cell_diagnosis_balanced_train_outliers = cell_diagnosis_balanced_for_train_outliers.select_dtypes(include=["number"]).columns.tolist()

numerical_cols_cell_diagnosis_balanced_train_outliers = [
    col for col in numerical_cols_cell_diagnosis_balanced_train_outliers if col not in exclude_columns
]

FEATURES_TO_PLOT = ["CellSize", "MitosisRate", "NucleusDensity"]


# In[14]:


plot_data_analysis_all_visualizations(
    cell_diagnosis_balanced_for_train_outliers,
    numerical_cols_cell_diagnosis_balanced_train_outliers,
    "Diagnosis",
    FEATURES_TO_PLOT,
    "figures",
    fig_output_dir_p1
);


# %% Reemplazo de Outliers

# In[15]:


cell_diagnosis_balanced_train, params = replace_outliers_iqr(
    cell_diagnosis_balanced_for_train_outliers, method="winsorize", return_params=True, target_column="Diagnosis"
)
cell_diagnosis_balanced_valid = replace_outliers_iqr(
    cell_diagnosis_balanced_for_valid_outliers, method="winsorize", params=params, target_column="Diagnosis"
)
cell_diagnosis_balanced_test = replace_outliers_iqr(
    cell_diagnosis_balanced_test_outliers, method="winsorize", params=params, target_column="Diagnosis"
)


# In[16]:


# Actualizar el loader con los datos procesados
loader_cell_diagnosis.update(
    df_train=cell_diagnosis_balanced_train,
    df_valid=cell_diagnosis_balanced_valid,
    df_test=cell_diagnosis_balanced_test
)

data_dict = {
    'df_train': cell_diagnosis_balanced_train,
    'df_valid': cell_diagnosis_balanced_valid,
    'df_test': cell_diagnosis_balanced_test
}

save_processed_data(
    loader=loader_cell_diagnosis,
    data_dict=data_dict,
    data_dir=data_dir_p1,
    dataset_name="cell_diagnosis_balanced",
    processing_type="processed"
)


# In[17]:


loader_cell_diagnosis.read_data(
    train_file="processed/cell_diagnosis_balanced_train.csv",
    valid_file="processed/cell_diagnosis_balanced_valid.csv",
    test_file="processed/cell_diagnosis_balanced_test.csv",
    splitted=True
)
cell_diagnosis_balanced_train, cell_diagnosis_balanced_valid, cell_diagnosis_balanced_test = loader_cell_diagnosis.get_pandas_data(splitted=True)


# In[18]:


print_numerical_features_range(cell_diagnosis_balanced_valid)


# In[19]:


# heatmap_parameters = {
#     "label_fontsize": 18,
#     "title_fontsize": 18,
#     "tick_fontsize": 18,
#     "cbar_fontsize": 18,
#     "annot_fontsize": 18,
#     "figsize": (18, 12)
# }

# fig_params = {
#     "heatmap_params": heatmap_parameters
# }


plot_data_analysis_all_visualizations(
    cell_diagnosis_balanced_train,
    numerical_cols_cell_diagnosis_balanced_train_outliers,
    "Diagnosis",
    FEATURES_TO_PLOT,
    "figures",
    fig_output_dir_p1

);


# # Punto 1.2: Modelado con Regresión Logística - Feature Engineering

# In[20]:


transformations = {
    "Nucleus_Cytoplasm_Ratio": lambda d: (d["CellSize"] - d["CytoplasmSize"]) / d["CytoplasmSize"],
    "ProliferationIndex": lambda d: d["GrowthFactor"] * d["MitosisRate"],
    "DensityTextureIndex": lambda d: d["NucleusDensity"] * d["ChromatinTexture"],
}
cell_diagnosis_balanced_train = apply_feature_engineering(cell_diagnosis_balanced_train, transformations, verbose=False)
cell_diagnosis_balanced_valid = apply_feature_engineering(cell_diagnosis_balanced_valid, transformations, verbose=False)
cell_diagnosis_balanced_test = apply_feature_engineering(cell_diagnosis_balanced_test, transformations, verbose=False)


# %% Cross Validation para Optimización de Lambda

# In[21]:


X_train_cell_diagnosis, y_train_cell_diagnosis, X_valid_cell_diagnosis, y_valid_cell_diagnosis = process_data(
    cell_diagnosis_balanced_train,
    cell_diagnosis_balanced_valid
)


# In[22]:


lambda_values = np.concatenate([
    np.logspace(-6, -3, 50),
    np.logspace(-3, 1, 200)
])


# In[23]:


best_lambda, mean_scores = cross_validate_lambda(
    X=X_train_cell_diagnosis,
    y=y_train_cell_diagnosis,
    lambda_values=lambda_values,
    metric_fn=f1_score,
    k_folds=5,
    verbose=False,
    threshold=0.2
)


# In[24]:


plot_lambda_tuning(
    lambda_values=lambda_values,
    scores=mean_scores,
    metric_name=f1_score.__name__,
    best_lambda=best_lambda,
    figsize=(15, 5)
)
plt.show()
print(f"Optimal lambda value: {best_lambda}")


# %% Entrenamiento Final y Evaluación en Conjunto de Validación

# In[141]:





# In[25]:


X_train_cell_diagnosis, y_train_cell_diagnosis, X_valid_cell_diagnosis, y_valid_cell_diagnosis = process_data(
    cell_diagnosis_balanced_train,
    cell_diagnosis_balanced_valid
)
final_config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-4,
    lambda_reg=best_lambda
)


# In[26]:


final_model = LogisticRegression(final_config)
final_model.fit(X_train_cell_diagnosis, y_train_cell_diagnosis);


# In[27]:


results = evaluate_model(
    threshold=0.2,
    model=final_model,
    X_test=X_valid_cell_diagnosis,
    y_test=y_valid_cell_diagnosis,
    class_names=["Benign", "Malignant"],
    save_dir=fig_output_dir_p1,
    figsize=(20, 5),
    base_filename="test_balanced"
)


# # Punto 1.3: Evaluación en Conjunto de Test Final

# In[28]:


X_final_test = cell_diagnosis_balanced_test.drop(columns=["Diagnosis"])
y_final_test = cell_diagnosis_balanced_test["Diagnosis"]


# In[29]:


X_final_test = DataLoader.encode_categorical(X_final_test)
X_final_test, y_final_test = X_final_test.to_numpy(), y_final_test.to_numpy()


# In[30]:


results = evaluate_model(
    model=final_model,
    X_test=X_final_test,
    y_test=y_final_test,
    class_names=["Benign", "Malignant"],
    save_dir=fig_output_dir_p1,
    figsize=(20, 5),
    base_filename="test_balanced"
)


# # Punto 1.4: Datos Desbalanceados - Carga y Preprocesamiento

# In[31]:


config_cell_diagnosis_imbalanced = DatasetConfig(
    data_dir=data_dir_p1,
    target_column="Diagnosis",
    split_config=SplitConfig(test_size=0.2, shuffle=True, random_state=42)
)
loader_cell_diagnosis_imbalanced = DataLoader(config_cell_diagnosis_imbalanced)
loader_cell_diagnosis_imbalanced.read_data(
    dev_file="raw/cell_diagnosis_imbalanced_dev.csv",
    test_file="raw/cell_diagnosis_imbalanced_test.csv",
    splitted=False
);


# In[32]:


cell_diagnosis_imbalanced_dev_outliers, cell_diagnosis_imbalanced_test_outliers = loader_cell_diagnosis_imbalanced.get_pandas_data(splitted=False)

cell_diagnosis_imbalanced_for_train_outliers, cell_diagnosis_imbalanced_for_valid_outliers = train_valid_split(
    cell_diagnosis_imbalanced_dev_outliers, test_size=0.2, random_state=42
)


# In[33]:


analyze_categorical_variables(cell_diagnosis_imbalanced_for_train_outliers, ["CellType", "GeneticMutation", "Diagnosis"]);


# In[34]:


analyze_null_values(
    [cell_diagnosis_imbalanced_for_train_outliers, cell_diagnosis_imbalanced_test_outliers],
    ["conjunto de entrenamiento imbalanceado", "conjunto de prueba imbalanceado"]
);


# In[35]:


cell_diagnosis_imbalanced_for_train_outliers, cell_diagnosis_imbalanced_for_valid_outliers, cell_diagnosis_imbalanced_test_outliers = impute_missing_values(
    cell_diagnosis_imbalanced_for_train_outliers,
    cell_diagnosis_imbalanced_for_valid_outliers,
    cell_diagnosis_imbalanced_test_outliers,
    knn_neighbors=8,
    knn_weights="distance"
)

loader_cell_diagnosis_imbalanced.update(
    df_train=cell_diagnosis_imbalanced_for_train_outliers,
    df_valid=cell_diagnosis_imbalanced_for_valid_outliers,
    df_test=cell_diagnosis_imbalanced_test_outliers
);


# In[36]:


# Guardar los datos preprocesados usando la función save_processed_data


data_dict = {
    'df_train': cell_diagnosis_imbalanced_for_train_outliers,
    'df_valid': cell_diagnosis_imbalanced_for_valid_outliers,
    'df_test': cell_diagnosis_imbalanced_test_outliers
}

save_processed_data(
    loader=loader_cell_diagnosis_imbalanced,
    data_dict=data_dict,
    data_dir=data_dir_p1,
    dataset_name="cell_diagnosis_imbalanced",
    processing_type="preprocessed"
)


# In[37]:


loader_cell_diagnosis_imbalanced.read_data(
    train_file="preprocessed/cell_diagnosis_imbalanced_train.csv",
    valid_file="preprocessed/cell_diagnosis_imbalanced_valid.csv",
    test_file="preprocessed/cell_diagnosis_imbalanced_test.csv",
    splitted=True
);


# In[38]:


print_numerical_features_range(cell_diagnosis_imbalanced_for_valid_outliers)


# In[39]:


plot_data_analysis_all_visualizations(
    cell_diagnosis_imbalanced_for_train_outliers,
    numerical_cols_cell_diagnosis_balanced_train_outliers,
    "Diagnosis",
    FEATURES_TO_PLOT,
    "figures",
    fig_output_dir_p1
)


# In[40]:


cell_diagnosis_imbalanced_for_train, params = replace_outliers_iqr(
    cell_diagnosis_imbalanced_for_train_outliers, method="winsorize", return_params=True, target_column="Diagnosis"
)
cell_diagnosis_imbalanced_for_valid = replace_outliers_iqr(
    cell_diagnosis_imbalanced_for_valid_outliers, method="winsorize", params=params, target_column="Diagnosis"
)
cell_diagnosis_imbalanced_test = replace_outliers_iqr(
    cell_diagnosis_imbalanced_test_outliers, method="winsorize", params=params, target_column="Diagnosis"
)


# In[41]:


data_dict = {
    'df_train': cell_diagnosis_imbalanced_for_train,
    'df_valid': cell_diagnosis_imbalanced_for_valid,
    'df_test': cell_diagnosis_imbalanced_test
}

save_processed_data(
    loader=loader_cell_diagnosis_imbalanced,
    data_dict=data_dict,
    data_dir=data_dir_p1,
    dataset_name="cell_diagnosis_imbalanced",
    processing_type="processed"
)


# In[42]:


loader_cell_diagnosis_imbalanced.read_data(
    train_file="processed/cell_diagnosis_imbalanced_train.csv",
    valid_file="processed/cell_diagnosis_imbalanced_valid.csv",
    test_file="processed/cell_diagnosis_imbalanced_test.csv",
    splitted=True
);


# In[43]:


plot_data_analysis_all_visualizations(
    cell_diagnosis_imbalanced_for_train,
    numerical_cols_cell_diagnosis_balanced_train_outliers,
    "Diagnosis",
    FEATURES_TO_PLOT,
    "figures",
    fig_output_dir_p1
)


# In[44]:


# Ejemplo de uso
X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced, X_val_cell_diagnosis_imbalanced, y_val_cell_diagnosis_imbalanced = process_data(
    cell_diagnosis_imbalanced_for_train,
    cell_diagnosis_imbalanced_for_valid
)


# In[45]:


X_test_cell_diagnosis_imbalanced, y_test_cell_diagnosis_imbalanced = loader_cell_diagnosis_imbalanced.get_processed_test_data(
    return_numpy=True,
    encode_categorical=True,
    normalize=False
)


# In[46]:


class_names = ["Malignant","Benign" ]


# In[47]:


all_models = {}


# In[48]:


# --- Validación cruzada sin rebalancing para elegir lambda ---
# Se asume que ya se tiene definido un rango de valores para lambda_values y la función f1_score.
best_lambda_nr, cv_scores_nr = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced,
    y=y_train_cell_diagnosis_imbalanced,
    lambda_values=lambda_values,       # Por ejemplo, un np.concatenate de logspace
    metric_fn=f1_score,
    k_folds=5,
    verbose=False,
    threshold=0.5,
    aggregate_predictions=True,         # Calcula la métrica global al final
    resampler=None                      # No aplica re-muestreo en este caso
)

print(f"Optimal lambda for non-rebalancing: {best_lambda_nr}")

# --- Configurar la regresión logística con el lambda óptimo obtenido ---

lr_config_no_rebalancing = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-4,
    lambda_reg=best_lambda_nr,
    random_state=42,
    verbose=False
)


# --- Entrenamiento del modelo sin aplicar ninguna técnica de rebalancing ---
model_no_rebalancing = LogisticRegression(lr_config_no_rebalancing)
model_no_rebalancing.fit(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced)

# --- Evaluación del modelo en el conjunto de validación ---
metrics_no_rebalancing = evaluate_model(
    model=model_no_rebalancing,
    X_test=X_val_cell_diagnosis_imbalanced,
    y_test=y_val_cell_diagnosis_imbalanced, 
    class_names=class_names, 
    show_plots=False,
    threshold=0.5,
    figsize=(10, 6),
    print_metrics=False
)

all_models["No rebalancing"] = {
    "model": model_no_rebalancing,
    "metrics": metrics_no_rebalancing
}


# In[49]:


# Primero, definimos la estrategia de undersampling:
undersampler = RandomUnderSampler(RebalancingConfig(random_state=42, sampling_strategy=0.5))


# Realizar validación cruzada aplicando undersampling en cada fold
best_lambda_under, cv_scores_under = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced,
    y=y_train_cell_diagnosis_imbalanced,
    lambda_values=lambda_values,
    metric_fn=f1_score,           # O la métrica que prefieras
    k_folds=5,
    verbose=False,
    threshold=0.6,
    aggregate_predictions=True,   # Calcula la métrica global al final
    resampler=undersampler         # Aplica undersampling en cada fold
)

print(f"\nOptimal lambda for undersampling: {best_lambda_under}")

# Crear la configuración final del modelo con el lambda óptimo obtenido
lr_config_under = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-4,
    lambda_reg=best_lambda_under,
    random_state=42,
    verbose=False
)

# Re-aplicar undersampling al conjunto de entrenamiento final (para consistencia)
X_train_under, y_train_under = undersampler.fit_resample(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced)

# Entrenar el modelo final con la configuración optimizada
model_undersampling = LogisticRegression(lr_config_under)
model_undersampling.fit(X_train_under, y_train_under)

# Evaluar el modelo en el conjunto de validación
metrics_undersampling = evaluate_model(
    threshold=0.6,
    model=model_undersampling, 
    X_test=X_val_cell_diagnosis_imbalanced, 
    y_test=y_val_cell_diagnosis_imbalanced, 
    class_names=class_names, 
    show_plots=False,
    figsize=(10, 6), 
    print_metrics=False
)

# Agregar el modelo y sus métricas al diccionario de modelos evaluados
all_models["Undersampling"] = {
    "model": model_undersampling,
    "metrics": metrics_undersampling
}


# In[50]:


# --- Definir el oversampler y re-muestrear el conjunto de entrenamiento original ---
oversampler = RandomOverSampler(RebalancingConfig(random_state=42))



# --- Definir un rango de valores para lambda ---
lambda_values = np.concatenate([
    np.logspace(-6, -3, 50),
    np.logspace(-3, 1, 200)
])

# --- Realizar validación cruzada utilizando el oversampler ---
# Se utiliza el parámetro "resampler" para que en cada fold se aplique la técnica de re-muestreo.
optimal_lambda, cv_scores = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced,             # Se utiliza el conjunto original; el oversampler se aplica internamente
    y=y_train_cell_diagnosis_imbalanced,
    lambda_values=lambda_values,
    metric_fn=f1_score,
    k_folds=5,
    verbose=False,
    threshold=0.5,
    resampler=oversampler  # Aplica oversampling en cada fold
)

print(f"\nOptimal lambda: {optimal_lambda}")

# --- Configurar el modelo con el lambda óptimo ---
lr_config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-4,
    lambda_reg=optimal_lambda,
    random_state=42,
    verbose=False
)


# --- Re-aplicar oversampling al conjunto de entrenamiento para el entrenamiento final ---
X_train_over, y_train_over = oversampler.fit_resample(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced)

# --- Entrenar el modelo final con la configuración optimizada ---
model_oversampling = LogisticRegression(lr_config)
model_oversampling.fit(X_train_over, y_train_over)

# --- Evaluar el modelo final en el conjunto de validación ---
metrics_oversampling = evaluate_model(
    model=model_oversampling,
    X_test=X_val_cell_diagnosis_imbalanced,
    y_test=y_val_cell_diagnosis_imbalanced,
    class_names=class_names,
    show_plots=False,
    threshold=0.5,
    figsize=(10, 6),
    print_metrics=False
)

all_models["Oversampling duplicate"] = {
    "model": model_oversampling,
    "metrics": metrics_oversampling
}


# In[51]:


smote = SMOTE(SMOTEConfig(random_state=42, k_neighbors=5))

optimal_lambda, cv_scores = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced,             # Se utiliza el conjunto original; el oversampler se aplica internamente
    y=y_train_cell_diagnosis_imbalanced,
    lambda_values=lambda_values,
    metric_fn=f1_score,
    k_folds=5,
    verbose=False,
    threshold=0.5,
    resampler=smote  # Aplica oversampling en cada fold
)

print(f"\nOptimal lambda: {optimal_lambda}")

# --- Configurar el modelo con el lambda óptimo ---
lr_config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=1000,
    tol=1e-4,
    lambda_reg=optimal_lambda,
    random_state=42,
    verbose=False
)


X_train_smote, y_train_smote = smote.fit_resample(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced)
model_smote = LogisticRegression(lr_config)
model_smote.fit(X_train_smote, y_train_smote);

metrics_smote = evaluate_model(
    model_smote, X_val_cell_diagnosis_imbalanced, y_val_cell_diagnosis_imbalanced, 
    class_names=class_names, show_plots=False,
    threshold=0.5, figsize=(10, 6), print_metrics=False
)
all_models["Oversampling SMOTE"] = {
    "model": model_smote,
    "metrics": metrics_smote
}


# In[52]:


print("\n===== Training model with Cost reweighting =====")
# Calcula los pesos de clase (por ejemplo, usando una función calculate_class_weights que ya tengas)
class_weights = calculate_class_weights(y_train_cell_diagnosis_imbalanced)
print(f"Class weights: {class_weights}")


# Realizar validación cruzada cost-sensitive para elegir el mejor lambda:
# Se pasa el parámetro resampler si se desea aplicar alguna técnica de re-muestreo en cada fold; de lo contrario, se puede dejar como None.
best_lambda_cost_rw, cv_scores = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced,
    y=y_train_cell_diagnosis_imbalanced,
    lambda_values=lambda_values,
    metric_fn=f1_score,           # Puedes elegir la métrica que prefieras
    k_folds=5,
    verbose=False,
    threshold=0.5,
    aggregate_predictions=True,   # Calcula la métrica al final usando todas las predicciones
    resampler=None                # O reemplázalo por un resampler (por ejemplo, oversampler) si lo deseas
)

print(f"\nOptimal lambda found: {best_lambda_cost_rw}")

# Crear la configuración del modelo cost-sensitive con el lambda óptimo obtenido
cost_sensitive_config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=3000,
    tol=1e-4,
    lambda_reg=best_lambda_cost_rw,
    random_state=42,
    verbose=False,
    class_weight=class_weights
)
model_cost = LogisticRegression(cost_sensitive_config)
model_cost.fit(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced)

# Evaluar el modelo entrenado en el conjunto de validación
metrics_cost = evaluate_model(
    model=model_cost,
    X_test=X_val_cell_diagnosis_imbalanced,
    y_test=y_val_cell_diagnosis_imbalanced,
    class_names=class_names,
    show_plots=False,
    threshold=0.5,
    figsize=(10, 6),
    print_metrics=False
)

all_models["Cost re-weighting"] = {
    "model": model_cost,
    "metrics": metrics_cost
}


# In[53]:


# Antes de ejecutar evaluate_all_models, necesitamos asegurar que y_val_cell_diagnosis_imbalanced sea un array NumPy
y_val_cell_diagnosis_imbalanced = np.asarray(y_val_cell_diagnosis_imbalanced)

val_metrics_df, val_metrics = evaluate_all_models(
    all_models, 
    X_val_cell_diagnosis_imbalanced, 
    y_val_cell_diagnosis_imbalanced, 
    class_names, 
    fig_output_dir_p1,
    prefix="validation",
    subplots=True,
    show_plot=True,
    individual_plots=False
)


# In[54]:


test_metrics_df, test_metrics = evaluate_all_models(
    all_models, 
    X_test_cell_diagnosis_imbalanced, 
    y_test_cell_diagnosis_imbalanced, 
    class_names, 
    fig_output_dir_p1,
    subplots=True,
    prefix="test",
    show_plot=True,
    individual_plots=False
)


# In[55]:


from utils.utils import format_metrics_table
val_metrics_display = format_metrics_table(val_metrics_df, "Validation Metrics Summary")
test_metrics_display = format_metrics_table(test_metrics_df, "Test Metrics Summary")


# # Punto 1.6
# 

# El mejor modelo es cost reweighting

# In[56]:


cell_diagnosis_imbalanced_dev_outliers = impute_missing_values(
    cell_diagnosis_imbalanced_dev_outliers,
    knn_neighbors=5,
    knn_weights="distance"
)


cell_diagnosis_imbalanced_dev_outliers = replace_outliers_iqr(
    cell_diagnosis_imbalanced_dev_outliers,
    method="iqr",
    target_column="Diagnosis"
)



X_train_cell_diagnosis_imbalanced_dev_outliers = cell_diagnosis_imbalanced_dev_outliers.drop(columns=["Diagnosis"])


# Encoder la variable diagnosis
X_train_cell_diagnosis_imbalanced_dev_outliers = DataLoader.encode_categorical(X_train_cell_diagnosis_imbalanced_dev_outliers).to_numpy()
y_train_cell_diagnosis_imbalanced_dev_outliers = cell_diagnosis_imbalanced_dev_outliers["Diagnosis"].to_numpy()


best_lambda_cost_rw, cv_scores = cross_validate_lambda(
    X=X_train_cell_diagnosis_imbalanced_dev_outliers,
    y=y_train_cell_diagnosis_imbalanced_dev_outliers,
    lambda_values=lambda_values,
    metric_fn=f1_score,          
    k_folds=5,
    verbose=False,
    threshold=0.5,
    aggregate_predictions=True,   
    resampler=None                
)

print(f"\nOptimal lambda found: {best_lambda_cost_rw}")

# Crear la configuración del modelo cost-sensitive con el lambda óptimo obtenido
cost_sensitive_config = LogisticRegressionConfig(
    learning_rate=0.01,
    max_iter=3000,
    tol=1e-4,
    lambda_reg=best_lambda_cost_rw,
    random_state=42,
    verbose=False,
    class_weight=class_weights
)
model_cost = LogisticRegression(cost_sensitive_config)
model_cost.fit(X_train_cell_diagnosis_imbalanced, y_train_cell_diagnosis_imbalanced);


# In[57]:


# Evaluar el modelo entrenado en el conjunto de validación
metrics_cost = evaluate_model(
    model=model_cost,
    X_test=X_test_cell_diagnosis_imbalanced,
    y_test=y_test_cell_diagnosis_imbalanced,
    class_names=class_names,
    show_plots=True,
    threshold=0.5,
    figsize=(20, 6),
    print_metrics=True
)


# # Punto 2: Análisis del Dataset WAR_class

# In[58]:


fig_output_dir_p2 = "./figures/p2"
os.makedirs(fig_output_dir_p2, exist_ok=True)
data_dir_p2 = data_dir / "p2"


# In[59]:


config_war_class = DatasetConfig(
    data_dir=data_dir_p2,
    target_column="war_class",
    split_config=SplitConfig(test_size=0.2, shuffle=True, random_state=42)
)
loader_war_class = DataLoader(config_war_class)
loader_war_class.read_data(
    dev_file="raw/WAR_class_dev.csv",
    test_file="raw/WAR_class_test.csv",
    splitted=False
)
war_class_dev, war_class_test = loader_war_class.get_pandas_data(splitted=False)


# In[60]:


war_class_for_train, war_class_for_valid = train_valid_split(war_class_dev, test_size=0.2, random_state=42)
print("Valores nulos en conjunto de entrenamiento:")
print(war_class_for_train.isnull().sum().sum())
print("Valores nulos en conjunto de validación:")
print(war_class_for_valid.isnull().sum().sum())


# In[61]:


print("WAR_class_train.csv - Información básica:")
print(f"Número de filas: {war_class_for_train.shape[0]}")
print(f"Número de columnas: {war_class_for_train.shape[1]}")
print("\nPrimeras filas del dataset:")
display(war_class_for_train.head())


# In[62]:


analyze_null_values(
    [war_class_for_train, war_class_test],
    ["conjunto de entrenamiento WAR_class", "conjunto de prueba WAR_class"]
);


# In[63]:


duplicated_rows = war_class_for_train.duplicated().sum()
print(f"Número de filas duplicadas en WAR_class_dev: {duplicated_rows}")


# In[64]:


analyze_categorical_variables(war_class_for_train, ["war_class"])


# In[65]:


numerical_cols_war = war_class_for_train.select_dtypes(include=["number"]).columns.tolist()
numerical_cols_war = [col for col in numerical_cols_war if col != "war_class"]


# In[66]:


print("\nCaracterísticas principales:")
for col in numerical_cols_war:
    print(f"  {col}: {war_class_for_train[col].min():.2f} a {war_class_for_train[col].max():.2f} (promedio: {war_class_for_train[col].mean():.2f})")


# quitamos valores negativos (los ponemos como NAN y usamos KNN para imputarlos)

# In[67]:


numerical_cols_to_remove_negatives = ["poss", "mp", "war_total"]
# Apply the function to each dataframe
war_class_for_train_outliers, neg_count_train = remove_negative_values(war_class_for_train, numerical_cols_to_remove_negatives)
war_class_for_valid_outliers, neg_count_valid = remove_negative_values(war_class_for_valid, numerical_cols_to_remove_negatives)
war_class_test, neg_count_test = remove_negative_values(war_class_test, numerical_cols_to_remove_negatives)

# Print results
print(f"Replaced {neg_count_train} negative values in training set")
print(f"Replaced {neg_count_valid} negative values in validation set")
print(f"Replaced {neg_count_test} negative values in test set")

# Now impute missing values
war_class_for_train, war_class_for_valid, war_class_test = impute_missing_values(
    war_class_for_train_outliers,
    war_class_for_valid_outliers,
    war_class_test,
    knn_neighbors=5,
    knn_weights="distance"
)



# In[68]:


print("\nCaracterísticas principales:")
for col in numerical_cols_war:
    print(f"  {col}: {war_class_for_train[col].min():.2f} a {war_class_for_train[col].max():.2f} (promedio: {war_class_for_train[col].mean():.2f})")


# In[69]:


FEATURES_TO_PLOT_WAR = ["poss", "mp", "raptor_total", "pace_impact"]

fig_params ={
    "heatmap_params":{
        "label_fontsize":18,
        "title_fontsize":18,
        "tick_fontsize":18,
        "cbar_fontsize":18,
        "annot_fontsize":18,
        # "correlation_method":"pearson",
        "figsize":(18,12)
    },
    "outliers_params":{
        "method":"iqr",
        "target_column":"war_class",
        "return_params":True,
        "figsize":(16,10)
    },
    "numerical_distributions_params":{
        "figsize":(16,5)
    }
}

plot_data_analysis_all_visualizations(
    war_class_for_train,
    numerical_cols_war,
    "war_class",
    FEATURES_TO_PLOT_WAR,
    fig_output_dir_p2,
    fig_output_dir_p2,
    fig_params
)


# In[70]:


war_class_for_train, war_iqr_params = replace_outliers_iqr(war_class_for_train, method="winsorize", target_column="war_class",return_params=True)
war_class_for_valid = replace_outliers_iqr(war_class_for_valid, method="winsorize", params=war_iqr_params, target_column="war_class")
war_class_test = replace_outliers_iqr(war_class_test, method="winsorize", params=war_iqr_params, target_column="war_class")




loader_war_class.update(
    df_train=war_class_for_train,
    df_valid=war_class_for_valid,
    df_test=war_class_test
);


# In[71]:


data_dict = {
    "df_train":war_class_for_train,
    "df_valid":war_class_for_valid,
    "df_test":war_class_test
}

save_processed_data(
    data_dict=data_dict,
    data_dir=data_dir_p2,
    loader=loader_war_class,
    dataset_name="WAR_class"
)


# In[72]:


loader_war_class.read_data(
    train_file="preprocessed/WAR_class_train.csv",
    valid_file="preprocessed/WAR_class_valid.csv",
    test_file="preprocessed/WAR_class_test.csv",
    splitted=True
)
war_class_for_train, war_class_for_valid, war_class_test = loader_war_class.get_pandas_data(splitted=True)


# In[73]:


fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Primer gráfico: relación entre minutos jugados y posesiones
sns.scatterplot(data=war_class_for_train, x="mp", y="poss", hue="war_class", palette="viridis", s=100, alpha=0.7, ax=axes[0])
axes[0].set_title("Relación entre minutos jugados y posesiones por clase", fontsize=14)
axes[0].grid(True, alpha=0.3)

# Segundo gráfico: relación entre raptor_total y war_total
sns.scatterplot(data=war_class_for_train, x="raptor_total", y="war_total", hue="war_class", palette="viridis", s=100, alpha=0.7, ax=axes[1])
axes[1].set_title("Relación entre raptor_total y war_total por clase", fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_output_dir_p2, "scatter_relationships.png"), dpi=300, bbox_inches="tight")
plt.show()


# ## Punto 2.2

# In[74]:


X_train, y_train, X_valid, y_valid = process_data(war_class_for_train, war_class_for_valid, "war_class",encode_categorical=False)


X_test, y_test = loader_war_class.get_processed_test_data(
    encode_categorical=False,
    normalize=False
)

# X_test = war_class_test.drop(columns=["war_class","war_total"]).to_numpy()
# y_test = war_class_test["war_class"].to_numpy()


# In[75]:


X_train_normalized, params = normalize_data(X_train, None, return_params=True)
X_valid_normalized = normalize_data(X_valid, params)
X_test_normalized = normalize_data(X_test, params)


# In[76]:


from models.random_forest import RandomForest, RandomForestConfig
from models.lda import LDA, LDAConfig


# In[77]:


lda_config = LDAConfig(solver="svd")
lda_model = LDA(lda_config)
lda_model.fit(X_train_normalized, y_train)




evaluate_model(
    model=lda_model,
    X_test=X_valid_normalized,
    y_test=y_valid,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],  # Real class names
    save_dir=fig_output_dir_p2,
    base_filename="lda",
    figsize=(20, 6)
);






# In[78]:


lambda_values = np.logspace(-4, 1, 10)

best_lambda_multiclass_logistic_regression, cv_scores = cross_validate_lambda(
    X=X_train_normalized,
    y=y_train,
    lambda_values=lambda_values,
    metric_fn=f1_score,
    k_folds=5,
    average="weighted",
    verbose=True,
    threshold=0.5,
    aggregate_predictions=False,
    resampler=None
)


multiclass_logistic_regression_config = LogisticRegressionConfig(
    lambda_reg=best_lambda_multiclass_logistic_regression,
    max_iter=1000,
    tol=1e-4,
    learning_rate=0.01
)


model_multiclass_logistic_regression = LogisticRegression(multiclass_logistic_regression_config)

model_multiclass_logistic_regression.fit(X_train_normalized, y_train)

result_multiclass_logistic_regression = evaluate_model(
    model=model_multiclass_logistic_regression,
    X_test=X_valid_normalized,
    y_test=y_valid,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],
    save_dir=fig_output_dir_p2,
    base_filename="multiclass_logistic_regression",
    figsize=(20, 6),
    average="weighted"
)


# In[79]:


random_forest_config = RandomForestConfig(
    n_estimators=3,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    criterion="entropy"
)

model_random_forest = RandomForest(random_forest_config)

model_random_forest.fit(X_train, y_train)


result_random_forest = evaluate_model(
    model=model_random_forest,
    X_test=X_valid,
    y_test=y_valid,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],
    save_dir=fig_output_dir_p2,
    base_filename="random_forest",
    figsize=(20, 6),
    average="weighted"
)




# ## Punto 2.4

# In[80]:


# Combine training and validation sets


war_class_dev, neg_count_war_class_dev = remove_negative_values(war_class_dev, numerical_cols_to_remove_negatives)

war_class_dev = impute_missing_values(
    war_class_dev,
    knn_neighbors=5,
    knn_weights="distance"
)


war_class_dev = replace_outliers_iqr(
    war_class_dev,
    method="iqr",
    target_column="war_class"
)


X_train_war_class_dev, y_train_war_class_dev = war_class_dev.drop(columns=["war_class","war_total"]), war_class_dev["war_class"]




X_train_full_dev_normalized, params = normalize_data(X_train_war_class_dev, None, return_params=True)
X_test_full_dev_normalized = normalize_data(X_test, params)
y_test_full_dev = y_test



# In[95]:


plot_data_analysis_all_visualizations(
    war_class_dev,
    numerical_cols_war,
    "war_class",
    FEATURES_TO_PLOT_WAR,
    fig_output_dir_p2,
    fig_output_dir_p2
)


# In[ ]:


lda_config = LDAConfig(solver="svd")
lda_model = LDA(lda_config)
lda_model.fit(X_train_full_dev_normalized, y_train_war_class_dev)



evaluate_model(
    model=lda_model,
    X_test=X_test_full_dev_normalized,
    y_test=y_test_full_dev,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],  # Real class names
    save_dir=fig_output_dir_p2,
    base_filename="lda",
    figsize=(20, 6)
);






# In[91]:


lambda_values = np.logspace(-4, 1, 10)

best_lambda_multiclass_logistic_regression, cv_scores = cross_validate_lambda(
    X=X_train_full_dev_normalized,
    y=y_train_war_class_dev,
    lambda_values=lambda_values,
    metric_fn=f1_score,
    k_folds=5,
    average="weighted",
    verbose=True,
    threshold=0.5,
    aggregate_predictions=False,
    resampler=None
)


multiclass_logistic_regression_config = LogisticRegressionConfig(
    lambda_reg=best_lambda_multiclass_logistic_regression,
    max_iter=1000,
    tol=1e-4,
    learning_rate=0.01
)


model_multiclass_logistic_regression = LogisticRegression(multiclass_logistic_regression_config)

model_multiclass_logistic_regression.fit(X_train_full_dev_normalized, y_train_war_class_dev)

result_multiclass_logistic_regression = evaluate_model(
    model=model_multiclass_logistic_regression,
    X_test=X_test_full_dev_normalized,
    y_test=y_test_full_dev,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],
    save_dir=fig_output_dir_p2,
    base_filename="multiclass_logistic_regression",
    figsize=(20, 6),
    average="weighted"
)


# In[92]:


random_forest_config = RandomForestConfig(
    n_estimators=3,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    criterion="entropy"
)

model_random_forest = RandomForest(random_forest_config)

model_random_forest.fit(X_train_full_dev_normalized, y_train_war_class_dev)

result_random_forest = evaluate_model(
    model=model_random_forest,
    X_test=X_test_full_dev_normalized,
    y_test=y_test_full_dev,
    class_names=["Negative WAR", "Null WAR", "Positive WAR"],
    save_dir=fig_output_dir_p2,
    base_filename="random_forest",
    figsize=(20, 6),
    average="weighted"
)




