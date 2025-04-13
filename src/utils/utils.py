import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from preprocessing.data_loader import DataLoader
from models.logistic_regression import LogisticRegression
from evaluation.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve, auc,
    compute_binary_curves, compute_multiclass_curves
)
from utils.visuals import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_comparative_curves, plot_model_evaluation, save_or_show_plot,
    plot_outliers_analysis, plot_numerical_distributions, plot_correlation_heatmap,
    plot_outlier_boxplot
)


from preprocessing.imputation import KNNImputer

def train_valid_split(df, test_size=0.2, random_state=42, stratify=None):
    """
    Divide un DataFrame en conjuntos de entrenamiento y validación.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a dividir
    test_size : float, default=0.2
        Proporción del conjunto de datos a incluir en la división de validación
    random_state : int, default=42
        Semilla para la reproducibilidad
    stratify : array-like, optional
        Array de etiquetas para realizar una división estratificada. Si se proporciona,
        la distribución de clases en los conjuntos de entrenamiento y validación
        será similar a la del conjunto original.
        
    Returns
    -------
    tuple
        (df_train, df_valid) - DataFrames de entrenamiento y validación
    """
    np.random.seed(random_state)
    
    if stratify is not None:
        # Obtener índices únicos para cada clase
        unique_classes = np.unique(stratify)
        train_indices = []
        valid_indices = []
        
        for cls in unique_classes:
            # Obtener índices para la clase actual
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            
            # Calcular tamaño de validación para esta clase
            cls_test_size = int(len(cls_indices) * test_size)
            
            # Dividir índices
            valid_indices.extend(cls_indices[:cls_test_size])
            train_indices.extend(cls_indices[cls_test_size:])
            
        # Convertir a arrays numpy y mezclar
        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
        
    else:
        # División aleatoria simple
        shuffled_indices = np.random.permutation(len(df))
        test_set_size = int(len(df) * test_size)
        valid_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
    
    return df.iloc[train_indices], df.iloc[valid_indices]
##################################################
# Helper Functions: Cálculo de curvas y AUC
##################################################





##################################################
# Funciones de evaluación y representación
##################################################

def print_numerical_features_range(df, include_dtypes=['number'], decimal_places=4):
    """
    Muestra el rango (mínimo y máximo) de cada característica numérica en una tabla HTML.
    """
    desc = df.select_dtypes(include=include_dtypes).describe().loc[["min", "max"]]
    ranges_df = pd.DataFrame({
        "Feature": desc.columns,
        "Mínimo": np.around(desc.loc["min"].values, decimal_places),
        "Máximo": np.around(desc.loc["max"].values, decimal_places)
    })
    display(HTML(ranges_df.to_html(index=False)))
    return ranges_df

def get_model_metrics(model, X_test, y_test, threshold=0.5, pos_label=1, average="weighted", print_metrics=True):
    """
    Calcula las métricas de evaluación para un modelo de clasificación.
    
    Parameters
    ----------
    model : object
        Modelo entrenado con un método predict() implementado
    X_test : array-like
        Features del conjunto de prueba
    y_test : array-like
        Etiquetas verdaderas del conjunto de prueba
    threshold : float, default=0.5
        Umbral para la clasificación binaria
    pos_label : int, default=1
        Etiqueta de la clase positiva para métricas binarias
    average : str, default="weighted"
        Método de promedio para métricas multiclase
    print_metrics : bool, default=True
        Si es True, imprime las métricas principales
    
    Returns
    -------
    dict
        Diccionario con todas las métricas calculadas
    """
    classes = np.unique(y_test)
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    
    # 1. PREDICCIONES
    y_pred = model.predict(X_test, threshold=threshold) if isinstance(model, LogisticRegression) else model.predict(X_test)
    try:
        y_pred_prob = model.predict_prob(X_test)
        has_proba = True
    except (AttributeError, NotImplementedError):
        has_proba = False
    
    # 2. CALCULAR MÉTRICAS
    accuracy = accuracy_score(y_test, y_pred)
    if is_binary:
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    else:
        precision = precision_score(y_test, y_pred, average=average, labels=classes)
        recall = recall_score(y_test, y_pred, average=average, labels=classes)
        f1 = f1_score(y_test, y_pred, average=average, labels=classes)
    
    # Calcular matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    if print_metrics:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # 3. CREAR DICCIONARIO DE MÉTRICAS
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'has_proba': has_proba
    }
    
    # Agregar datos de curvas si hay probabilidades
    if has_proba:
        metrics['y_pred_prob'] = y_pred_prob
        
        if is_binary:
            fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val = compute_binary_curves(y_test, y_pred_prob[:, 1], pos_label)
            # Almacenar los datos completos de las curvas ROC y PR
            metrics['roc'] = {
                'auc': roc_auc_val,
                'fpr': fpr,
                'tpr': tpr
            }
            metrics['pr'] = {
                'average_precision': pr_auc_val,
                'precision': precision_vals,
                'recall': recall_vals
            }
        else:
            # Para multiclase, se guardan los datos de las curvas para cada clase
            fpr_dict, tpr_dict, roc_auc_scores, prec_dict, rec_dict, pr_auc_scores = compute_multiclass_curves(y_test, y_pred_prob, classes)
            avg_roc_auc = np.mean(roc_auc_scores)
            avg_pr_auc = np.mean(pr_auc_scores)
            
            metrics['roc'] = {
                'auc': avg_roc_auc,
                'fpr': fpr_dict,
                'tpr': tpr_dict,
                'auc_scores': roc_auc_scores
            }
            metrics['pr'] = {
                'average_precision': avg_pr_auc,
                'precision': prec_dict,
                'recall': rec_dict,
                'ap_scores': pr_auc_scores
            }
    else:
        metrics['roc'] = metrics['pr'] = None
    
    # Agregar datos adicionales que serán útiles para la visualización
    metrics['classes'] = classes
    metrics['is_binary'] = is_binary
    metrics['y_test'] = y_test
    metrics['pos_label'] = pos_label
    
    return metrics


def evaluate_model(model, X_test, y_test, class_names=None, threshold=0.5, pos_label=1, 
                   figsize=(16, 5), save_dir=None, base_filename=None, 
                   print_metrics=True, show_plots=True, subplots=True, average="weighted"):
    """
    Evalúa un modelo de clasificación calculando métricas y generando gráficos.
    Esta función es un wrapper que combina get_model_metrics y plot_model_evaluation.
    
    Parameters
    ----------
    model : object
        Modelo con métodos predict() y opcionalmente predict_prob()
    X_test : array-like
        Features del conjunto de prueba
    y_test : array-like
        Etiquetas verdaderas
    class_names : list, optional
        Nombres de las clases para los gráficos
    threshold : float, default=0.5
        Umbral de decisión para clasificación binaria
    pos_label : int, default=1
        Etiqueta de la clase positiva
    figsize : tuple, default=(16, 5)
        Tamaño de las figuras
    save_dir : str, optional
        Directorio para guardar gráficos
    base_filename : str, optional
        Nombre base para archivos guardados
    print_metrics : bool, default=True
        Si es True, imprime las métricas principales
    show_plots : bool, default=True
        Si es True, muestra los gráficos
    subplots : bool, default=True
        Si es True, combina gráficos en subplots
    average : str, default="weighted"
        Método de promedio para métricas multiclase

    Returns
    -------
    dict
        Diccionario con todas las métricas calculadas
    """
    # Obtener métricas
    metrics = get_model_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        pos_label=pos_label,
        average=average,
        print_metrics=print_metrics
    )
    
    # Generar visualizaciones si se solicita
    if show_plots or save_dir:
        plot_model_evaluation(
            metrics=metrics,
            class_names=class_names,
            figsize=figsize,
            save_dir=save_dir,
            base_filename=base_filename,
            show_plots=show_plots,
            subplots=subplots
        )
    
    # Eliminar datos que solo se usaron para visualización
    if 'y_test' in metrics:
        del metrics['y_test']
    if 'y_pred' in metrics:
        del metrics['y_pred']
    if 'y_pred_prob' in metrics:
        del metrics['y_pred_prob']
    
    return metrics


def evaluate_all_models(all_models, X, y, class_names, output_dir, prefix="", show_plot=False, individual_plots=False, subplots=True,
                      title_fontsize=20, label_fontsize=16, tick_fontsize=14, legend_fontsize=14):
    """
    Evalúa múltiples modelos sobre el mismo conjunto de datos,
    generando una tabla comparativa de métricas y (opcionalmente) gráficos comparativos.
    Utiliza get_model_metrics y plot_model_evaluation para una mejor organización del código.
    
    Parameters
    ----------
    all_models : dict
        Diccionario de modelos a evaluar
    X : array-like
        Features del conjunto a evaluar
    y : array-like
        Etiquetas verdaderas
    class_names : list
        Nombres de las clases para los gráficos
    output_dir : str
        Directorio para guardar los resultados
    prefix : str, default=""
        Prefijo para los archivos generados
    show_plot : bool, default=False
        Si es True, muestra los gráficos
    individual_plots : bool, default=False
        Si es True, genera gráficos individuales para cada modelo
    subplots : bool, default=True
        Si es True, combina gráficos en subplots
    title_fontsize : int, default=20
        Tamaño de fuente para títulos
    label_fontsize : int, default=16
        Tamaño de fuente para etiquetas de ejes
    tick_fontsize : int, default=14
        Tamaño de fuente para ticks
    legend_fontsize : int, default=14
        Tamaño de fuente para leyendas
    """
    # Evaluar todos los modelos
    print(f"Evaluando modelos en el conjunto de {prefix if prefix else 'validación'}")
    evaluation_metrics = {}
    
    for model_name, model_data in all_models.items():
        print(f"Evaluando {model_name}")
        
        # Calcular métricas
        metrics = get_model_metrics(
            model=model_data["model"],
            X_test=X,
            y_test=y,
            threshold=0.5,
            print_metrics=False
        )
        
        # Generar visualizaciones individuales si se solicita
        if individual_plots:
            plot_model_evaluation(
                metrics=metrics,
                class_names=class_names,
                figsize=(12, 8),  # Aumentado de (10, 6) a (12, 8)
                save_dir=output_dir,
                base_filename=f"{model_name.replace(' ', '_').lower()}_{prefix}",
                show_plots=show_plot,
                subplots=subplots
            )
        
        # Almacenar métricas para la comparación
        evaluation_metrics[model_name] = metrics
    
    # Crear tabla comparativa
    metrics_df = pd.DataFrame({
        "Model": list(evaluation_metrics.keys()),
        "Accuracy": [evaluation_metrics[m].get("accuracy", np.nan) for m in evaluation_metrics],
        "Precision": [evaluation_metrics[m].get("precision", np.nan) for m in evaluation_metrics],
        "Recall": [evaluation_metrics[m].get("recall", np.nan) for m in evaluation_metrics],
        "F-Score": [evaluation_metrics[m].get("f1", np.nan) for m in evaluation_metrics],
        "AUC-ROC": [evaluation_metrics[m].get("roc", {}).get("auc", np.nan) if evaluation_metrics[m].get("roc") is not None else np.nan for m in evaluation_metrics],
        "AUC-PR": [evaluation_metrics[m].get("pr", {}).get("average_precision", np.nan) if evaluation_metrics[m].get("pr") is not None else np.nan for m in evaluation_metrics]
    })
    
    # Formatear los valores numéricos
    for col in metrics_df.columns:
        if col != "Model":
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, int, np.number)) else x)
    
    # Guardar métricas en CSV
    metrics_file_path = os.path.join(output_dir, f"{prefix}_metrics_comparison.csv")
    metrics_df.to_csv(metrics_file_path, index=False)
    
    # Configurar tamaños de fuente
    _set_font_sizes = plt.rcParams.copy()
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize
    })
    
    # Generar gráficos comparativos si se solicita
    if show_plot:
        models_for_plotting = {name: {"model": all_models[name]["model"], "metrics": evaluation_metrics[name]} for name in all_models}
        plot_comparative_curves(
            models_for_plotting, 
            output_dir, 
            prefix=f"{prefix}_", 
            show_plot=show_plot, 
            subplots=subplots,
            figsize=(20, 10)  # Aumentado de tamaño predeterminado a (20, 10)
        )
    
    # Restaurar la configuración original de fuentes
    plt.rcParams.update(_set_font_sizes)
    
    return metrics_df, evaluation_metrics

def analyze_null_values(dataframes, dataset_names=None):
    """
    Analiza y muestra información detallada sobre valores nulos en uno o más DataFrames.
    """
    from IPython.display import display
    if not isinstance(dataframes, list):
        dataframes = [dataframes]
    if dataset_names is None or len(dataset_names) != len(dataframes):
        dataset_names = [f"Dataset {i+1}" for i in range(len(dataframes))]
    results = {}
    for df, name in zip(dataframes, dataset_names):
        print(f"Valores nulos en {name}:")
        null_counts = df.isnull().sum()
        total_rows = len(df)
        null_percentage = (null_counts / total_rows) * 100
        null_table = pd.DataFrame({
            'Columna': null_counts.index,
            'Cantidad de nulos': null_counts.values,
            'Porcentaje (%)': null_percentage.values.round(2)
        })
        display(null_table)
        samples_with_nulls = df.isnull().any(axis=1).sum()
        samples_without_nulls = total_rows - samples_with_nulls
        samples_percentage = (samples_with_nulls / total_rows) * 100
        summary = pd.DataFrame({
            'Métrica': ['Muestras con al menos un valor nulo', 'Muestras sin valores nulos', 'Total de muestras'],
            'Cantidad': [samples_with_nulls, samples_without_nulls, total_rows],
            'Porcentaje (%)': [samples_percentage.round(2), (100 - samples_percentage).round(2), 100.0]
        })
        display(summary)
        results[name] = {
            'null_table': null_table,
            'summary': summary,
            'total_nulls': null_counts.sum(),
            'total_rows': total_rows,
            'null_percentage': (null_counts.sum() / (total_rows * len(df.columns)) * 100).round(2)
        }
        print("\n")
    return results

def remove_negative_values(df, numerical_cols):
    """
    Reemplaza todos los valores negativos en las columnas numéricas del DataFrame por NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a procesar.
    
    Returns
    -------
    pd.DataFrame, int
        DataFrame procesado y el número total de valores negativos reemplazados.
    """
    df_copy = df.copy()
    # Seleccionar solo las columnas numéricas
    total_negatives = 0
    for col in numerical_cols:
        mask = (df_copy[col] < 0) & df_copy[col].notna()
        negatives = mask.sum()
        total_negatives += negatives
        df_copy.loc[mask, col] = np.nan
    return df_copy, total_negatives

def impute_missing_values(train_df, valid_df=None, test_df=None, knn_neighbors=8, knn_weights="distance"):
    """
    Imputa valores faltantes utilizando un KNNImputer personalizado en uno o varios DataFrames.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame de entrenamiento.
    valid_df : pd.DataFrame, optional
        DataFrame de validación.
    test_df : pd.DataFrame, optional
        DataFrame de test.
    knn_neighbors : int, default=8
        Número de vecinos a usar.
    knn_weights : str, default="distance"
        'uniform' o 'distance'.
    
    Returns
    -------
    Tuple[pd.DataFrame,...]
        Los DataFrames imputados; si solo se pasa train_df, se retorna ese DataFrame.
    """
    # Resetear índices para evitar problemas
    train_df = train_df.reset_index(drop=True)
    if valid_df is not None:
        valid_df = valid_df.reset_index(drop=True)
    if test_df is not None:
        test_df = test_df.reset_index(drop=True)
    
    print("Missing values before imputation:")
    print(f"Train: {train_df.isna().sum().sum()} missing values")
    if valid_df is not None:
        print(f"Valid: {valid_df.isna().sum().sum()} missing values")
    if test_df is not None:
        print(f"Test: {test_df.isna().sum().sum()} missing values")
    
    imputer = KNNImputer(n_neighbors=knn_neighbors, weights=knn_weights, return_df=True)
    imputer.fit(train_df)
    
    imputed_train = imputer.transform(train_df)
    imputed_valid = imputer.transform(valid_df) if valid_df is not None else None
    imputed_test = imputer.transform(test_df) if test_df is not None else None
    
    print("\nMissing values after imputation:")
    print(f"Train: {imputed_train.isna().sum().sum()} missing values")
    if imputed_valid is not None:
        print(f"Valid: {imputed_valid.isna().sum().sum()} missing values")
    if imputed_test is not None:
        print(f"Test: {imputed_test.isna().sum().sum()} missing values")
    
    results = [imputed_train]
    if valid_df is not None:
        results.append(imputed_valid)
    if test_df is not None:
        results.append(imputed_test)
    return tuple(results) if len(results) > 1 else results[0]

def apply_feature_engineering(df, transformations, inplace=False, verbose=False):
    """
    Aplica transformaciones de feature engineering a un DataFrame.
    """
    if not inplace:
        df = df.copy()
    for new_feature, transform_func in transformations.items():
        if verbose:
            print(f"Aplicando transformación para la feature: '{new_feature}'")
        df[new_feature] = transform_func(df)
    return df

def save_processed_data(loader, data_dict, data_dir, dataset_name, processing_type="preprocessed"):
    """
    Guarda los datos procesados o preprocesados en archivos CSV.
    
    Args:
        loader: Instancia del DataLoader
        data_dict: Diccionario con los DataFrames a guardar
        data_dir: Directorio base de datos
        dataset_name: Nombre del conjunto de datos
        processing_type: Tipo de procesamiento ("preprocessed" o "processed")
    """
    # Actualizar el loader con los nuevos datos
    loader.update(**data_dict)
    
    # Construir las rutas de los archivos
    base_path = data_dir / processing_type
    
    # Crear directorio si no existe
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Construir rutas completas
    file_paths = {
        'train': base_path / f"{dataset_name}_train.csv",
        'valid': base_path / f"{dataset_name}_valid.csv",
        'test': base_path / f"{dataset_name}_test.csv",
        'dev': base_path / f"{dataset_name}_dev.csv" if 'dev' in data_dict else None
    }
    
    # Guardar los datos
    loader.save_processed_data(
        df_train_dir=file_paths['train'],
        df_valid_dir=file_paths['valid'],
        df_test_dir=file_paths['test'],
        df_dev_dir=file_paths['dev'] if 'dev' in file_paths else None
    )

def calculate_class_weights(y):
    """
    Calculate class weights for cost-sensitive learning according to the formula C = π2/π1
    where π1 is the prior probability of the minority class and π2 is the prior probability 
    of the majority class.
    
    Parameters
    ----------
    y : np.ndarray
        Target vector with class labels
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping class labels to weights
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    # Calculate class probabilities
    probs = counts / total

    # Find majority class and its probability
    majority_idx = np.argmax(counts)
    majority_class = classes[majority_idx]
    majority_prob = probs[majority_idx]

    # Initialize weights with 1.0 for all classes
    weights = {cls: 1.0 for cls in classes}

    # Apply weight C = π2/π1 only to minority classes
    for i, cls in enumerate(classes):
        if cls != majority_class:  # If it's a minority class
            weights[cls] = majority_prob / probs[i]
            
    
    return weights

def normalize_data(X,params,return_params=False):
    if params is None:
        params = {}
        params["mean"] = X.mean()
        params["std"] = X.std()
    X_normalized = (X - params["mean"]) / params["std"]
    if return_params:
        return X_normalized, params
    return X_normalized


def format_metrics_table(metrics_df, title="Metrics Summary"):
    """
    Formatea una tabla de métricas para mostrar, extrayendo valores AUC de diccionarios cuando sea necesario.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame con las métricas a formatear
        title (str): Título a mostrar antes de la tabla
        
    Returns:
        pd.DataFrame: DataFrame formateado para visualización
    """
    # Crear una copia para no modificar el original
    display_df = metrics_df.copy()
    
    # Formatear columnas AUC si contienen diccionarios
    for col in ['AUC-ROC', 'AUC-PR']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x['auc']:.4f}" if isinstance(x, dict) and 'auc' in x 
                else (f"{x['average_precision']:.4f}" if isinstance(x, dict) and 'average_precision' in x 
                else str(x))
            )
    
    print(f"\n===== {title} =====")
    display(display_df)
    return display_df


def process_training_data(train_df, valid_df, target_column="Diagnosis", encode_categorical=True):
    # Resetear índices
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    

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
