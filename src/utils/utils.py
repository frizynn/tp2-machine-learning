import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from models.logistic_regression import LogisticRegression
from evaluation.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve, auc
)
from utils.visuals import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_comparative_curves
)


from preprocessing.imputation import KNNImputer

def train_valid_split(df, test_size=0.2, random_state=42):
    """
    Divide un DataFrame en conjuntos de entrenamiento y validación.
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    valid_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[valid_indices]

##################################################
# Helper Functions: Cálculo de curvas y AUC
##################################################

def compute_binary_curves(y_true, y_score, pos_label):
    """
    Calcula curvas ROC y de Precisión-Recall para clasificación binaria.
    
    Retorna:
      fpr, tpr, roc_auc, precision_vals, recall_vals, pr_auc
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc_val = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    pr_auc_val = auc(recall_vals, precision_vals)
    return fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val

def compute_multiclass_curves(y_true, y_pred_prob, classes):
    """
    Calcula curvas ROC y de Precisión-Recall para cada clase en modo one-vs-rest.
    
    Retorna diccionarios con:
      - fpr, tpr, roc_auc_scores
      - precision_dict, recall_dict, pr_auc_scores
    """
    n_classes = len(classes)
    fpr, tpr = {}, {}
    roc_auc_scores = np.empty(n_classes)
    precision_dict, recall_dict = {}, {}
    pr_auc_scores = np.empty(n_classes)
    # Para cada clase, se binariza y se calcula la curva correspondiente.
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
        roc_auc_scores[i] = auc(fpr[i], tpr[i])
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
        pr_auc_scores[i] = auc(recall_dict[i], precision_dict[i])
    return fpr, tpr, roc_auc_scores, precision_dict, recall_dict, pr_auc_scores

def save_or_show_plot(fig, save_dir, base_filename, show_plot):
    """
    Guarda la figura en el directorio indicado si se especifica; 
    de lo contrario la muestra o la cierra según show_plot.
    """
    if save_dir and base_filename:
        fig.savefig(os.path.join(save_dir, f"{base_filename}.png"), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

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

def evaluate_model(model, X_test, y_test, class_names=None, threshold=0.5, pos_label=1, 
                   figsize=(16, 5), save_dir=None, base_filename=None, 
                   print_metrics=True, show_plots=True, subplots=True, average="weighted"):
    """
    Evalúa un modelo de clasificación calculando métricas (accuracy, precision, recall, f1)
    y generando gráficos (matriz de confusión, curvas ROC y PR) de forma optimizada y con código
    reutilizable.
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
    
    # Calcular matriz de confusión siempre (no solo en la sección de visualizaciones)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    if print_metrics:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # 3. VISUALIZACIONES
    if show_plots or save_dir:
        if class_names is None or len(class_names) != n_classes:
            class_names = ["Negative", "Positive"] if is_binary else [f"Class {c}" for c in classes]
        
        if subplots and has_proba:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            # (a) Matriz de confusión
            plot_confusion_matrix(conf_matrix, class_names=class_names, ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            # (b) Curva ROC y (c) Curva PR
            if is_binary:
                fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val = compute_binary_curves(y_test, y_pred_prob[:, 1], pos_label)
                axes[1].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
                axes[1].plot([0,1],[0,1],'k--',lw=2)
                axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC Curve'); axes[1].legend(loc="lower right")
                axes[2].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
                axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
                axes[2].set_title('Precision-Recall Curve'); axes[2].legend(loc="lower left")
            else:
                colors = plt.cm.get_cmap('tab10', n_classes)
                # ROC
                roc_auc_dict = {}
                for i, cls in enumerate(classes):
                    y_true_bin = (y_test == cls).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
                    roc_auc_dict[i] = auc(fpr, tpr)
                    axes[1].plot(fpr, tpr, lw=2, color=colors(i), 
                                 label=f'{class_names[i]} (AUC = {roc_auc_dict[i]:.2f})')
                axes[1].plot([0,1],[0,1],'k--',lw=1)
                axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title('ROC Curves'); axes[1].legend(loc="lower right", fontsize='small')
                # PR
                pr_auc_dict = {}
                for i, cls in enumerate(classes):
                    y_true_bin = (y_test == cls).astype(int)
                    prec_vals, rec_vals, _ = precision_recall_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
                    pr_auc_dict[i] = auc(rec_vals, prec_vals)
                    axes[2].plot(rec_vals, prec_vals, lw=2, color=colors(i), 
                                 label=f'{class_names[i]} (AUC = {pr_auc_dict[i]:.2f})')
                axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
                axes[2].set_title('Precision-Recall Curves'); axes[2].legend(loc="lower left", fontsize='small')
            plt.tight_layout()
            save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_all_plots", show_plots)
        else:
            # Gráficos separados: primero matriz de confusión
            plt.figure(figsize=figsize)
            plot_confusion_matrix(conf_matrix, class_names=class_names)
            save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_confusion_matrix", show_plots)
            # Si hay probabilidades, trazar curvas
            if has_proba:
                if is_binary:
                    plot_and_save_binary_curves(y_test, y_pred_prob, pos_label, save_dir, base_filename, show_plots, figsize, subplots=False)
                else:
                    plot_and_save_multiclass_curves(y_test, y_pred_prob, classes, class_names, save_dir, base_filename, show_plots, figsize, subplots=False)
    
    # 4. RETORNAR MÉTRICAS
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }
    if has_proba:
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
    return metrics

def plot_and_save_binary_curves(y_test, y_pred_prob, pos_label, save_dir, base_filename, show_plots, figsize, subplots=True):
    """
    Función auxiliar para trazar y (opcionalmente) guardar las curvas ROC y PR para clasificación binaria.
    Se reutiliza compute_binary_curves.
    """
    fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val = compute_binary_curves(y_test, y_pred_prob[:, 1], pos_label)
    if subplots:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        axes[0].plot([0,1], [0,1], 'k--', lw=2)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC Curve'); axes[0].legend(loc="lower right")
        axes[1].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('PR Curve'); axes[1].legend(loc="lower left")
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_curves", show_plots)
    else:
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=2)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_roc_curve", show_plots)
        plt.figure(figsize=figsize)
        plt.plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend(loc="lower left")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_pr_curve", show_plots)

def plot_and_save_multiclass_curves(y_test, y_pred_prob, classes, class_names, save_dir, base_filename, show_plots, figsize, subplots=True):
    """
    Función auxiliar para trazar y (opcionalmente) guardar curvas ROC y PR (One-vs-Rest) para clasificación multiclase.
    Se utiliza compute_multiclass_curves para calcular las curvas para cada clase.
    Retorna AUC promedio para ROC y PR.
    """
    fpr_dict, tpr_dict, roc_auc_scores, prec_dict, rec_dict, pr_auc_scores = compute_multiclass_curves(y_test, y_pred_prob, classes)
    avg_roc_auc = np.mean(roc_auc_scores)
    avg_pr_auc = np.mean(pr_auc_scores)
    n_classes = len(classes)
    
    if subplots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i in range(n_classes):
            ax1.plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
            ax2.plot(rec_dict[i], prec_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {pr_auc_scores[i]:.2f})')
        ax1.plot([0,1], [0,1], 'k--', lw=1)
        ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC Curves'); ax1.legend(loc="lower right", fontsize='small')
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('PR Curves'); ax2.legend(loc="lower left", fontsize='small')
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_multiclass_curves", show_plots)
    else:
        colors = plt.cm.get_cmap('tab10', n_classes)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curves'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_roc", show_plots)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(rec_dict[i], prec_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {pr_auc_scores[i]:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curves'); plt.legend(loc="lower left")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_pr", show_plots)
    return avg_roc_auc, avg_pr_auc

def evaluate_all_models(all_models, X, y, class_names, output_dir, prefix="", show_plot=False, individual_plots=False, subplots=True):
    """
    Evalúa múltiples modelos (almacenados en un diccionario) sobre el mismo conjunto de datos,
    generando una tabla comparativa de métricas y (opcionalmente) gráficos comparativos.
    Se utiliza una comprensión de diccionarios para evitar ciclos for redundantes.
    """
    # Evaluar todos los modelos
    print(f"Evaluando modelos en el conjunto de {prefix if prefix else 'validación'}")
    evaluation_metrics = {}
    for model_name, model_data in all_models.items():
        print(f"Evaluando {model_name}")
        evaluation_metrics[model_name] = evaluate_model(
            model=model_data["model"],
            X_test=X,
            y_test=y,
            class_names=class_names,
            threshold=0.5,
            figsize=(10, 6),
            save_dir=output_dir if individual_plots else None,
            base_filename=f"{model_name.replace(' ', '_').lower()}_{prefix}" if individual_plots else None,
            print_metrics=False,
            show_plots=individual_plots,
            subplots=subplots
        )
    
    # Crear tabla comparativa asegurando que todas las claves existan
    metrics_df = pd.DataFrame({
        "Model": list(evaluation_metrics.keys()),
        "Accuracy": [evaluation_metrics[m].get("accuracy", np.nan) for m in evaluation_metrics],
        "Precision": [evaluation_metrics[m].get("precision", np.nan) for m in evaluation_metrics],
        "Recall": [evaluation_metrics[m].get("recall", np.nan) for m in evaluation_metrics],
        "F-Score": [evaluation_metrics[m].get("f1", np.nan) for m in evaluation_metrics],
        "AUC-ROC": [evaluation_metrics[m].get("roc", np.nan) for m in evaluation_metrics],
        "AUC-PR": [evaluation_metrics[m].get("pr", np.nan) for m in evaluation_metrics]
    })
    
    # Formatear los valores numéricos
    for col in metrics_df.columns:
        if col != "Model":
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, int, np.number)) else x)
    
    # Guardar métricas en CSV
    metrics_file_path = os.path.join(output_dir, f"{prefix}_metrics_comparison.csv")
    metrics_df.to_csv(metrics_file_path, index=False)
    
    # Generar gráficos comparativos si se solicita
    if show_plot:
        models_for_plotting = {name: {"model": all_models[name]["model"], "metrics": evaluation_metrics[name]} for name in all_models}
        plot_comparative_curves(models_for_plotting, output_dir, prefix=f"{prefix}_", show_plot=show_plot, subplots=subplots)
    
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
            
    # Log the weights for debugging
    print(f"Class weights: {weights}")
    print(f"Class probabilities: {dict(zip(classes, probs))}")
    
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
