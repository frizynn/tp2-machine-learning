import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from IPython.display import display
from matplotlib import rc
from typing import List, Optional, Dict, Union
import numpy as np


from evaluation.metrics import compute_binary_curves, compute_multiclass_curves

# ---------------------- Funciones Helper ----------------------
def _save_plot(save_path: Optional[str], dpi: int = 300):
    """Si se proporciona save_path, crea el directorio y guarda la figura."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

def _set_font_sizes(title_fs: int, label_fs: int, tick_fs: int, legend_fs: int):
    """Configura tamaños de fuente globalmente."""
    plt.rcParams.update({
        'axes.titlesize': title_fs,
        'axes.labelsize': label_fs,
        'xtick.labelsize': tick_fs,
        'ytick.labelsize': tick_fs,
        'legend.fontsize': legend_fs
    })

# ---------------------- Funciones de Plot ----------------------
def analyze_categorical_variables(df: pd.DataFrame, categorical_columns: List[str],
                                  normalize: bool = True, dpi: int = 300):
    """
    Muestra la distribución de variables categóricas en forma de tabla.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar.
    categorical_columns : list
        Lista de nombres de columnas categóricas.
    normalize : bool, default True
        Si se muestran porcentajes junto con los conteos.
    dpi : int, default 300
        Resolución para guardar figuras (en caso de ser necesario).
    """
    print("Distribución de variables categóricas:")
    for col in categorical_columns:
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True).mul(100).round(2)
        distribution_df = pd.DataFrame({
            col: counts.index,
            'Conteo': counts.values,
            'Porcentaje (%)': percentages.values
        })
        print(f"\nDistribución de {col}:")
        display(distribution_df)

def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: List[str],
                                 target_column: Optional[str] = None, bins: int = 20,
                                 figsize: tuple = (16, 12), output_dir: Optional[str] = None,
                                 filename: Optional[str] = None, title: Optional[str] = None,
                                 title_fontsize: int = 16, label_fontsize: int = 12,
                                 tick_fontsize: int = 10, legend_fontsize: int = 10,
                                 dpi: int = 300, features_to_plot: Optional[List[str]] = None):
    """
    Grafica la distribución (histograma con KDE) de variables numéricas.

    Si se especifica target_column, separa los datos por esa variable.
    """
    # Determinar columnas a plotear
    plot_cols = features_to_plot if features_to_plot is not None else numerical_cols
    plot_cols = [col for col in plot_cols if col in numerical_cols]
    if features_to_plot and len(plot_cols) != len(features_to_plot):
        print("Advertencia: Algunas características en features_to_plot no están en numerical_cols.")
        if not plot_cols:
            print("Error: No hay características válidas para graficar.")
            return None

    plt.figure(figsize=figsize)
    n_cols = len(plot_cols)
    n_rows = (n_cols + 2) // 3  # 3 columnas por fila

    _set_font_sizes(title_fontsize, label_fontsize, tick_fontsize, legend_fontsize)

    for i, col in enumerate(plot_cols):
        plt.subplot(n_rows, 3, i+1)
        if target_column:
            sns.histplot(data=df, x=col, hue=target_column, bins=bins, kde=True)
        else:
            sns.histplot(data=df, x=col, bins=bins, kde=True)
        plt.title(f'Distribución de {col}')

    plt.tight_layout()
    if title:
        plt.suptitle(title, fontsize=title_fontsize+4, y=1.02)
        plt.subplots_adjust(top=0.9)
    if output_dir and filename:
        _save_plot(os.path.join(output_dir, filename), dpi)
    return plt.gcf()

def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str], output_dir: Optional[str] = None,
                             filename: Optional[str] = None, figsize: tuple = (12, 10),
                             annot: bool = True, cmap: str = 'coolwarm', fmt: str = ".2f",
                             title: Optional[str] = None, use_serif_font: bool = True,
                             title_fontsize: int = 16, label_fontsize: int = 12,
                             tick_fontsize: int = 10, annot_fontsize: int = 8, cbar_fontsize: int = 10,
                             dpi: int = 300):
    """
    Crea y guarda (opcional) un mapa de calor de correlación para las columnas seleccionadas.
    """
    if use_serif_font:
        rc('font', **{'family': 'serif', 'serif': ['Helvetica']})

    plt.figure(figsize=figsize)
    corr_matrix = df[columns].corr()
    _set_font_sizes(title_fontsize, label_fontsize, tick_fontsize, legend_fs=cbar_fontsize)
    
    cbar_kws = {'label': r'Coeficiente de Correlación ($\rho$)'}
    annot_kws = {"size": annot_fontsize}
    heatmap = sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt,
                          linewidths=0.5, annot_kws=annot_kws, cbar_kws=cbar_kws)
    heatmap.collections[0].colorbar.ax.yaxis.label.set_fontsize(cbar_fontsize)
    
    plt.title(title if title else 'Mapa de Calor de Correlación', fontsize=title_fontsize)
    plt.tight_layout()
    if output_dir and filename:
        _save_plot(os.path.join(output_dir, filename), dpi)
    return plt.gcf()

def plot_outlier_boxplot(df: pd.DataFrame, column: str, figsize: tuple = (10, 6),
                         save_path: Optional[str] = None, show_plot: bool = True,
                         dpi: int = 300, title: Optional[str] = None):
    """
    Muestra un boxplot para detectar outliers en una columna.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column])
    plt.title(title if title else f'Boxplot de {column}')
    plt.tight_layout()
    _save_plot(save_path, dpi)
    if show_plot:
        plt.show()
    else:
        plt.close()
    return plt.gcf()

def plot_outliers_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           features_to_plot: Optional[List[str]] = None,
                           save_dir: Optional[str] = None, show_plots: bool = True,
                           filename: Optional[str] = None, figsize: tuple = (15, 6),
                           dpi: int = 300, use_subplots: bool = True) -> List[plt.Figure]:
    """
    Genera boxplots para un conjunto de características que presenten outliers.
    """
    # Usar todas las columnas numéricas si no se especifica
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    columns_to_analyze = [col for col in (columns if columns else numeric_cols) if col in df.columns and col in numeric_cols]
    features_to_plot = [col for col in (features_to_plot or []) if col in columns_to_analyze]
    
    if save_dir and features_to_plot:
        os.makedirs(save_dir, exist_ok=True)
    figures = []
    if use_subplots and features_to_plot:
        n_features = len(features_to_plot)
        n_rows = (n_features + 2) // 3  # 3 columnas por fila
        fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], figsize[1]*n_rows/2))
        axes = axes.flatten() if n_features > 1 else [axes]
        for i, column in enumerate(features_to_plot):
            sns.boxplot(x=df[column], ax=axes[i])
            axes[i].set_title(f'Boxplot de {column}')
        for j in range(len(features_to_plot), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        if save_dir:
            file_name = filename if filename else 'boxplots_all_features.png'
            _save_plot(os.path.join(save_dir, file_name), dpi)
        if show_plots:
            plt.show()
        else:
            plt.close()
        figures.append(fig)
    elif features_to_plot:
        for column in features_to_plot:
            if save_dir:
                if filename:
                    base_name, ext = os.path.splitext(filename)
                    ext = ext if ext else '.png'
                    sp = os.path.join(save_dir, f'{base_name}_{column}{ext}')
                else:
                    sp = os.path.join(save_dir, f'boxplot_{column}.png')
            else:
                sp = None
            fig = plot_outlier_boxplot(df, column, figsize, sp, show_plots, dpi)
            figures.append(fig)
    return figures

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: Optional[List[str]] = None,
                          figsize: tuple = (10, 8), cmap: str = 'Blues', normalize: bool = False,
                          title: Optional[str] = None, save_dir: Optional[str] = None,
                          filename: Optional[str] = None, dpi: int = 300,
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Genera una matriz de confusión (heatmap) a partir de los datos proporcionados.
    """
    fmt = '.2f' if normalize else 'd'
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cax = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    if title:
        ax.set_title(title, fontsize=16)
    fig = ax.figure
    fig.colorbar(cax, ax=ax)
    
    tick_marks = np.arange(len(conf_matrix))
    if class_names:
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=14,  # Aumentado de 10 a 14
                    fontweight='bold')  # Agregado para hacer los números más visibles
    
    ax.set_ylabel('Etiqueta real', fontsize=18)
    ax.set_xlabel('Etiqueta predicha', fontsize=18)
    if save_dir and filename:
        _save_plot(os.path.join(save_dir, filename), dpi)
    return ax.figure
def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: Optional[float] = None,
                   figsize: tuple = (10, 8),
                   title: str = 'Curva ROC (Característica Operativa del Receptor)',
                   save_dir: Optional[str] = None, filename: Optional[str] = None,
                   dpi: int = 300, label: Optional[str] = None, color: Optional[str] = None,
                   linestyle: str = '-', multiple_curves: bool = False,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Grafica la curva ROC.
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    curve_label = label if label is not None else (f'AUC = {auc_score:.3f}' if auc_score is not None else None)
    ax.plot(fpr, tpr, color=color, linestyle=linestyle, label=curve_label)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)  # línea diagonal
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    if title:
        ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    if curve_label:
        ax.legend(loc='lower right', frameon=True, fontsize=10)
    if save_dir and filename and not multiple_curves:
        _save_plot(os.path.join(save_dir, filename), dpi)
    if not multiple_curves:
        return plt.gcf()
    return ax

def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray,
                                average_precision: Optional[float] = None,
                                figsize: tuple = (10, 8), title: str = 'Curva Precision-Recall',
                                save_dir: Optional[str] = None, filename: Optional[str] = None,
                                dpi: int = 300, label: Optional[str] = None, color: Optional[str] = None,
                                linestyle: str = '-', multiple_curves: bool = False,
                                ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Grafica la curva Precision-Recall.
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    curve_label = label if label is not None else (f'AP = {average_precision:.3f}' if average_precision is not None else None)
    ax.plot(recall, precision, color=color, linestyle=linestyle, label=curve_label)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    if title:
        ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    if curve_label:
        ax.legend(loc='best', frameon=True, fontsize=10)
    if save_dir and filename and not multiple_curves:
        _save_plot(os.path.join(save_dir, filename), dpi)
    if not multiple_curves:
        return plt.gcf()
    return ax

def plot_lambda_tuning(lambda_values: List[float], scores: List[float],
                       metric_name: str = "Métrica", save_dir: Optional[str] = None,
                       filename: Optional[str] = None, best_lambda: Optional[float] = None,
                       figsize: tuple = (10, 6), dpi: int = 300, log_scale: bool = True) -> None:
    """
    Grafica los resultados de la búsqueda de lambda mediante validación cruzada.
    """
    plt.figure(figsize=figsize)
    plt.plot(lambda_values, scores, 'o-', linewidth=2, markersize=8)
    if best_lambda is not None:
        best_idx = np.argmin(np.abs(np.array(lambda_values) - best_lambda))
        best_score = scores[best_idx]
        plt.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7)
        plt.plot(best_lambda, best_score, 'ro', markersize=10, label=f'Mejor λ = {best_lambda}')
        plt.legend(fontsize=8)
    plt.xlabel('Lambda (λ)', fontsize=24)
    plt.ylabel(metric_name, fontsize=24)
    plt.grid(True, alpha=0.3)
    if log_scale and min(lambda_values) > 0:
        plt.xscale('log')
    if save_dir and filename:
        _save_plot(os.path.join(save_dir, filename), dpi)
    plt.tight_layout()
    plt.show()

def plot_comparative_curves(all_models: Dict, output_dir: str, prefix: str = "",
                            show_plot: bool = False, subplots: bool = True):
    """
    Genera gráficos comparativos (curvas ROC y PR) para múltiples modelos.
    Si los datos de curva no están disponibles, genera un gráfico de barras con los valores AUC.
    """
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    
    # Verifica si al menos un modelo tiene datos para curvas (diccionario con "fpr")
    can_plot_curves = any(
        isinstance(model_data.get("metrics", {}).get("roc"), dict) and 
        "fpr" in model_data["metrics"]["roc"]
        for model_data in all_models.values()
    )
    
    if not can_plot_curves:
        print("No se pueden generar gráficos comparativos porque las métricas no contienen datos de curva.")
        print("Solo se tienen valores numéricos de AUC-ROC y AUC-PR.")
        plt.figure(figsize=(10, 6))
        model_names = list(all_models.keys())
        roc_values = [model_data["metrics"]["roc"] for model_data in all_models.values()]
        pr_values = [model_data["metrics"]["pr"] for model_data in all_models.values()]
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, roc_values, width, label='AUC-ROC')
        plt.bar(x + width/2, pr_values, width, label='AUC-PR')
        plt.xlabel('Modelos')
        plt.ylabel('Valor AUC')
        plt.title('Comparación de AUC-ROC y AUC-PR entre modelos')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        bar_path = os.path.join(output_dir, f"{prefix}auc_comparison.png")
        _save_plot(bar_path, 300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return bar_path

    if subplots:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))
        # Curvas ROC
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data.get("metrics", {}).get("roc", {})
            if isinstance(roc_data, dict) and "auc" in roc_data:
                plot_roc_curve(roc_data["fpr"], roc_data["tpr"], auc_score=roc_data["auc"],
                               label=f"{name} (AUC = {roc_data['auc']:.4f})",
                               color=colors[i % len(colors)],
                               linestyle=linestyles[i % len(linestyles)],
                               multiple_curves=True, ax=ax_roc)
        ax_roc.set_title("Curvas ROC para Diferentes Técnicas de Rebalanceo", fontsize=14)
        ax_roc.legend(loc="lower right", fontsize=10)
        
        # Curvas PR
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data.get("metrics", {}).get("pr", {})
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                plot_precision_recall_curve(pr_data["precision"], pr_data["recall"],
                                            average_precision=pr_data["average_precision"],
                                            label=f"{name} (AP = {pr_data['average_precision']:.4f})",
                                            color=colors[i % len(colors)],
                                            linestyle=linestyles[i % len(linestyles)],
                                            multiple_curves=True, ax=ax_pr)
        ax_pr.set_title("Curvas Precision-Recall para Diferentes Técnicas de Rebalanceo", fontsize=14)
        ax_pr.legend(loc="lower left", fontsize=10)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f"{prefix}combined_curves_comparison.png")
        _save_plot(combined_path, 300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return combined_path
    else:
        # Gráficos por separado para ROC y PR
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data.get("metrics", {}).get("roc", {})
            if isinstance(roc_data, dict) and "auc" in roc_data:
                plot_roc_curve(roc_data["fpr"], roc_data["tpr"], auc_score=roc_data["auc"],
                               label=f"{name} (AUC = {roc_data['auc']:.4f})",
                               color=colors[i % len(colors)],
                               linestyle=linestyles[i % len(linestyles)],
                               multiple_curves=True, ax=ax_roc)
        ax_roc.set_title("Curvas ROC para Diferentes Técnicas de Rebalanceo", fontsize=14)
        ax_roc.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, f"{prefix}roc_curves_comparison.png")
        _save_plot(roc_path, 300)
        if not show_plot:
            plt.close()
        
        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data.get("metrics", {}).get("pr", {})
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                plot_precision_recall_curve(pr_data["precision"], pr_data["recall"],
                                            average_precision=pr_data["average_precision"],
                                            label=f"{name} (AP = {pr_data['average_precision']:.4f})",
                                            color=colors[i % len(colors)],
                                            linestyle=linestyles[i % len(linestyles)],
                                            multiple_curves=True, ax=ax_pr)
        ax_pr.set_title("Curvas Precision-Recall para Diferentes Técnicas de Rebalanceo", fontsize=14)
        ax_pr.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        pr_path = os.path.join(output_dir, f"{prefix}pr_curves_comparison.png")
        _save_plot(pr_path, 300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return roc_path, pr_path


def plot_model_evaluation(metrics, class_names=None, figsize=(16, 5), save_dir=None, 
                         base_filename=None, show_plots=True, subplots=True):
    """
    Genera visualizaciones para la evaluación de un modelo de clasificación utilizando
    las métricas previamente calculadas.
    
    Parameters
    ----------
    metrics : dict
        Diccionario con métricas calculadas por get_model_metrics
    class_names : list, optional
        Nombres de las clases para mostrar en gráficos
    figsize : tuple, default=(16, 5)
        Tamaño de la figura
    save_dir : str, optional
        Directorio donde guardar los gráficos
    base_filename : str, optional
        Nombre base para los archivos guardados
    show_plots : bool, default=True
        Si es True, muestra los gráficos
    subplots : bool, default=True
        Si es True, combina gráficos en subplots
    
    Returns
    -------
    None
    """
    # Extraer datos necesarios del diccionario de métricas
    classes = metrics['classes']
    n_classes = len(classes)
    is_binary = metrics['is_binary']
    conf_matrix = metrics['confusion_matrix']
    has_proba = metrics['has_proba']
    pos_label = metrics['pos_label']
    y_test = metrics['y_test']
    
    # Asignar nombres de clase si no se proporcionan
    if class_names is None or len(class_names) != n_classes:
        class_names = ["Negativa", "Positiva"] if is_binary else [f"Clase {c}" for c in classes]
    
    # No hacer nada si no hay nada que mostrar
    if not (show_plots or save_dir):
        return
    
    # VISUALIZACIONES
    if subplots and has_proba:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        # (a) Matriz de confusión
        plot_confusion_matrix(conf_matrix, class_names=class_names, ax=axes[0])
        axes[0].set_title('Matriz de Confusión')
        
        # (b) Curva ROC y (c) Curva PR
        if is_binary:
            # Extraer datos de curvas binarias
            fpr = metrics['roc']['fpr']
            tpr = metrics['roc']['tpr']
            roc_auc_val = metrics['roc']['auc']
            precision_vals = metrics['pr']['precision']
            recall_vals = metrics['pr']['recall']
            pr_auc_val = metrics['pr']['average_precision']
            
            # Graficar ROC
            axes[1].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
            axes[1].plot([0,1],[0,1],'k--',lw=2)
            axes[1].set_xlabel('Tasa de Falsos Positivos')
            axes[1].set_ylabel('Tasa de Verdaderos Positivos')
            axes[1].set_title('Curva ROC')
            axes[1].legend(loc="lower right")
            
            # Graficar PR
            axes[2].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Curva Precision-Recall')
            axes[2].legend(loc="lower left")
        else:
            # Extraer datos de curvas multiclase
            fpr_dict = metrics['roc']['fpr']
            tpr_dict = metrics['roc']['tpr']
            roc_auc_scores = metrics['roc']['auc_scores']
            prec_dict = metrics['pr']['precision']
            rec_dict = metrics['pr']['recall']
            pr_auc_scores = metrics['pr']['ap_scores']
            
            # Configurar mapa de colores
            colors = plt.cm.get_cmap('tab10', n_classes)
            
            # Graficar curvas ROC para cada clase
            for i, cls in enumerate(classes):
                axes[1].plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                             label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
            axes[1].plot([0,1],[0,1],'k--',lw=1)
            axes[1].set_xlabel('Tasa de Falsos Positivos')
            axes[1].set_ylabel('Tasa de Verdaderos Positivos')
            axes[1].set_title('Curvas ROC')
            axes[1].legend(loc="lower right", fontsize='small')
            
            # Graficar curvas PR para cada clase
            for i, cls in enumerate(classes):
                axes[2].plot(rec_dict[i], prec_dict[i], lw=2, color=colors(i),
                             label=f'{class_names[i]} (AUC = {pr_auc_scores[i]:.2f})')
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Curvas Precision-Recall')
            axes[2].legend(loc="lower left", fontsize='small')
            
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
                # Usar datos ya calculados para las curvas binarias
                y_pred_prob = metrics['y_pred_prob']
                plot_and_save_binary_curves(y_test, y_pred_prob, pos_label, save_dir, base_filename, show_plots, figsize, subplots=False)
            else:
                # Usar datos ya calculados para las curvas multiclase
                y_pred_prob = metrics['y_pred_prob']
                plot_and_save_multiclass_curves(y_test, y_pred_prob, classes, class_names, save_dir, base_filename, show_plots, figsize, subplots=False)


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
        axes[0].set_xlabel('TFP'); axes[0].set_ylabel('TVP'); axes[0].set_title('Curva ROC'); axes[0].legend(loc="lower right")
        axes[1].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Curva PR'); axes[1].legend(loc="lower left")
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_curves", show_plots)
    else:
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=2)
        plt.xlabel('TFP'); plt.ylabel('TVP'); plt.title('Curva ROC'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_roc_curve", show_plots)
        plt.figure(figsize=figsize)
        plt.plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curva PR'); plt.legend(loc="lower left")
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
        ax1.set_xlabel('TFP'); ax1.set_ylabel('TVP'); ax1.set_title('Curvas ROC'); ax1.legend(loc="lower right", fontsize='small')
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Curvas PR'); ax2.legend(loc="lower left", fontsize='small')
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_multiclass_curves", show_plots)
    else:
        colors = plt.cm.get_cmap('tab10', n_classes)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.xlabel('TFP'); plt.ylabel('TVP'); plt.title('Curvas ROC'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_roc", show_plots)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(rec_dict[i], prec_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {pr_auc_scores[i]:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curvas PR'); plt.legend(loc="lower left")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_pr", show_plots)
    return avg_roc_auc, avg_pr_auc


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