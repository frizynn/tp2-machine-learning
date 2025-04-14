import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from IPython.display import display
from matplotlib import rc
from typing import List, Optional, Dict
import numpy as np

from evaluation.metrics import compute_binary_curves, compute_multiclass_curves

# ---------------------- Helper Functions ----------------------
def _save_plot(save_path: Optional[str], dpi: int = 300):
    """
    Saves the current matplotlib figure to the specified path.
    
    This helper function handles the saving of plots by creating necessary directories
    and saving the figure with specified resolution. It provides feedback about the
    save location.
    
    Parameters
    ----------
    save_path : Optional[str]
        Path where the figure should be saved. If None, no saving is performed.
    dpi : int, default=300
        Resolution in dots per inch for the saved figure.
        
    Returns
    -------
    None
    
    Examples
    --------
    >>> _save_plot("output/plots/my_plot.png")
    Gráfico guardado en: output/plots/my_plot.png
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
def _set_font_sizes(title_fs: int, label_fs: int, tick_fs: int, legend_fs: int):
    """
    Sets global font sizes for matplotlib plots.
    
    This helper function updates the matplotlib rcParams to set consistent font sizes
    across all plot elements. It affects titles, axis labels, tick labels, and legend text.
    
    Parameters
    ----------
    title_fs : int
        Font size for plot titles.
    label_fs : int
        Font size for axis labels.
    tick_fs : int
        Font size for tick labels on both axes.
    legend_fs : int
        Font size for legend text.
        
    Returns
    -------
    None
    
    Examples
    --------
    >>> _set_font_sizes(title_fs=16, label_fs=12, tick_fs=10, legend_fs=10)
    """
    plt.rcParams.update({
        'axes.titlesize': title_fs,
        'axes.labelsize': label_fs,
        'xtick.labelsize': tick_fs,
        'ytick.labelsize': tick_fs,
        'legend.fontsize': legend_fs
    })
# ---------------------- Plot Functions ----------------------
def analyze_categorical_variables(df: pd.DataFrame, categorical_columns: List[str],
                                  normalize: bool = True, dpi: int = 300):
    """
    Display the distribution of categorical variables as a table.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to analyze.
    categorical_columns : list
        List of categorical column names.
    normalize : bool, default True
        Whether to show percentages along with counts.
    dpi : int, default 300
        Resolution for saving figures (if needed).
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
    Plot the distribution (histogram with KDE) of numerical variables.
    If target_column is specified, data is separated by that variable.
    """
    plot_cols = features_to_plot if features_to_plot is not None else numerical_cols
    plot_cols = [col for col in plot_cols if col in numerical_cols]
    if features_to_plot and len(plot_cols) != len(features_to_plot):
        print("Warning: Some features in features_to_plot are not in numerical_cols.")
        if not plot_cols:
            print("Error: No valid features to plot.")
            return None

    plt.figure(figsize=figsize)
    n_cols = len(plot_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row

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
    Creates a correlation heatmap visualization for specified columns in a DataFrame.
    
    This function generates a heatmap showing the correlation coefficients between
    selected numerical columns. The visualization includes options for customization
    of appearance, font settings, and saving capabilities.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to analyze.
    columns : List[str]
        List of column names to include in the correlation analysis.
    output_dir : Optional[str], default=None
        Directory path where to save the plot. If None, plot is not saved.
    filename : Optional[str], default=None
        Name of the file to save the plot as. Required if output_dir is specified.
    figsize : tuple, default=(12, 10)
        Figure size as (width, height) in inches.
    annot : bool, default=True
        Whether to display correlation values on the heatmap.
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap.
    fmt : str, default=".2f"
        Format string for the correlation values.
    title : Optional[str], default=None
        Title for the plot. If None, uses default title.
    use_serif_font : bool, default=True
        Whether to use serif font for text elements.
    title_fontsize : int, default=16
        Font size for the title.
    label_fontsize : int, default=12
        Font size for axis labels.
    tick_fontsize : int, default=10
        Font size for tick labels.
    annot_fontsize : int, default=8
        Font size for correlation value annotations.
    cbar_fontsize : int, default=10
        Font size for colorbar label.
    dpi : int, default=300
        Dots per inch for saved figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> fig = plot_correlation_heatmap(df, ['A', 'B', 'C'], title='Correlation Analysis')
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
                         dpi: int = 300, title: Optional[str] = None) -> plt.Figure:
    """
    Generate a boxplot to visualize and detect outliers in a specific column of a DataFrame.
    
    This function creates a boxplot visualization that helps identify potential outliers
    in the specified column. The boxplot shows the distribution of the data, including
    the median, quartiles, and any potential outliers beyond the whiskers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to visualize.
    column : str
        Name of the column to analyze for outliers.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    save_path : Optional[str], default=None
        Path where to save the generated plot. If None, the plot is not saved.
    show_plot : bool, default=True
        Whether to display the plot using plt.show().
    dpi : int, default=300
        Dots per inch for the saved figure.
    title : Optional[str], default=None
        Custom title for the plot. If None, a default title is generated.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object containing the boxplot.
        
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
    >>> fig = plot_outlier_boxplot(df, 'values', title='Outlier Analysis')
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
    Generate boxplots for analyzing outliers in multiple features of a DataFrame.
    
    This function creates boxplot visualizations to identify potential outliers in specified
    numeric columns of a DataFrame. It supports both single plots and subplot arrangements,
    with options for saving the plots and customizing their appearance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to analyze.
    columns : Optional[List[str]], default=None
        List of column names to consider for analysis. If None, all numeric columns are used.
    features_to_plot : Optional[List[str]], default=None
        Specific features to plot. If None, all numeric columns are plotted.
    save_dir : Optional[str], default=None
        Directory path where to save the generated plots. If None, plots are not saved.
    show_plots : bool, default=True
        Whether to display the plots using plt.show().
    filename : Optional[str], default=None
        Base filename for saving plots. If None, default names are used.
    figsize : tuple, default=(15, 6)
        Figure size as (width, height) in inches.
    dpi : int, default=300
        Dots per inch for the saved figures.
    use_subplots : bool, default=True
        Whether to arrange multiple plots in a grid using subplots.
        
    Returns
    -------
    List[plt.Figure]
        List of generated figure objects containing the boxplots.
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5, 100],
    ...     'feature2': [10, 20, 30, 40, 50, 200]
    ... })
    >>> figures = plot_outliers_analysis(df, features_to_plot=['feature1', 'feature2'])
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    columns_to_analyze = [col for col in (columns if columns else numeric_cols) if col in df.columns and col in numeric_cols]
    features_to_plot = [col for col in (features_to_plot or []) if col in columns_to_analyze]
    if save_dir and features_to_plot:
        os.makedirs(save_dir, exist_ok=True)
    figures = []
    if use_subplots and features_to_plot:
        n_features = len(features_to_plot)
        n_rows = (n_features + 2) // 3  # 3 columns per row
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
    Generates and displays a confusion matrix visualization as a heatmap.

    This function creates a detailed visualization of a confusion matrix, with options for
    normalization, custom class names, and various styling parameters. The matrix can be
    displayed directly or saved to a file.

    Parameters
    ----------
    conf_matrix : np.ndarray
        The confusion matrix to visualize. Should be a square matrix where each element
        represents the count or probability of predictions.
    class_names : Optional[List[str]], default=None
        List of class names to display on the axes. If None, numeric indices are used.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
    cmap : str, default='Blues'
        Colormap to use for the heatmap visualization.
    normalize : bool, default=False
        If True, normalizes the confusion matrix by dividing each row by its sum.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    save_dir : Optional[str], default=None
        Directory to save the plot. If None, the plot is not saved.
    filename : Optional[str], default=None
        Name of the file to save the plot. Only used if save_dir is provided.
    dpi : int, default=300
        Dots per inch for the saved figure.
    ax : Optional[plt.Axes], default=None
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the confusion matrix plot.

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> fig = plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])
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
                    fontsize=14,
                    fontweight='bold')
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
    Plots a Receiver Operating Characteristic (ROC) curve.
    
    This function creates a ROC curve visualization showing the trade-off between
    true positive rate (sensitivity) and false positive rate (1-specificity) for
    different classification thresholds. The curve is accompanied by a diagonal
    reference line representing random guessing.
    
    Parameters
    ----------
    fpr : np.ndarray
        Array of false positive rates.
    tpr : np.ndarray
        Array of true positive rates.
    auc_score : float, optional
        Area Under the Curve score. If provided, will be included in the label.
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height).
    title : str, default='Curva ROC (Característica Operativa del Receptor)'
        Title for the plot.
    save_dir : str, optional
        Directory to save the plot.
    filename : str, optional
        Name of the file to save the plot.
    dpi : int, default=300
        Dots per inch for saved figure.
    label : str, optional
        Custom label for the curve. If None, will use AUC score if available.
    color : str, optional
        Color for the ROC curve.
    linestyle : str, default='-'
        Line style for the ROC curve.
    multiple_curves : bool, default=False
        If True, allows plotting multiple curves on the same axes.
    ax : plt.Axes, optional
        Matplotlib Axes object to plot on. If None, creates a new figure.
        
    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the ROC curve plot.
        
    Examples
    --------
    >>> from sklearn.metrics import roc_curve, auc
    >>> y_true = [0, 1, 0, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8]
    >>> fpr, tpr, _ = roc_curve(y_true, y_scores)
    >>> auc_score = auc(fpr, tpr)
    >>> fig = plot_roc_curve(fpr, tpr, auc_score)
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    curve_label = label if label is not None else (f'AUC = {auc_score:.3f}' if auc_score is not None else None)
    ax.plot(fpr, tpr, color=color, linestyle=linestyle, label=curve_label)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)  # diagonal line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos', fontsize=16)
    ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    if curve_label:
        ax.legend(loc='lower right', frameon=True, fontsize=16)
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
    Plot the Precision-Recall curve.

    Parameters
    ----------
    average_precision : float, optional
        Average precision score. If provided, will be included in the curve label.
    figsize : tuple, default=(10, 8)
        Figure size as (width, height) in inches.
    title : str, default='Curva Precision-Recall'
        Title for the plot.
    save_dir : str, optional
        Directory path to save the plot. If None, plot is not saved.
    filename : str, optional
        Name of the file to save the plot. Required if save_dir is provided.
    dpi : int, default=300
        Dots per inch for saved figure.
    label : str, optional
        Custom label for the curve. If None, will use AP score if available.
    color : str, optional
        Color for the PR curve.
    linestyle : str, default='-'
        Line style for the PR curve.
    multiple_curves : bool, default=False
        If True, allows plotting multiple curves on the same axes.
    ax : plt.Axes, optional
        Matplotlib Axes object to plot on. If None, creates a new figure.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the PR curve plot.

    Examples
    --------
    >>> from sklearn.metrics import precision_recall_curve, average_precision_score
    >>> y_true = [0, 1, 0, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8]
    >>> precision, recall, _ = precision_recall_curve(y_true, y_scores)
    >>> ap = average_precision_score(y_true, y_scores)
    >>> fig = plot_precision_recall_curve(precision, recall, average_precision=ap)
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    curve_label = label if label is not None else (f'AP = {average_precision:.3f}' if average_precision is not None else None)
    ax.plot(recall, precision, color=color, linestyle=linestyle, label=curve_label)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precisión', fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    if curve_label:
        ax.legend(loc='best', frameon=True, fontsize=16)
    if save_dir and filename and not multiple_curves:
        _save_plot(os.path.join(save_dir, filename), dpi)
    if not multiple_curves:
        return plt.gcf()
    return ax

def plot_lambda_tuning(lambda_values: List[float], scores: List[float],
                       metric_name: str = "Metric", save_dir: Optional[str] = None,
                       filename: Optional[str] = None, best_lambda: Optional[float] = None,
                       figsize: tuple = (10, 6), dpi: int = 300, log_scale: bool = True) -> None:
    """
    Plots lambda search results using cross-validation.
    
    Parameters:
    -----------
    lambda_values : list
        List of lambda values tested.
    scores : list
        Corresponding scores for each lambda value.
    metric_name : str, default "Metric"
        Name of the metric used for evaluation.
    save_dir : str, optional
        Directory to save the plot.
    filename : str, optional
        Filename for saving the plot.
    best_lambda : float, optional
        Best lambda value found.
    figsize : tuple, default (10, 6)
        Figure size (width, height).
    dpi : int, default 300
        Resolution for saving figures.
    log_scale : bool, default True
        Whether to use logarithmic scale for x-axis.
    """
    plt.figure(figsize=figsize)
    plt.plot(lambda_values, scores, 'o-', linewidth=2, markersize=8)
    if best_lambda is not None:
        best_idx = np.argmin(np.abs(np.array(lambda_values) - best_lambda))
        best_score = scores[best_idx]
        plt.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7)
        plt.plot(best_lambda, best_score, 'ro', markersize=10, label=f'Best λ = {best_lambda}')
        plt.legend(fontsize=16)
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
                            show_plot: bool = False, subplots: bool = True, figsize=(18, 8),
                            title_fontsize=24, label_fontsize=20, tick_fontsize=18, legend_fontsize=16):
    """
    Generates comparative plots (ROC and PR curves) for multiple models.
    If curve data is not available, generates a bar chart with AUC values.
    
    Parameters:
    -----------
    all_models : dict
        Dictionary with models and associated metrics
    output_dir : str
        Directory to save the plots
    prefix : str, default=""
        Prefix for generated files
    show_plot : bool, default=False
        If True, displays the plots
    subplots : bool, default=True
        If True, combines plots in subplots
    figsize : tuple, default=(18, 8)
        Figure size (width, height)
    title_fontsize : int, default=24
        Font size for titles
    label_fontsize : int, default=20
        Font size for axis labels
    tick_fontsize : int, default=18
        Font size for ticks
    legend_fontsize : int, default=16
        Font size for legends
        
    Returns:
    --------
    str or tuple
        Path(s) to the saved plot(s)
    """
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    
    # check if at least one model has curve data (dictionary with "fpr")
    can_plot_curves = any(
        isinstance(model_data.get("metrics", {}).get("roc"), dict) and 
        "fpr" in model_data["metrics"]["roc"]
        for model_data in all_models.values()
    )
    
    if not can_plot_curves:
        print("Cannot generate comparative plots because metrics don't contain curve data.")
        print("Only numerical values of AUC-ROC and AUC-PR are available.")
        plt.figure(figsize=figsize)
        model_names = list(all_models.keys())
        roc_values = [model_data["metrics"]["roc"] for model_data in all_models.values()]
        pr_values = [model_data["metrics"]["pr"] for model_data in all_models.values()]
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, roc_values, width, label='AUC-ROC')
        plt.bar(x + width/2, pr_values, width, label='AUC-PR')
        plt.xlabel('Models', fontsize=label_fontsize)
        plt.ylabel('AUC Value', fontsize=label_fontsize)
        plt.title('AUC-ROC and AUC-PR Comparison Between Models', fontsize=title_fontsize)
        plt.xticks(x, model_names, rotation=45, ha='right', fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        
        bar_path = os.path.join(output_dir, f"{prefix}auc_comparison.png")
        _save_plot(bar_path, 300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return bar_path

    if subplots:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
        # ROC curves
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data.get("metrics", {}).get("roc", {})
            if isinstance(roc_data, dict) and "auc" in roc_data:
                plot_roc_curve(roc_data["fpr"], roc_data["tpr"], auc_score=roc_data["auc"],
                               label=f"{name} (AUC = {roc_data['auc']:.4f})",
                               color=colors[i % len(colors)],
                               linestyle=linestyles[i % len(linestyles)],
                               multiple_curves=True, ax=ax_roc)
        ax_roc.set_title("Curvas ROC para diferentes técnicas de reequilibrio", fontsize=title_fontsize)
        ax_roc.set_xlabel('Tasa de Falsos Positivos', fontsize=label_fontsize)
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos', fontsize=label_fontsize)
        ax_roc.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_roc.legend(loc="lower right", fontsize=legend_fontsize)
        
        # PR curves
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data.get("metrics", {}).get("pr", {})
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                plot_precision_recall_curve(pr_data["precision"], pr_data["recall"],
                                            average_precision=pr_data["average_precision"],
                                            label=f"{name} (AP = {pr_data['average_precision']:.4f})",
                                            color=colors[i % len(colors)],
                                            linestyle=linestyles[i % len(linestyles)],
                                            multiple_curves=True, ax=ax_pr)
        ax_pr.set_title("Curvas Precision-Recall para diferentes técnicas de reequilibrio", fontsize=title_fontsize)
        ax_pr.set_xlabel('Recall', fontsize=label_fontsize)
        ax_pr.set_ylabel('Precision', fontsize=label_fontsize)
        ax_pr.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_pr.legend(loc="lower left", fontsize=legend_fontsize)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, f"{prefix}combined_curves_comparison.png")
        _save_plot(combined_path, 300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return combined_path
    else:
        # individual values for separate plot sizes
        single_figsize = (figsize[0]//2, figsize[1])
        
        # separate plots for ROC and PR
        fig_roc, ax_roc = plt.subplots(figsize=single_figsize)
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data.get("metrics", {}).get("roc", {})
            if isinstance(roc_data, dict) and "auc" in roc_data:
                plot_roc_curve(roc_data["fpr"], roc_data["tpr"], auc_score=roc_data["auc"],
                               label=f"{name} (AUC = {roc_data['auc']:.4f})",
                               color=colors[i % len(colors)],
                               linestyle=linestyles[i % len(linestyles)],
                               multiple_curves=True, ax=ax_roc)
        ax_roc.set_title("Curvas ROC para diferentes técnicas de reequilibrio", fontsize=title_fontsize)
        ax_roc.set_xlabel('Tasa de Falsos Positivos', fontsize=label_fontsize)
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos', fontsize=label_fontsize)
        ax_roc.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_roc.legend(loc="lower right", fontsize=legend_fontsize)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, f"{prefix}roc_curves_comparison.png")
        _save_plot(roc_path, 300)
        if not show_plot:
            plt.close()
        
        fig_pr, ax_pr = plt.subplots(figsize=single_figsize)
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data.get("metrics", {}).get("pr", {})
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                plot_precision_recall_curve(pr_data["precision"], pr_data["recall"],
                                            average_precision=pr_data["average_precision"],
                                            label=f"{name} (AP = {pr_data['average_precision']:.4f})",
                                            color=colors[i % len(colors)],
                                            linestyle=linestyles[i % len(linestyles)],
                                            multiple_curves=True, ax=ax_pr)
        ax_pr.set_title("Curvas Precision-Recall para diferentes técnicas de reequilibrio", fontsize=title_fontsize)
        ax_pr.set_xlabel('Recall', fontsize=label_fontsize)
        ax_pr.set_ylabel('Precision', fontsize=label_fontsize)
        ax_pr.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_pr.legend(loc="lower left", fontsize=legend_fontsize)
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
    Generates visualizations for the evaluation of a classification model using
    previously calculated metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with metrics calculated by get_model_metrics
    class_names : list, optional
        Class names for display in plots
    figsize : tuple, default=(16, 5)
        Figure size
    save_dir : str, optional
        Directory to save the plots
    base_filename : str, optional
        Base name for saved files
    show_plots : bool, default=True
        If True, displays the plots
    subplots : bool, default=True
        If True, combines plots in subplots
    """
    # extract needed data from metrics dictionary
    classes = metrics['classes']
    n_classes = len(classes)
    is_binary = metrics['is_binary']
    conf_matrix = metrics['confusion_matrix']
    has_proba = metrics['has_proba']
    pos_label = metrics['pos_label']
    y_test = metrics['y_test']
    
    # assign class names if not provided
    if class_names is None or len(class_names) != n_classes:
        class_names = ["Negative", "Positive"] if is_binary else [f"Class {c}" for c in classes]
    
    # do nothing if there's nothing to show
    if not (show_plots or save_dir):
        return
    
    # VISUALIZATIONS
    if subplots and has_proba:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        # (a) Confusion Matrix
        plot_confusion_matrix(conf_matrix, class_names=class_names, ax=axes[0])
        axes[0].set_title('Matriz de Confusión')
        
        # (b) ROC Curve and (c) PR Curve
        if is_binary:
            # extract binary curve data
            fpr = metrics['roc']['fpr']
            tpr = metrics['roc']['tpr']
            roc_auc_val = metrics['roc']['auc']
            precision_vals = metrics['pr']['precision']
            recall_vals = metrics['pr']['recall']
            pr_auc_val = metrics['pr']['average_precision']
            
            # plot ROC
            axes[1].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
            axes[1].plot([0,1],[0,1],'k--',lw=2)
            axes[1].set_xlabel('Tasa de Falsos Positivos')
            axes[1].set_ylabel('Tasa de Verdaderos Positivos')
            axes[1].set_title('Curva ROC')
            axes[1].legend(loc="lower right")
            
            # plot PR
            axes[2].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Curva Precision-Recall')
            axes[2].legend(loc="lower left")
        else:
            # extract multiclass curve data
            fpr_dict = metrics['roc']['fpr']
            tpr_dict = metrics['roc']['tpr']
            roc_auc_scores = metrics['roc']['auc_scores']
            prec_dict = metrics['pr']['precision']
            rec_dict = metrics['pr']['recall']
            pr_auc_scores = metrics['pr']['ap_scores']
            
            # set color map
            colors = plt.cm.get_cmap('tab10', n_classes)
            
            # plot ROC curves for each class
            for i, cls in enumerate(classes):
                axes[1].plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                             label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
            axes[1].plot([0,1],[0,1],'k--',lw=1)
            axes[1].set_xlabel('Tasa de Falsos Positivos')
            axes[1].set_ylabel('Tasa de Verdaderos Positivos')
            axes[1].set_title('Curvas ROC')
            axes[1].legend(loc="lower right", fontsize='small')
            
            # plot PR curves for each class
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
        # separate plots: first confusion matrix
        plt.figure(figsize=figsize)
        plot_confusion_matrix(conf_matrix, class_names=class_names)
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_confusion_matrix", show_plots)
        
        # if there are probabilities, plot curves
        if has_proba:
            if is_binary:
                # use already calculated data for binary curves
                y_pred_prob = metrics['y_pred_prob']
                plot_and_save_binary_curves(y_test, y_pred_prob, pos_label, save_dir, base_filename, show_plots, figsize, subplots=False)
            else:
                # use already calculated data for multiclass curves
                y_pred_prob = metrics['y_pred_prob']
                plot_and_save_multiclass_curves(y_test, y_pred_prob, classes, class_names, save_dir, base_filename, show_plots, figsize, subplots=False)


def plot_and_save_binary_curves(y_test, y_pred_prob, pos_label, save_dir, base_filename, show_plots, figsize, subplots=True):
    """
    Helper function to plot and (optionally) save ROC and PR curves for binary classification.
    Reuses compute_binary_curves.
    
    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred_prob : array-like
        Predicted probabilities
    pos_label : int
        Positive class label
    save_dir : str, optional
        Directory to save plots
    base_filename : str, optional
        Base name for saved files
    show_plots : bool
        Whether to display plots
    figsize : tuple
        Figure size
    subplots : bool, default=True
        Whether to use subplots for ROC and PR curves
    """
    fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val = compute_binary_curves(y_test, y_pred_prob[:, 1], pos_label)
    if subplots:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        axes[0].plot([0,1], [0,1], 'k--', lw=2)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('Curva ROC'); axes[0].legend(loc="lower right")
        axes[1].plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Curva Precision-Recall'); axes[1].legend(loc="lower left")
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_curves", show_plots)
    else:
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=2)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Curva ROC'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_roc_curve", show_plots)
        plt.figure(figsize=figsize)
        plt.plot(recall_vals, precision_vals, lw=2, label=f'PR (AUC = {pr_auc_val:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curva Precision-Recall'); plt.legend(loc="lower left")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_pr_curve", show_plots)

def plot_and_save_multiclass_curves(y_test, y_pred_prob, classes, class_names, save_dir, base_filename, show_plots, figsize, subplots=True):
    """
    Helper function to plot and (optionally) save ROC and PR curves (One-vs-Rest) for multiclass classification.
    Uses compute_multiclass_curves to calculate curves for each class.
    Returns average AUC for ROC and PR.
    
    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred_prob : array-like
        Predicted probabilities
    classes : list
        List of class labels
    class_names : list
        Names of classes for display
    save_dir : str, optional
        Directory to save plots
    base_filename : str, optional
        Base name for saved files
    show_plots : bool
        Whether to display plots
    figsize : tuple
        Figure size
    subplots : bool, default=True
        Whether to use subplots for ROC and PR curves
        
    Returns:
    --------
    tuple
        Average ROC AUC and PR AUC
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
        ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('Curvas ROC'); ax1.legend(loc="lower right", fontsize='small')
        ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Curvas Precision-Recall'); ax2.legend(loc="lower left", fontsize='small')
        plt.tight_layout()
        save_or_show_plot(fig, save_dir, f"{base_filename}_multiclass_curves", show_plots)
    else:
        colors = plt.cm.get_cmap('tab10', n_classes)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(fpr_dict[i], tpr_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {roc_auc_scores[i]:.2f})')
        plt.plot([0,1], [0,1], 'k--', lw=1)
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Curvas ROC'); plt.legend(loc="lower right")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_roc", show_plots)
        plt.figure(figsize=figsize)
        for i in range(n_classes):
            plt.plot(rec_dict[i], prec_dict[i], lw=2, color=colors(i),
                     label=f'{class_names[i]} (AUC = {pr_auc_scores[i]:.2f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Curvas Precision-Recall'); plt.legend(loc="lower left")
        save_or_show_plot(plt.gcf(), save_dir, f"{base_filename}_multiclass_pr", show_plots)
    return avg_roc_auc, avg_pr_auc


def save_or_show_plot(fig, save_dir, base_filename, show_plot):
    """
    Saves the figure to the specified directory if provided;
    otherwise displays or closes it according to show_plot.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save or display
    save_dir : str, optional
        Directory to save the figure
    base_filename : str, optional
        Base filename for saving
    show_plot : bool
        Whether to display the plot
    """
    if save_dir and base_filename:
        fig.savefig(os.path.join(save_dir, f"{base_filename}.png"), bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_data_analysis_all_visualizations(df, numerical_cols, target_column, features_to_plot, output_dir, fig_output_dir_p1, fig_params={}):
    """
    Creates a complete set of visualizations for data analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    numerical_cols : list
        List of numerical column names
    target_column : str
        Target variable column name
    features_to_plot : list
        Specific features to plot
    output_dir : str
        Main output directory
    fig_output_dir_p1 : str
        Specific output directory for figures
    fig_params : dict, default={}
        Dictionary with parameters for different plots
        
    Returns:
    --------
    tuple
        Generated figure objects
    """
    
    dist_params = fig_params.get("dist_params", {
        "filename": "numerical_distributions_outliers.png",
        "features_to_plot": features_to_plot,
        "tick_fontsize": 18,
        "label_fontsize": 18,
        "title_fontsize": 18,
        "title": " ",
        "figsize": (16, 5)
    })

   
    
    
    # plot numerical distributions
    fig1 = plot_numerical_distributions(
        df,
        numerical_cols,
        target_column,
        output_dir=output_dir,
        **dist_params
    )
    plt.show()

    heatmap_params = fig_params.get("heatmap_params", {
            "label_fontsize": 18,
            "title_fontsize": 18,
            "tick_fontsize": 18,
            "cbar_fontsize": 18,
            "annot_fontsize": 18,
            "figsize": (18, 12),
            "title": " ",
            "filename": "correlation_heatmap_numerical_features_outliers.png"
        })

    # plot correlation heatmap
    fig2 = plot_correlation_heatmap(
        df,
        numerical_cols + [target_column],
        output_dir=fig_output_dir_p1,
        **heatmap_params
    )
    plt.show()


    outlier_params = fig_params.get("outlier_params", {
        "filename": "boxplots_outliers_analysis.png",
        "features_to_plot": features_to_plot,
        "figsize": (16, 5)
    })

    # plot outliers analysis
    fig3 = plot_outliers_analysis(
        df=df,
        save_dir=fig_output_dir_p1,
        **outlier_params
    )
    return fig1, fig2, fig3
