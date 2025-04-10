import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from IPython.display import display, HTML
from matplotlib import rc
from typing import List, Optional, Dict, Union
import numpy as np




def analyze_categorical_variables(df, categorical_columns, normalize=True, dpi=300):
    """
    Display distribution of categorical variables in a dataframe as tables in Jupyter.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    categorical_columns : list
        List of categorical column names to analyze
    normalize : bool, default=True
        Whether to display percentages along with counts
    dpi : int, default=300
        DPI for saving figures
    """
    
    print("Categorical variables distribution:")
    
    for col in categorical_columns:
        print(f"\n{col} distribution:")
        
        # Crear una tabla única con conteo y porcentaje
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True).mul(100).round(2)
        
        # Combinar en un solo DataFrame
        distribution_df = pd.DataFrame({
            col: counts.index,
            'Count': counts.values,
            'Percentage (%)': percentages.values
        })
        
        # Mostrar la tabla combinada
        display(distribution_df)
  

def plot_numerical_distributions(df, numerical_cols, target_column=None, bins=20, 
                               figsize=(16, 12), output_dir=None, filename=None, 
                               title=None, title_fontsize=16, label_fontsize=12, 
                               tick_fontsize=10, legend_fontsize=10, dpi=300,
                               features_to_plot=None):
    """
    Plot the distribution of numerical features with KDE.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with features to plot
    numerical_cols : list
        List of numerical column names to plot
    target_column : str, optional
        If provided, will color the distributions by target values
    bins : int, default=20
        Number of bins for histograms
    figsize : tuple, default=(16, 12)
        Size of the figure
    output_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    title : str, optional
        Main title for the entire figure
    title_fontsize : int, default=16
        Font size for subplot titles
    label_fontsize : int, default=12
        Font size for axis labels
    tick_fontsize : int, default=10
        Font size for tick labels
    legend_fontsize : int, default=10
        Font size for legends
    dpi : int, default=300
        DPI for saving figures
    features_to_plot : list, optional
        Subset of numerical_cols to plot. If None, all columns in numerical_cols will be plotted.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Si features_to_plot está especificado, usarlo como lista de columnas a graficar
    # asegurándose de que sean un subconjunto de numerical_cols
    if features_to_plot is not None:
        # Verificar que features_to_plot sea un subconjunto de numerical_cols
        plot_cols = [col for col in features_to_plot if col in numerical_cols]
        if len(plot_cols) != len(features_to_plot):
            print("Advertencia: Algunas características en features_to_plot no están en numerical_cols.")
        if not plot_cols:
            print("Error: No hay características válidas para graficar.")
            return None
    else:
        plot_cols = numerical_cols
    
    plt.figure(figsize=figsize)
    n_cols = len(plot_cols)
    n_rows = (n_cols + 2) // 3  # Calculate rows needed (3 columns per row)
    
    # Set global font sizes
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize
    })
    
    for i, col in enumerate(plot_cols):
        plt.subplot(n_rows, 3, i+1)
        if target_column:
            sns.histplot(data=df, x=col, hue=target_column, bins=bins, kde=True)
        else:
            sns.histplot(data=df, x=col, bins=bins, kde=True)
        plt.title(f'Distribución de {col}')
    
    plt.tight_layout()
    
    # Add main title if provided
    if title:
        plt.suptitle(title, fontsize=title_fontsize+4, y=1.02)
        plt.subplots_adjust(top=0.9)
    
    # Save the figure if output directory and filename are provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        figure_path = os.path.join(output_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {figure_path}")
    
    return plt.gcf()  # Return the figure for potential saving

def plot_correlation_heatmap(df, columns, output_dir=None, filename=None, figsize=(12, 10), 
                              annot=True, cmap='coolwarm', fmt=".2f", title=None, use_serif_font=True,
                              title_fontsize=16, label_fontsize=12, tick_fontsize=10, annot_fontsize=8, 
                              cbar_fontsize=10, dpi=300):
    """
    Create and optionally save a correlation heatmap for selected columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with features to analyze
    columns : list
        List of column names to include in correlation analysis
    output_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    figsize : tuple, default=(12, 10)
        Size of the figure
    annot : bool, default=True
        Whether to annotate cells with correlation values
    cmap : str, default='coolwarm'
        Colormap for the heatmap
    fmt : str, default=".2f"
        Format string for annotations
    title : str, optional
        Title for the heatmap
    use_serif_font : bool, default=True
        Whether to use serif font for the plot
    title_fontsize : int, default=16
        Font size for title
    label_fontsize : int, default=12
        Font size for axis labels
    tick_fontsize : int, default=10
        Font size for tick labels
    annot_fontsize : int, default=8
        Font size for annotations in the heatmap
    cbar_fontsize : int, default=10
        Font size for the colorbar label
    dpi : int, default=300
        DPI for saving figures
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if use_serif_font:
        rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
    
    plt.figure(figsize=figsize)
    correlation_matrix = df[columns].corr()
    
    # Set font sizes
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize
    })
    
    # Fix the colorbar keyword arguments - remove 'labelsize' and set label separately
    cbar_kws = {'label': r'Coeficiente de Correlación ($\rho$)'}
    
    # Set annotation font size explicitly
    annot_kws = {"size": annot_fontsize}
    
    heatmap = sns.heatmap(correlation_matrix,
                annot=annot,
                cmap=cmap,
                fmt=fmt,
                linewidths=0.5,
                annot_kws=annot_kws,
                cbar_kws=cbar_kws)
    
    # Set the colorbar label font size after creating the heatmap
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(cbar_fontsize)
    
    if title:
        plt.title(title, fontsize=title_fontsize)
    else:
        plt.title('Mapa de Calor de Correlación', fontsize=title_fontsize)
    
    plt.tight_layout()
    
    # Save the figure if output directory and filename are provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        figure_path = os.path.join(output_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {figure_path}")
    
    return plt.gcf()  # Return the figure
def plot_categorical_distributions(df, categorical_cols, target_column=None, 
                                 figsize=(16, 12), output_dir=None, filename=None,
                                 title=None, title_fontsize=16, label_fontsize=12, 
                                 tick_fontsize=10, legend_fontsize=10, dpi=300):
    """
    Create bar plots for categorical variables, optionally grouped by target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with features to plot
    categorical_cols : list
        List of categorical column names to plot
    target_column : str, optional
        If provided, will color the bars by target values
    figsize : tuple, default=(16, 12)
        Size of the figure
    output_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    title : str, optional
        Main title for the entire figure
    title_fontsize : int, default=16
        Font size for subplot titles
    label_fontsize : int, default=12
        Font size for axis labels
    tick_fontsize : int, default=10
        Font size for tick labels
    legend_fontsize : int, default=10
        Font size for legends
    dpi : int, default=300
        DPI for saving figures
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 1) // 2  # Calculate rows needed (2 columns per row)
    
    # Set global font sizes
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize
    })
    
    for i, col in enumerate(categorical_cols):
        plt.subplot(n_rows, 2, i+1)
        if target_column:
            # Create a grouped bar plot
            contingency = pd.crosstab(df[col], df[target_column])
            contingency.plot(kind='bar', stacked=False, ax=plt.gca())
        else:
            # Create a simple count plot
            sns.countplot(data=df, x=col)
        plt.title(f'Distribución de {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Add main title if provided
    if title:
        plt.suptitle(title, fontsize=title_fontsize+4, y=1.02)
        plt.subplots_adjust(top=0.9)
    
    # Save the figure if output directory and filename are provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        figure_path = os.path.join(output_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {figure_path}")
    
    return plt.gcf()  # Return the figure

def display_feature_ranges(df, columns, output_dir=None, filename=None):
    """
    Print the range (min to max) of each feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with features to analyze
    columns : list
        List of column names to display ranges for
    output_dir : str, optional
        Directory path where to save the results as text file
    filename : str, optional
        Filename for saving the results
    """
    results = ["\nFeature ranges:"]
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        result = f"{col}: {min_val} to {max_val}"
        results.append(result)
        print(result)
    
    # Save results if output directory and filename are provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w') as f:
            for line in results:
                f.write(line + '\n')
        print(f"Saved feature ranges to {file_path}")


def set_plot_style(style='whitegrid', context='notebook', palette='deep', font_scale=1.2):
    """
    Set the global plotting style for seaborn.
    
    Parameters:
    -----------
    style : str, default='whitegrid'
        The seaborn style to use
    context : str, default='notebook'
        The seaborn context (paper, notebook, talk, poster)
    palette : str, default='deep'
        The seaborn color palette
    font_scale : float, default=1.2
        Scaling factor for font sizes
    """
    sns.set_theme(style=style, context=context, palette=palette, font_scale=font_scale)
    print(f"Plot style set to: {style}, context: {context}, palette: {palette}")

def plot_missing_values(df, figsize=(10, 6), output_dir=None, filename=None, 
                      title="Missing Values Heatmap", title_fontsize=16, 
                      label_fontsize=12, tick_fontsize=10, dpi=300):
    """
    Create a heatmap showing missing values in a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
    figsize : tuple, default=(10, 6)
        Size of the figure
    output_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    title : str, default="Missing Values Heatmap"
        Title for the plot
    title_fontsize : int, default=16
        Font size for title
    label_fontsize : int, default=12
        Font size for axis labels
    tick_fontsize : int, default=10
        Font size for tick labels
    dpi : int, default=300
        DPI for saving figures
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    plt.figure(figsize=figsize)
    
    # Set font sizes
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize
    })
    
    # Create a binary heatmap of missing values
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Columns', fontsize=label_fontsize)
    plt.ylabel('Rows', fontsize=label_fontsize)
    
    plt.tight_layout()
    
    # Save the figure if output directory and filename are provided
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        figure_path = os.path.join(output_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {figure_path}")
    
    return plt.gcf()  # Return the figure

def plot_outlier_boxplot(df: pd.DataFrame, column: str, figsize: tuple = (10, 6),
                        save_path: Optional[str] = None, show_plot: bool = True, 
                        dpi: int = 300, title: Optional[str] = None):
    """
    Visualiza los outliers de una sola columna usando un boxplot.
    
    Args:
        df: DataFrame de pandas
        column: Nombre de la columna a visualizar
        figsize: Tamaño de la figura
        save_path: Ruta donde guardar el gráfico. Si es None, no se guarda.
        show_plot: Si es True, muestra el gráfico en pantalla. Si es False, solo lo guarda.
        dpi: Resolución del gráfico guardado.
        title: Título del gráfico. Si es None, se usa 'Boxplot de {column}'
        
    Returns:
        matplotlib.figure.Figure: La figura generada
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column])
    
    if title is None:
        title = f'Boxplot de {column}'
    plt.title(title)
    plt.tight_layout()
    
    # Guardar el gráfico si se especificó una ruta
    if save_path is not None:
        # Crear el directorio si no existe
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    # Mostrar el gráfico solo si show_plot es True
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return plt.gcf()


def plot_outliers_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          features_to_plot: Optional[List[str]] = None,
                          save_dir: Optional[str] = None, show_plots: bool = True,
                          filename: Optional[str] = None,
                          figsize: tuple = (15, 6), dpi: int = 300,
                          use_subplots: bool = True) -> List[plt.Figure]:
    """
    Visualiza boxplots para un conjunto de características con outliers.
    
    Args:
        df: DataFrame de pandas a analizar
        columns: Lista de columnas a analizar. Si es None, se usan todas las columnas numéricas.
        features_to_plot: Lista específica de columnas para generar boxplots individuales.
                         Si es None, no se generan boxplots individuales.
        save_dir: Directorio donde guardar los gráficos. Si es None, no se guardan.
        show_plots: Si es True, muestra los gráficos en pantalla.
        filename: Nombre base para los archivos guardados. Si es None, se usan nombres predeterminados.
        figsize: Tamaño de las figuras individuales
        dpi: Resolución de los gráficos guardados
        use_subplots: Si es True, muestra todos los boxplots en una sola figura con subplots.
                     Si es False, genera figuras individuales para cada característica.
        
    Returns:
        List[plt.Figure]: Lista de las figuras generadas
    """
    # Obtener las columnas a analizar
    if columns is None:
        columns_to_analyze = df.select_dtypes(include=['number']).columns.tolist()
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns
        columns_to_analyze = [col for col in columns if col in df.columns and col in numeric_cols]
        
    # Determinar qué columnas visualizar con boxplots individuales
    if features_to_plot is None:
        features_to_plot = []
    else:
        features_to_plot = [col for col in features_to_plot if col in columns_to_analyze]

    # Crear directorio para guardar gráficos si es necesario
    if save_dir is not None and features_to_plot:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = []
    
    # Si use_subplots es True, crear una figura con subplots
    if use_subplots and features_to_plot:
        n_features = len(features_to_plot)
        n_rows = (n_features + 2) // 3  # Calcular número de filas (3 columnas)
        fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], figsize[1] * n_rows / 2))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, column in enumerate(features_to_plot):
            if i < len(axes):
                sns.boxplot(x=df[column], ax=axes[i])
                axes[i].set_title(f'Boxplot de {column}')
        
        # Ocultar ejes no utilizados
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        
        # Guardar la figura con subplots si se especificó un directorio
        if save_dir is not None:
            file_name = filename if filename is not None else 'boxplots_all_features.png'
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        figures.append(fig)
    
    # Si use_subplots es False o no hay features_to_plot, generar boxplots individuales
    elif not use_subplots and features_to_plot:
        for column in features_to_plot:
            save_path = None
            if save_dir is not None:
                if filename is not None:
                    # Usar el nombre base y añadir el nombre de la columna
                    base_name, ext = os.path.splitext(filename)
                    if not ext:  # Si no hay extensión, añadir .png
                        ext = '.png'
                    save_path = os.path.join(save_dir, f'{base_name}_{column}{ext}')
                else:
                    save_path = os.path.join(save_dir, f'boxplot_{column}.png')
            
            fig = plot_outlier_boxplot(
                df=df,
                column=column,
                figsize=figsize,
                save_path=save_path,
                show_plot=show_plots,
                dpi=dpi
            )
            figures.append(fig)
    
    return figures

def plot_confusion_matrix(conf_matrix, class_names=None, figsize=(10, 8), 
                         cmap='Blues', normalize=False, title=None,
                         save_dir=None, filename=None, dpi=300, ax=None):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        The confusion matrix to plot
    class_names : list, optional
        List of class names for axis labels
    figsize : tuple, default=(10, 8)
        Size of the figure
    cmap : str, default='Blues'
        Colormap to use
    normalize : bool, default=False
        If True, normalize the confusion matrix
    title : str, optional
        Title for the plot
    save_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    dpi : int, default=300
        DPI for saving figures
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create a new figure if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    # Plot the confusion matrix
    cax = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    
    if title:
        ax.set_title(title, fontsize=16)
    
    # Add colorbar
    fig = ax.figure
    fig.colorbar(cax, ax=ax)
    
    # Add axis labels
    tick_marks = np.arange(len(conf_matrix))
    if class_names is not None:
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
    else:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black")
    
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    
    # Save the figure if output directory and filename are provided
    if save_dir and filename and ax.figure is not None:
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved confusion matrix to {figure_path}")
    
    return ax.figure  # Return the figure for potential saving


def plot_roc_curve(fpr, tpr, auc_score=None, figsize=(10, 8), 
                  title='Receiver Operating Characteristic (ROC) Curve',
                  save_dir=None, filename=None, dpi=300,
                  label=None, color=None, linestyle='-',
                  multiple_curves=False, ax=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates
    auc_score : float, optional
        Area under the ROC curve
    figsize : tuple, default=(10, 8)
        Size of the figure
    title : str, default='Receiver Operating Characteristic (ROC) Curve'
        Title for the plot
    save_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    dpi : int, default=300
        DPI for saving figures
    label : str, optional
        Label for the curve in the legend
    color : str, optional
        Color for the curve
    linestyle : str, default='-'
        Line style for the curve
    multiple_curves : bool, default=False
        If True, don't create a new figure (used for plotting multiple curves)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plot the curve
    curve_label = label
    if curve_label is None and auc_score is not None:
        curve_label = f'AUC = {auc_score:.3f}'
    
    ax.plot(fpr, tpr, 
           color=color, 
           linestyle=linestyle,
           label=curve_label)
    
    # Plot the diagonal (random classifier line)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    
    # Set limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=16)
    
    ax.grid(True, alpha=0.3)
    
    if curve_label:
        ax.legend(loc='lower right', frameon=True, fontsize=10)
    
    # Save the figure if output directory and filename are provided
    if save_dir and filename and not multiple_curves:
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved ROC curve to {figure_path}")
    
    if not multiple_curves:
        return plt.gcf()  # Return the figure for potential saving
    return ax


def plot_precision_recall_curve(precision, recall, average_precision=None, figsize=(10, 8),
                               title='Precision-Recall Curve',
                               save_dir=None, filename=None, dpi=300,
                               label=None, color=None, linestyle='-',
                               multiple_curves=False, ax=None):
    """
    Plot the Precision-Recall curve.
    
    Parameters:
    -----------
    precision : numpy.ndarray
        Precision values
    recall : numpy.ndarray
        Recall values
    average_precision : float, optional
        Average precision score
    figsize : tuple, default=(10, 8)
        Size of the figure
    title : str, default='Precision-Recall Curve'
        Title for the plot
    save_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    dpi : int, default=300
        DPI for saving figures
    label : str, optional
        Label for the curve in the legend
    color : str, optional
        Color for the curve
    linestyle : str, default='-'
        Line style for the curve
    multiple_curves : bool, default=False
        If True, don't create a new figure (used for plotting multiple curves)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if not multiple_curves and ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plot the curve
    curve_label = label
    if curve_label is None and average_precision is not None:
        curve_label = f'AP = {average_precision:.3f}'
    
    ax.plot(recall, precision, 
           color=color,
           linestyle=linestyle,
           label=curve_label)
    
    # Set limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=16)
    
    ax.grid(True, alpha=0.3)
    
    if curve_label:
        ax.legend(loc='best', frameon=True, fontsize=10)
    
    # Save the figure if output directory and filename are provided
    if save_dir and filename and not multiple_curves:
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved precision-recall curve to {figure_path}")
    
    if not multiple_curves:
        return plt.gcf()  # Return the figure for potential saving
    return ax


def plot_lambda_tuning(lambda_values: List[float], 
                     scores: List[float], 
                     metric_name: str = "Metric",
                     save_dir: str = None,
                     filename: str = None,
                     best_lambda: float = None,
                     figsize: tuple = (10, 6),
                     dpi: int = 300,
                     log_scale: bool = True) -> None:
    """
    Plot the results of lambda tuning from cross-validation.
    
    Parameters
    ----------
    lambda_values : List[float]
        List of lambda values that were evaluated
    scores : List[float]
        Corresponding performance metric scores
    metric_name : str, default="Metric"
        Name of the performance metric used (e.g., "F1 Score", "Accuracy")
    best_lambda : float, optional
        The optimal lambda value to highlight on the plot
    log_scale : bool, default=True
        Whether to use a logarithmic scale for the x-axis
    save_dir : str, optional
        Directory path where to save the figure
    filename : str, optional
        Filename for saving the figure
    """
    plt.figure(figsize=figsize)
    
    plt.plot(lambda_values, scores, 'o-', linewidth=2, markersize=8)
    
    if best_lambda is not None:
        # Find index of the closest lambda value in the array
        best_idx = np.argmin(np.abs(np.array(lambda_values) - best_lambda))
        best_score = scores[best_idx]
        plt.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7)
        plt.plot(best_lambda, best_score, 'ro', markersize=10, label=f'Best λ = {best_lambda}')
    
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel(f'{metric_name}', fontsize=12)
    plt.title(f'Effect of L2 Regularization (λ) on {metric_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if log_scale and min(lambda_values) > 0:
        plt.xscale('log')
    
    if best_lambda is not None:
        plt.legend()
    
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        figure_path = os.path.join(save_dir, filename)
        plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved lambda tuning plot to {figure_path}")
    
    plt.tight_layout()
    plt.show()




def plot_comparative_curves(all_models, output_dir, prefix="", show_plot=False, subplots=True):
    """
    Genera gráficos comparativos de curvas ROC y PR para múltiples modelos.
    
    Parameters
    ----------
    all_models : dict
        Diccionario con los modelos y sus métricas
    output_dir : str or Path
        Directorio donde guardar los gráficos
    prefix : str, optional
        Prefijo para los nombres de archivo
    show_plot : bool, optional
        Si es True, muestra los gráficos además de guardarlos
    subplots : bool, optional
        Si es True, genera ROC y PR en un mismo gráfico con subplots
    """
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    linestyles = ['-', '--', ':', '-.', '-']
    
    # Verificamos el formato de los datos para decidir si podemos graficar curvas
    can_plot_curves = False
    for name, model_data in all_models.items():
        roc_data = model_data["metrics"]["roc"]
        if isinstance(roc_data, dict) and "fpr" in roc_data and "tpr" in roc_data:
            can_plot_curves = True
            break
    
    if not can_plot_curves:
        print("No se pueden generar gráficos comparativos porque las métricas no contienen datos de curva.")
        print("Solo se tienen valores numéricos de AUC-ROC y AUC-PR.")
        
        # En su lugar, podemos generar una tabla o gráfico de barras con los valores AUC
        plt.figure(figsize=(10, 6))
        model_names = list(all_models.keys())
        roc_values = [model_data["metrics"]["roc"] for name, model_data in all_models.items()]
        pr_values = [model_data["metrics"]["pr"] for name, model_data in all_models.items()]
        
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
        plt.savefig(bar_path, dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return bar_path
    
    # Si podemos graficar curvas, procedemos normalmente
    if subplots:
        # Crear una figura con dos subplots (ROC y PR)
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot ROC curves
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data["metrics"]["roc"]
            if isinstance(roc_data, dict) and "auc" in roc_data:
                auc_roc = roc_data["auc"]
                fpr = roc_data["fpr"]
                tpr = roc_data["tpr"]
                
                plot_roc_curve(
                    fpr,
                    tpr,
                    auc_score=auc_roc,
                    label=f"{name} (AUC = {auc_roc:.4f})",
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    multiple_curves=True,
                    ax=ax_roc
                )
        
        ax_roc.set_title("ROC Curves for Different Rebalancing Techniques", fontsize=14)
        ax_roc.legend(loc="lower right", fontsize=10)
        
        # Plot PR curves
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data["metrics"]["pr"]
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                auc_pr = pr_data["average_precision"]
                precision = pr_data["precision"]
                recall = pr_data["recall"]
                
                plot_precision_recall_curve(
                    precision,
                    recall,
                    average_precision=auc_pr,
                    label=f"{name} (AP = {auc_pr:.4f})",
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    multiple_curves=True,
                    ax=ax_pr
                )
        
        ax_pr.set_title("Precision-Recall Curves for Different Rebalancing Techniques", fontsize=14)
        ax_pr.legend(loc="lower left", fontsize=10)
        
        plt.tight_layout()
        combined_file_path = os.path.join(output_dir, f"{prefix}combined_curves_comparison.png")
        plt.savefig(combined_file_path, dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return combined_file_path
    
    else:
        # Plot ROC curves
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        
        for i, (name, model_data) in enumerate(all_models.items()):
            roc_data = model_data["metrics"]["roc"]
            if isinstance(roc_data, dict) and "auc" in roc_data:
                auc_roc = roc_data["auc"]
                fpr = roc_data["fpr"]
                tpr = roc_data["tpr"]
                
                plot_roc_curve(
                    fpr,
                    tpr,
                    auc_score=auc_roc,
                    label=f"{name} (AUC = {auc_roc:.4f})",
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    multiple_curves=True,
                    ax=ax_roc
                )
        
        ax_roc.set_title("ROC Curves for Different Rebalancing Techniques", fontsize=14)
        ax_roc.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        roc_file_path = os.path.join(output_dir, f"{prefix}roc_curves_comparison.png")
        plt.savefig(roc_file_path, dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Plot PR curves
        fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
        
        for i, (name, model_data) in enumerate(all_models.items()):
            pr_data = model_data["metrics"]["pr"]
            if isinstance(pr_data, dict) and "average_precision" in pr_data:
                auc_pr = pr_data["average_precision"]
                precision = pr_data["precision"]
                recall = pr_data["recall"]
                
                plot_precision_recall_curve(
                    precision,
                    recall,
                    average_precision=auc_pr,
                    label=f"{name} (AP = {auc_pr:.4f})",
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    multiple_curves=True,
                    ax=ax_pr
                )
        
        ax_pr.set_title("Precision-Recall Curves for Different Rebalancing Techniques", fontsize=14)
        ax_pr.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        pr_file_path = os.path.join(output_dir, f"{prefix}pr_curves_comparison.png")
        plt.savefig(pr_file_path, dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return roc_file_path, pr_file_path
