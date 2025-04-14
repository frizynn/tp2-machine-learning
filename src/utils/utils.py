import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from IPython.display import display, HTML
from preprocessing.data_loader import DataLoader
from models.logistic_regression import LogisticRegression
from evaluation.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score,
    compute_binary_curves, compute_multiclass_curves
)
from utils.visuals import (
    plot_comparative_curves, plot_model_evaluation
)


from preprocessing.imputation import KNNImputer

def train_valid_split(df, test_size=0.2, random_state=42, stratify=None):
    """
    Split a DataFrame into training and validation sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split
    test_size : float, default=0.2
        Proportion of the dataset to include in the validation split
    random_state : int, default=42
        Seed for reproducibility
    stratify : array-like, optional
        Array of labels for stratified split. If provided,
        the class distribution in the training and validation sets
        will be similar to that of the original set.
        
    Returns
    -------
    tuple
        (df_train, df_valid) - Training and validation DataFrames
    """
    np.random.seed(random_state)
    
    if stratify is not None:
        unique_classes = np.unique(stratify)
        train_indices = []
        valid_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            np.random.shuffle(cls_indices)
            
            cls_test_size = int(len(cls_indices) * test_size)
            
            valid_indices.extend(cls_indices[:cls_test_size])
            train_indices.extend(cls_indices[cls_test_size:])
            
        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
        
    else:
        shuffled_indices = np.random.permutation(len(df))
        test_set_size = int(len(df) * test_size)
        valid_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
    
    return df.iloc[train_indices], df.iloc[valid_indices]



def print_numerical_features_range(df: pd.DataFrame, include_dtypes: List[str] = ['number'], decimal_places: int = 4) -> pd.DataFrame:
    """
    Display the range (minimum and maximum) of each numerical feature in an HTML table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the numerical features to analyze
    include_dtypes : List[str], default=['number']
        List of data types to include in the analysis. Default includes all numeric types
    decimal_places : int, default=4
        Number of decimal places to round the values to
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the feature names and their corresponding minimum and maximum values
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> ranges_df = print_numerical_features_range(df)
    >>> print(ranges_df)
       Feature  Minimum  Maximum
    0       A        1        3
    1       B        4        6
    """
    desc = df.select_dtypes(include=include_dtypes).describe().loc[["min", "max"]]
    ranges_df = pd.DataFrame({
        "Feature": desc.columns,
        "Minimum": np.around(desc.loc["min"].values, decimal_places),
        "Maximum": np.around(desc.loc["max"].values, decimal_places)
    })
    display(HTML(ranges_df.to_html(index=False)))
    return ranges_df

def get_model_metrics(model, X_test, y_test, threshold=0.5, pos_label=1, average="weighted", print_metrics=True):
    """
    Calculate evaluation metrics for a classification model.
    
    Parameters
    ----------
    model : object
        Trained model with a predict() method implemented
    X_test : array-like
        Features of the test set
    y_test : array-like
        True labels of the test set
    threshold : float, default=0.5
        Threshold for binary classification
    pos_label : int, default=1
        Label of the positive class for binary metrics
    average : str, default="weighted"
        Averaging method for multiclass metrics
    print_metrics : bool, default=True
        If True, prints the main metrics
    
    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    classes = np.unique(y_test)
    n_classes = len(classes)
    is_binary = (n_classes == 2)
    
    y_pred = model.predict(X_test, threshold=threshold) if isinstance(model, LogisticRegression) else model.predict(X_test)
    try:
        y_pred_prob = model.predict_prob(X_test)
        has_proba = True
    except (AttributeError, NotImplementedError):
        has_proba = False
    
    accuracy = accuracy_score(y_test, y_pred)
    if is_binary:
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    else:
        precision = precision_score(y_test, y_pred, average=average, labels=classes)
        recall = recall_score(y_test, y_pred, average=average, labels=classes)
        f1 = f1_score(y_test, y_pred, average=average, labels=classes)
    
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    if print_metrics:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'has_proba': has_proba
    }
    
    if has_proba:
        metrics['y_pred_prob'] = y_pred_prob
        
        if is_binary:
            # handle binary classification curves
            fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val = compute_binary_curves(y_test, y_pred_prob[:, 1], pos_label)
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
            # handle multi-class curves with one-vs-rest approach
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
    
    metrics['classes'] = classes
    metrics['is_binary'] = is_binary
    metrics['y_test'] = y_test
    metrics['pos_label'] = pos_label
    
    return metrics


def evaluate_model(model, X_test, y_test, class_names=None, threshold=0.5, pos_label=1, 
                   figsize=(16, 5), save_dir=None, base_filename=None, 
                   print_metrics=True, show_plots=True, subplots=True, average="weighted"):
    """
    Evaluate a classification model by calculating metrics and generating plots.
    This function is a wrapper that combines get_model_metrics and plot_model_evaluation.
    
    Parameters
    ----------
    model : object
        Model with predict() and optionally predict_prob() methods
    X_test : array-like
        Features of the test set
    y_test : array-like
        True labels
    class_names : list, optional
        Class names for plots
    threshold : float, default=0.5
        Decision threshold for binary classification
    pos_label : int, default=1
        Label of the positive class
    figsize : tuple, default=(16, 5)
        Size of the figures
    save_dir : str, optional
        Directory to save plots
    base_filename : str, optional
        Base name for saved files
    print_metrics : bool, default=True
        If True, prints the main metrics
    show_plots : bool, default=True
        If True, shows the plots
    subplots : bool, default=True
        If True, combines plots in subplots
    average : str, default="weighted"
        Averaging method for multiclass metrics

    Returns
    -------
    dict
        Dictionary with all calculated metrics
    """
    metrics = get_model_metrics(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        pos_label=pos_label,
        average=average,
        print_metrics=print_metrics
    )
    
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
    Evaluate multiple models on the same dataset,
    generating a comparative metrics table and (optionally) comparative plots.
    Uses get_model_metrics and plot_model_evaluation for better code organization.
    
    Parameters
    ----------
    all_models : dict
        Dictionary of models to evaluate
    X : array-like
        Features of the set to evaluate
    y : array-like
        True labels
    class_names : list
        Class names for plots
    output_dir : str
        Directory to save results
    prefix : str, default=""
        Prefix for generated files
    show_plot : bool, default=False
        If True, shows the plots
    individual_plots : bool, default=False
        If True, generates individual plots for each model
    subplots : bool, default=True
        If True, combines plots in subplots
    title_fontsize : int, default=20
        Font size for titles
    label_fontsize : int, default=16
        Font size for axis labels
    tick_fontsize : int, default=14
        Font size for ticks
    legend_fontsize : int, default=14
        Font size for legends
    """
    print(f"Evaluating models on the {prefix if prefix else 'validation'} set")
    evaluation_metrics = {}
    
    for model_name, model_data in all_models.items():
        print(f"Evaluating {model_name}")
        
        metrics = get_model_metrics(
            model=model_data["model"],
            X_test=X,
            y_test=y,
            threshold=0.5,
            print_metrics=False
        )
        
        if individual_plots:
            plot_model_evaluation(
                metrics=metrics,
                class_names=class_names,
                figsize=(12, 8),
                save_dir=output_dir,
                base_filename=f"{model_name.replace(' ', '_').lower()}_{prefix}",
                show_plots=show_plot,
                subplots=subplots
            )
        
        evaluation_metrics[model_name] = metrics
    
    metrics_df = pd.DataFrame({
        "Model": list(evaluation_metrics.keys()),
        "Accuracy": [evaluation_metrics[m].get("accuracy", np.nan) for m in evaluation_metrics],
        "Precision": [evaluation_metrics[m].get("precision", np.nan) for m in evaluation_metrics],
        "Recall": [evaluation_metrics[m].get("recall", np.nan) for m in evaluation_metrics],
        "F-Score": [evaluation_metrics[m].get("f1", np.nan) for m in evaluation_metrics],
        "AUC-ROC": [evaluation_metrics[m].get("roc", {}).get("auc", np.nan) if evaluation_metrics[m].get("roc") is not None else np.nan for m in evaluation_metrics],
        "AUC-PR": [evaluation_metrics[m].get("pr", {}).get("average_precision", np.nan) if evaluation_metrics[m].get("pr") is not None else np.nan for m in evaluation_metrics]
    })
    
    for col in metrics_df.columns:
        if col != "Model":
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (float, int, np.number)) else x)
    
    metrics_file_path = os.path.join(output_dir, f"{prefix}_metrics_comparison.csv")
    metrics_df.to_csv(metrics_file_path, index=False)
    
    # preserve original font settings before modifying
    _set_font_sizes = plt.rcParams.copy()
    plt.rcParams.update({
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize
    })
    
    if show_plot:
        models_for_plotting = {name: {"model": all_models[name]["model"], "metrics": evaluation_metrics[name]} for name in all_models}
        plot_comparative_curves(
            models_for_plotting, 
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            legend_fontsize=legend_fontsize,
            output_dir=output_dir, 
            prefix=f"{prefix}_", 
            show_plot=show_plot, 
            subplots=subplots,
            figsize=(20, 10)
        )
    
    plt.rcParams.update(_set_font_sizes)
    
    return metrics_df, evaluation_metrics

def analyze_null_values(dataframes, dataset_names=None):
    """
    Analyzes and displays detailed information about null values in one or more DataFrames.
    
    This function provides a comprehensive analysis of missing values in the input DataFrame(s),
    including per-column statistics and overall dataset statistics. It displays the results
    in a formatted table and returns detailed metrics for further analysis.
    
    Parameters
    ----------
    dataframes : pd.DataFrame or list of pd.DataFrame
        One or more DataFrames to analyze for null values.
    dataset_names : list of str, optional
        Names to use for each dataset in the output. If not provided, defaults to
        "Dataset 1", "Dataset 2", etc.
        
    Returns
    -------
    dict
        Dictionary containing detailed null value analysis for each dataset with keys:
        - 'null_table': DataFrame with per-column null statistics
        - 'summary': DataFrame with overall dataset null statistics
        - 'total_nulls': Total number of null values in the dataset
        - 'total_rows': Total number of rows in the dataset
        - 'null_percentage': Overall percentage of null values in the dataset
        
    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    >>> df2 = pd.DataFrame({'C': [None, 2, 3], 'D': [4, None, 6]})
    >>> results = analyze_null_values([df1, df2], ['Train', 'Test'])
    >>> print(results['Train']['null_percentage'])
    16.67
    """
    from IPython.display import display
    if not isinstance(dataframes, list):
        dataframes = [dataframes]
    if dataset_names is None or len(dataset_names) != len(dataframes):
        dataset_names = [f"Dataset {i+1}" for i in range(len(dataframes))]
    results = {}
    for df, name in zip(dataframes, dataset_names):
        print(f"Null values in {name}:")
        null_counts = df.isnull().sum()
        total_rows = len(df)
        null_percentage = (null_counts / total_rows) * 100
        null_table = pd.DataFrame({
            'Column': null_counts.index,
            'Number of nulls': null_counts.values,
            'Percentage (%)': null_percentage.values.round(2)
        })
        display(null_table)
        samples_with_nulls = df.isnull().any(axis=1).sum()
        samples_without_nulls = total_rows - samples_with_nulls
        samples_percentage = (samples_with_nulls / total_rows) * 100
        summary = pd.DataFrame({
            'Metric': ['Samples with at least one null value', 'Samples without null values', 'Total samples'],
            'Count': [samples_with_nulls, samples_without_nulls, total_rows],
            'Percentage (%)': [samples_percentage.round(2), (100 - samples_percentage).round(2), 100.0]
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
    Replaces all negative values in the numerical columns of the DataFrame with NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.
    numerical_cols : list
        List of numerical columns to check for negative values.
    
    Returns
    -------
    pd.DataFrame, int
        Processed DataFrame and the total number of negative values replaced.
    """
    df_copy = df.copy()
    total_negatives = 0
    for col in numerical_cols:
        mask = (df_copy[col] < 0) & df_copy[col].notna()
        negatives = mask.sum()
        total_negatives += negatives
        df_copy.loc[mask, col] = np.nan
    return df_copy, total_negatives

def impute_missing_values(train_df, valid_df=None, test_df=None, knn_neighbors=8, knn_weights="distance"):
    """
    Imputes missing values using a custom KNNImputer on one or more DataFrames.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame.
    valid_df : pd.DataFrame, optional
        Validation DataFrame.
    test_df : pd.DataFrame, optional
        Test DataFrame.
    knn_neighbors : int, default=8
        Number of neighbors to use.
    knn_weights : str, default="distance"
        'uniform' or 'distance'.
    
    Returns
    -------
    Tuple[pd.DataFrame,...]
        The imputed DataFrames; if only train_df is passed, returns that DataFrame.
    """
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
    Applies feature engineering transformations to a DataFrame.
    
    This function allows for the creation of new features by applying transformation functions
    to existing columns in the DataFrame. It supports both in-place modifications and
    creation of a new DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to transform
    transformations : dict
        Dictionary mapping new feature names to transformation functions.
        Each function should take the DataFrame as input and return a Series or array
        of the same length as the DataFrame.
    inplace : bool, default=False
        If True, modifies the original DataFrame. If False, returns a new DataFrame.
    verbose : bool, default=False
        If True, prints information about each transformation as it's applied.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new features added. If inplace=True, returns the modified original
        DataFrame. If inplace=False, returns a new DataFrame with the transformations applied.
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> transformations = {
    ...     'C': lambda x: x['A'] + x['B'],
    ...     'D': lambda x: x['A'] * x['B']
    ... }
    >>> new_df = apply_feature_engineering(df, transformations)
    >>> print(new_df)
       A  B  C   D
    0  1  4  5   4
    1  2  5  7  10
    2  3  6  9  18
    """
    if not inplace:
        df = df.copy()
    for new_feature, transform_func in transformations.items():
        if verbose:
            print(f"Applying transformation for feature: '{new_feature}'")
        df[new_feature] = transform_func(df)
    return df

def save_processed_data(loader, data_dict, data_dir, dataset_name, processing_type="preprocessed"):
    """
    Saves processed or preprocessed data to CSV files in a structured directory hierarchy.
    
    This function handles the saving of different dataset splits (train, validation, test, and optionally dev)
    to CSV files in a specified directory structure. It creates the necessary directories if they don't exist
    and uses the DataLoader instance to perform the actual saving operation.
    
    Parameters
    ----------
    loader : DataLoader
        Instance of DataLoader class that handles the actual data saving operations
    data_dict : dict
        Dictionary containing the DataFrames to save. Keys should be 'train', 'valid', 'test',
        and optionally 'dev'
    data_dir : pathlib.Path or str
        Base directory where the processed data will be saved
    dataset_name : str
        Name of the dataset, used to construct the output filenames
    processing_type : str, default="preprocessed"
        Type of processing applied to the data. Used to create a subdirectory.
        Common values are "preprocessed" or "processed"
        
    Returns
    -------
    None
    
    Examples
    --------
    >>> from pathlib import Path
    >>> from data_loader import DataLoader
    >>> loader = DataLoader()
    >>> data = {
    ...     'train': train_df,
    ...     'valid': valid_df,
    ...     'test': test_df
    ... }
    >>> save_processed_data(loader, data, Path('data'), 'my_dataset')
    """
    loader.update(**data_dict)
    
    base_path = data_dir / processing_type
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    file_paths = {
        'train': base_path / f"{dataset_name}_train.csv",
        'valid': base_path / f"{dataset_name}_valid.csv",
        'test': base_path / f"{dataset_name}_test.csv",
        'dev': base_path / f"{dataset_name}_dev.csv" if 'dev' in data_dict else None
    }
    
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

    probs = counts / total

    # complex part: finding majority class and setting weights for minority classes
    majority_idx = np.argmax(counts)
    majority_class = classes[majority_idx]
    majority_prob = probs[majority_idx]

    weights = {cls: 1.0 for cls in classes}

    for i, cls in enumerate(classes):
        if cls != majority_class:
            weights[cls] = majority_prob / probs[i]
            
    return weights

def normalize_data(X, params, return_params=False):
    """
    Normalizes data using provided parameters or calculates them if not provided.
    
    Parameters
    ----------
    X : array-like
        Data to normalize
    params : dict or None
        Dictionary with 'mean' and 'std' parameters. If None, they are calculated.
    return_params : bool, default=False
        Whether to return normalization parameters
        
    Returns
    -------
    array-like or tuple
        Normalized data or tuple (normalized_data, params)
    """
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
    Formats a metrics table for display, extracting AUC values from dictionaries when necessary.
    
    This function takes a DataFrame containing model evaluation metrics and formats it for display,
    handling nested dictionary values for AUC-ROC and AUC-PR metrics. It creates a copy of the
    input DataFrame to avoid modifying the original data.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing model evaluation metrics. Expected to have columns for various
        metrics including potentially 'AUC-ROC' and 'AUC-PR' which may contain nested dictionaries.
    title : str, default="Metrics Summary"
        Title to display before the metrics table.
        
    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with extracted values from nested dictionaries, ready for display.
        
    Examples
    --------
    >>> metrics_df = pd.DataFrame({
    ...     'Model': ['Model1'],
    ...     'AUC-ROC': [{'auc': 0.85}],
    ...     'AUC-PR': [{'average_precision': 0.78}]
    ... })
    >>> formatted_df = format_metrics_table(metrics_df)
    >>> print(formatted_df['AUC-ROC'].iloc[0])
    0.8500
    """
    display_df = metrics_df.copy()
    
    # complex part: extract values from nested dictionaries for display
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
    """
    Processes training data by separating features and target, and optionally encoding categorical variables.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame
    valid_df : pd.DataFrame
        Validation DataFrame
    target_column : str, default="Diagnosis"
        Name of the target column
    encode_categorical : bool, default=True
        Whether to encode categorical variables
        
    Returns
    -------
    tuple
        (X_train_encoded, y_train, X_val_encoded, y_val)
    """
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column].to_numpy()
    
    X_val = valid_df.drop(columns=[target_column])
    y_val = valid_df[target_column].to_numpy()
    
    if encode_categorical:
        X_train_encoded = DataLoader.encode_categorical(X_train).to_numpy()
        X_val_encoded = DataLoader.encode_categorical(X_val).to_numpy()
    else:
        X_train_encoded = X_train.to_numpy()
        X_val_encoded = X_val.to_numpy()
    
    return X_train_encoded, y_train, X_val_encoded, y_val
