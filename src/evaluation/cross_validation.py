import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Union, Optional
from concurrent.futures import ProcessPoolExecutor

from models.logistic_regression import LogisticRegression, LogisticRegressionConfig
from preprocessing.outliers import replace_outliers_iqr
from preprocessing.imputation import KNNImputer
from preprocessing.categorical_encoder import CategoricalEncoder
from utils.utils import calculate_class_weights, normalize_data


def preprocess_data_local(df: pd.DataFrame, categorical_columns: Optional[List[str]] = None, 
                          iqr_params: Optional[Tuple[float, float]] = None, return_params: bool = False,
                          normalize: bool = False, norm_params: Optional[dict] = None,
                          imputer: Optional[KNNImputer] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Tuple[float, float], dict, KNNImputer]]:
    """
    Preprocesses a dataframe by handling missing values, encoding categorical variables,
    handling outliers, and normalizing data if specified.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to preprocess
    categorical_columns : Optional[List[str]], default=None
        List of categorical column names to encode
    iqr_params : Optional[Tuple[float, float]], default=None
        Parameters for IQR-based outlier detection (lower_factor, upper_factor)
    return_params : bool, default=False
        Whether to return preprocessing parameters along with processed dataframe
    normalize : bool, default=False
        Whether to normalize the data
    norm_params : Optional[dict], default=None
        Parameters for normalization if already computed
    imputer : Optional[KNNImputer], default=None
        Pre-fitted imputer to use, prevents data leakage in cross-validation
        
    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Tuple[float, float], dict, KNNImputer]]
        Preprocessed dataframe and optionally the preprocessing parameters
    """
    df_proc = df.reset_index(drop=True).copy()
    
    if imputer is None:
        imputer = KNNImputer(n_neighbors=5, weights="distance", return_df=True)
        imputer.fit(df_proc)
    
    df_proc = imputer.transform(df_proc)

    if categorical_columns:
        valid_cat = [col for col in categorical_columns if col in df_proc.columns]
        if valid_cat:
            df_proc = CategoricalEncoder.encode_categorical(df_proc, categorical_columns=valid_cat)
    
    if iqr_params:
        df_proc = replace_outliers_iqr(df_proc, method="winsorize", params=iqr_params)
        params = iqr_params
    else:
        df_proc, params = replace_outliers_iqr(df_proc, method="winsorize", return_params=True)
    
    norm_params_result = None
    if normalize:
        if return_params:
            df_proc, norm_params_result = normalize_data(df_proc, norm_params, return_params=True)
        else:
            df_proc = normalize_data(df_proc, norm_params)
    
    if return_params:
        return (df_proc, params, norm_params_result, imputer) if normalize else (df_proc, params, imputer)
    return df_proc


def create_stratified_folds(X: pd.DataFrame, y: np.ndarray, k_folds: int, 
                           random_state: int = 42) -> List[np.ndarray]:
    """
    Creates stratified folds to maintain class distribution across folds.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : np.ndarray
        Target values array
    k_folds : int
        Number of folds to create
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    List[np.ndarray]
        List of index arrays for each fold
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    unique_classes = np.unique(y)
    class_indices = [np.where(y == cls)[0] for cls in unique_classes]
    
    folds = [[] for _ in range(k_folds)]
    
    # distribute indices from each class evenly across folds
    for cls_idx in class_indices:
        np.random.shuffle(cls_idx)
        fold_sizes = np.array_split(np.arange(len(cls_idx)), k_folds)
        for i, fold_size in enumerate(fold_sizes):
            folds[i].extend(cls_idx[fold_size])
    
    return [np.array(fold) for fold in folds]


def process_fold(fold_indices: np.ndarray, folds: List[np.ndarray],
                 X: pd.DataFrame, y: np.ndarray, lambda_val: float,
                 preprocess_per_fold: bool, global_preprocessed: Optional[pd.DataFrame],
                 categorical_columns: Optional[List[str]], resampler: Optional[object],
                 learning_rate: float, max_iter: int, tol: float, random_state: int, threshold: float,
                 metric_fn: Callable, average: str, apply_class_reweighting: bool,
                 normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a single fold during cross-validation, handling preprocessing, model training and prediction.
    
    Parameters
    ----------
    fold_indices : np.ndarray
        Indices for the current validation fold
    folds : List[np.ndarray]
        List of all fold indices
    X : pd.DataFrame
        Features dataframe
    y : np.ndarray
        Target values array
    lambda_val : float
        Regularization parameter value to test
    preprocess_per_fold : bool
        Whether to preprocess data within each fold
    global_preprocessed : Optional[pd.DataFrame]
        Pre-processed data if preprocessing is done globally
    categorical_columns : Optional[List[str]]
        List of categorical columns
    resampler : Optional[object]
        Instance of a resampling technique (e.g., SMOTE)
    learning_rate : float
        Learning rate for model training
    max_iter : int
        Maximum number of iterations for optimization
    tol : float
        Tolerance for convergence
    random_state : int
        Random seed
    threshold : float
        Prediction threshold
    metric_fn : Callable
        Metric function to evaluate model performance
    average : str
        Averaging method for metrics
    apply_class_reweighting : bool
        Whether to apply class weight adjustment
    normalize : bool
        Whether to normalize features
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        True labels and predictions for validation set
    """
    val_indices = fold_indices
    train_indices = np.concatenate([f for f in folds if not np.array_equal(f, fold_indices)])
    
    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    if preprocess_per_fold:
        if normalize:
            # calculate normalization parameters only on training data to avoid data leakage
            X_train, iqr_params, norm_params, imputer = preprocess_data_local(
                X_train, categorical_columns, return_params=True, normalize=True)
            
            # apply same parameters to validation data
            X_val = preprocess_data_local(
                X_val, categorical_columns, iqr_params=iqr_params, 
                normalize=True, norm_params=norm_params, imputer=imputer)
        else:
            X_train, iqr_params, imputer = preprocess_data_local(
                X_train, categorical_columns, return_params=True, normalize=False)
            
            X_val = preprocess_data_local(
                X_val, categorical_columns, iqr_params=iqr_params, 
                normalize=False, imputer=imputer)
    else:
        X_train = global_preprocessed.iloc[train_indices].reset_index(drop=True)
        X_val = global_preprocessed.iloc[val_indices].reset_index(drop=True)
    
    X_train_array = X_train.to_numpy()
    X_val_array = X_val.to_numpy()
    
    if resampler is not None:
        X_train_array, y_train = resampler.fit_resample(X_train_array, y_train)
    
    config = LogisticRegressionConfig(
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        lambda_reg=lambda_val,
        random_state=random_state,
        verbose=False
    )
    if apply_class_reweighting:
        cw = calculate_class_weights(y_train)
        config.class_weight = cw
    
    model = LogisticRegression(config)
    model.fit(X_train_array, y_train)
    y_pred = model.predict(X_val_array, threshold=threshold)
    return y_val, y_pred


def cross_validate_lambda(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    lambda_values: List[float],
    metric_fn: Callable,
    k_folds: int = 5,
    verbose: bool = False,
    normalize: bool = False,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    threshold: float = 0.5,
    aggregate_predictions: bool = True,
    average: str = "binary",
    resampler: Optional[object] = None,
    preprocess_per_fold: bool = True,
    apply_class_reweighting: bool = False,
    categorical_columns: Optional[List[str]] = None,
    stratified: bool = False  
) -> Tuple[float, List[float]]:
    """
    Performs cross-validation to find the optimal regularization parameter for logistic regression.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features dataframe
    y : Union[pd.Series, np.ndarray]
        Target values
    lambda_values : List[float]
        List of regularization parameter values to evaluate
    metric_fn : Callable
        Metric function to evaluate model performance
    k_folds : int, default=5
        Number of cross-validation folds
    verbose : bool, default=False
        Whether to print progress information
    normalize : bool, default=False
        Whether to normalize features
    learning_rate : float, default=0.01
        Learning rate for model training
    max_iter : int, default=1000
        Maximum number of iterations for optimization
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=42
        Random seed
    threshold : float, default=0.5
        Prediction threshold
    aggregate_predictions : bool, default=True
        Whether to aggregate predictions across folds before computing metrics
    average : str, default="binary"
        Averaging method for metrics
    resampler : Optional[object], default=None
        Instance of a resampling technique (e.g., SMOTE)
    preprocess_per_fold : bool, default=True
        Whether to preprocess data within each fold
    apply_class_reweighting : bool, default=False
        Whether to apply class weight adjustment
    categorical_columns : Optional[List[str]], default=None
        List of categorical columns
    stratified : bool, default=False
        Whether to use stratified folds
        
    Returns
    -------
    Tuple[float, List[float]]
        Best lambda value and list of metric scores for each lambda
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    
    np.random.seed(random_state)
    
    if stratified:
        folds = create_stratified_folds(X, y, k_folds, random_state)
    else:
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)
    
    global_preprocessed = None
    if not preprocess_per_fold:
        # this could still risk data leakage, but is kept for compatibility
        if normalize:
            global_preprocessed, _, _, _ = preprocess_data_local(X, categorical_columns, normalize=True, return_params=True)
        else:
            global_preprocessed, _, _ = preprocess_data_local(X, categorical_columns, return_params=True, normalize=False)
    
    mean_scores = []

    for lambda_val in lambda_values:
        if aggregate_predictions:
            all_y_true, all_y_pred = [], []
        else:
            fold_scores = []
        
        # parallelize fold processing for performance
        with ProcessPoolExecutor() as executor:
            futures = []
            for fold in folds:
                futures.append(executor.submit(process_fold, fold, folds, X, y, lambda_val,
                                                 preprocess_per_fold, global_preprocessed, categorical_columns,
                                                 resampler, learning_rate, max_iter, tol,
                                                 random_state, threshold, metric_fn, average, apply_class_reweighting,
                                                 normalize))
            for future in futures:
                y_true_fold, y_pred_fold = future.result()
                if aggregate_predictions:
                    all_y_true.extend(y_true_fold)
                    all_y_pred.extend(y_pred_fold)
                else:
                    fold_scores.append(metric_fn(np.array(y_true_fold), np.array(y_pred_fold), average=average))
        
        score = (metric_fn(np.array(all_y_true), np.array(all_y_pred), average=average)
                 if aggregate_predictions else np.mean(fold_scores))
        mean_scores.append(score)

        if verbose:
            mode = "Aggregated" if aggregate_predictions else "Mean"
            print(f"Lambda: {lambda_val:.4f}, {mode} {metric_fn.__name__}: {score:.4f}")
    
    best_idx = np.argmax(mean_scores)
    best_lambda = lambda_values[best_idx]
    
    return best_lambda, mean_scores
