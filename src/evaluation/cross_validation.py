import numpy as np
from typing import Callable, List, Tuple, Union, Optional

from models.logistic_regression import LogisticRegression, LogisticRegressionConfig

def cross_validate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambda_values: List[float],
    metric_fn: Callable,
    k_folds: int = 5,
    verbose: bool = False,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    threshold: float = 0.5,
    aggregate_predictions: bool = True,
    average: str = "binary",
    resampler: Optional[object] = None  # Nuevo parámetro para re-muestreo
) -> Tuple[float, List[float]]:
    """
    Perform k-fold cross validation to find the optimal L2 regularization parameter (lambda).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    lambda_values : List[float]
        List of lambda values to evaluate
    metric_fn : Callable
        Function to compute the metric (e.g., f1_score, accuracy_score)
    k_folds : int, default=5
        Number of folds for cross validation
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=42
        Random state for reproducibility
    threshold : float, default=0.5
        Classification threshold for binary predictions
    aggregate_predictions : bool, default=True
        If True, aggregates all predictions across folds before computing the metric;
        otherwise, computes per-fold metrics and then averages them.
    resampler : Optional[object], default=None
        An object implementing fit_resample(X, y) for rebalancing.
        If None, no rebalancing is applied.
        
    Returns
    -------
    Tuple[float, List[float]]
        Optimal lambda value and list of mean metric scores for each lambda.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Generate folds using np.array_split for robust splitting
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    mean_scores = []
    
    for lambda_val in lambda_values:
        # Si aggregate_predictions es True, recolectamos todas las predicciones
        # y calculamos la métrica solo una vez al final de todos los folds
        if aggregate_predictions:
            all_y_true, all_y_pred = [], []
        else:
            # Si aggregate_predictions es False, calculamos métricas por cada fold
            fold_scores = []
        
        for fold in folds:
            # Current fold is validation; the rest is training
            val_indices = fold
            train_indices = np.concatenate([f for f in folds if not np.array_equal(f, fold)])
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Si se proporciona un resampler, aplica fit_resample a los datos de entrenamiento
            if resampler is not None:
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            
            config = LogisticRegressionConfig(
                learning_rate=learning_rate,
                max_iter=max_iter,
                tol=tol,
                lambda_reg=lambda_val
            )
            model = LogisticRegression(config)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val, threshold=threshold)
            
            if aggregate_predictions:
                # Si aggregate_predictions es True, acumulamos resultados sin calcular métricas por fold
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)
            else:
                # Si aggregate_predictions es False, calculamos la métrica para este fold
                fold_scores.append(metric_fn(y_val, y_pred, average=average))
        
        # Cálculo de la métrica final
        if aggregate_predictions:
            # Si aggregate_predictions es True, calculamos la métrica una sola vez con todos los datos
            score = metric_fn(np.array(all_y_true), np.array(all_y_pred), average=average)
        else:
            # Si aggregate_predictions es False, promediamos las métricas de cada fold
            score = np.mean(fold_scores)
            
        mean_scores.append(score)
        
        if verbose:
            mode = "Aggregated" if aggregate_predictions else "Mean"
            print(f"Lambda: {lambda_val:.4f}, {mode} {metric_fn.__name__}: {score:.4f}")
    
    best_idx = np.argmax(mean_scores)
    best_lambda = lambda_values[best_idx]
    
    return best_lambda, mean_scores


def stratified_cross_validate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambda_values: List[float],
    metric_fn: Callable,
    k_folds: int = 5,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    threshold: float = 0.5,
    aggregate_predictions: bool = True,
    average: str = "binary",
    resampler: Optional[object] = None  # Nuevo parámetro para re-muestreo
) -> Tuple[float, List[float]]:
    """
    Perform stratified k-fold cross validation to find the optimal L2 regularization parameter (lambda).
    This ensures a similar class distribution in each fold, which is crucial for imbalanced datasets.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    lambda_values : List[float]
        List of lambda values to evaluate
    metric_fn : Callable
        Function to compute the metric (e.g., f1_score, accuracy_score)
    k_folds : int, default=5
        Number of folds for cross validation
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=42
        Random state for reproducibility
    threshold : float, default=0.5
        Classification threshold for binary predictions
    aggregate_predictions : bool, default=True
        If True, aggregates all predictions across folds before computing the metric;
        otherwise, computes per-fold metrics and then averages them.
    average : str, default="binary"
        Type of averaging for multiclass metrics.
    resampler : Optional[object], default=None
        An object implementing fit_resample(X, y) for rebalancing.
        If None, no rebalancing is applied.
        
    Returns
    -------
    Tuple[float, List[float]]
        Optimal lambda value and list of mean metric scores for each lambda.
    """
    np.random.seed(random_state)
    
    # Get unique classes and their indices
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}
    
    folds = [[] for _ in range(k_folds)]
    for cls, indices_cls in class_indices.items():
        np.random.shuffle(indices_cls)
        cls_folds = np.array_split(indices_cls, k_folds)
        for i, fold in enumerate(cls_folds):
            folds[i].extend(fold.tolist())
    folds = [np.array(fold) for fold in folds]

    mean_scores = []
    
    for lambda_val in lambda_values:
        # Si aggregate_predictions es True, recolectamos todas las predicciones
        # y calculamos la métrica solo una vez al final de todos los folds
        if aggregate_predictions:
            all_y_true, all_y_pred = [], []
        else:
            # Si aggregate_predictions es False, calculamos métricas por cada fold
            fold_scores = []
        
        for fold in folds:
            val_indices = fold
            train_indices = np.concatenate([f for f in folds if not np.array_equal(f, fold)])
            
            X_train_fold, X_val_fold = X[train_indices], X[val_indices]
            y_train_fold, y_val_fold = y[train_indices], y[val_indices]
            
            # Apply rebalancing if a resampler is provided
            if resampler is not None:
                X_train_fold, y_train_fold = resampler.fit_resample(X_train_fold, y_train_fold)
            
            config = LogisticRegressionConfig(
                learning_rate=learning_rate,
                max_iter=max_iter,
                tol=tol,
                lambda_reg=lambda_val
            )
            model = LogisticRegression(config)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict(X_val_fold, threshold=threshold)
            
            if aggregate_predictions:
                # Si aggregate_predictions es True, acumulamos resultados sin calcular métricas por fold
                all_y_true.extend(y_val_fold)
                all_y_pred.extend(y_pred)
            else:
                # Si aggregate_predictions es False, calculamos la métrica para este fold
                fold_scores.append(metric_fn(y_val_fold, y_pred, average=average))
        
        # Cálculo de la métrica final
        if aggregate_predictions:
            # Si aggregate_predictions es True, calculamos la métrica una sola vez con todos los datos
            score = metric_fn(np.array(all_y_true), np.array(all_y_pred), average=average)
        else:
            # Si aggregate_predictions es False, promediamos las métricas de cada fold
            score = np.mean(fold_scores)
            
        mean_scores.append(score)
        
        if verbose:
            mode = "Aggregated" if aggregate_predictions else "Mean"
            print(f"Lambda: {lambda_val:.4f}, {mode} {metric_fn.__name__}: {score:.4f}")
    
    best_idx = np.argmax(mean_scores)
    best_lambda = lambda_values[best_idx]
    
    if verbose:
        print(f"\nBest lambda: {best_lambda}")
        if not aggregate_predictions:
            print(f"Fold scores for best lambda: {fold_scores}")
        else:
            print(f"Aggregated score for best lambda: {mean_scores[best_idx]:.4f}")
    
    return best_lambda, mean_scores
