from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calcula la matriz de confusión a partir de los valores verdaderos y predichos de forma vectorizada.

    Parameters
    ----------
    y_true : np.ndarray
        Vector de etiquetas verdaderas.
    y_pred : np.ndarray
        Vector de etiquetas predichas.
    labels : Optional[np.ndarray]
        Array de etiquetas a considerar. Si es None se calcula como la unión única de y_true e y_pred.

    Returns
    -------
    np.ndarray
        Matriz de confusión de dimensión (n_labels, n_labels).
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    # Aseguramos que las etiquetas estén ordenadas
    labels = np.sort(labels)
    n_labels = labels.shape[0]
    # Mapear etiquetas a índices usando búsqueda en array ordenado
    y_true_idx = np.searchsorted(labels, y_true)
    y_pred_idx = np.searchsorted(labels, y_pred)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    np.add.at(cm, (y_true_idx, y_pred_idx), 1)
    return cm


def safe_division(numerator: float, denominator: float) -> float:
    """Realiza una división segura (retorna 0 si el denominador es 0)."""
    return numerator / denominator if denominator > 0 else 0.0


def compute_precision_recall(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula los arreglos de precisión y recall para cada clase usando la matriz de confusión de forma vectorizada.
    
    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusión de dimensión (n_classes, n_classes).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays de precisión y recall para cada clase.
    """
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    denom_precision = tp + fp
    denom_recall = tp + fn
    
    # Uso de np.divide con parámetros 'out' para manejar divisiones por cero de forma más robusta
    precisions = np.zeros_like(tp)
    recalls = np.zeros_like(tp)
    
    np.divide(tp, denom_precision, out=precisions, where=denom_precision > 0)
    np.divide(tp, denom_recall, out=recalls, where=denom_recall > 0)
    
    return precisions, recalls


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la precisión (accuracy) como la fracción de predicciones correctas.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return np.mean(y_true == y_pred)


def precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', labels: Optional[np.ndarray] = None,
    pos_label: int = 1
) -> Union[float, np.ndarray]:
    """
    Calcula la precisión (precision) usando TP / (TP + FP).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Vectores de etiquetas verdaderas y predichas.
    average : str, default 'binary'
        Tipo de promedio ('binary', 'micro', 'macro', 'weighted' o None).
    labels : Optional[np.ndarray]
        Conjunto de etiquetas a considerar.
    pos_label : int, default 1
        Etiqueta positiva (para clasificación binaria).

    Returns
    -------
    Union[float, np.ndarray]
        Precisión única si se promedia o vector de precisiones si average=None.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    precisions, _ = compute_precision_recall(cm)
    
    if average == 'binary':
        if labels.shape[0] != 2:
            raise ValueError("average='binary' requires binary classification")
        pos_idx = np.where(labels == pos_label)[0]
        if pos_idx.size == 0:
            raise ValueError(f"pos_label={pos_label} is not a valid label: {labels}")
        return precisions[pos_idx[0]]
    elif average == 'micro':
        tp_sum = np.sum(np.diag(cm))
        total = np.sum(cm)
        return safe_division(tp_sum, total)
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(precisions, weights=weights)
    elif average is None:
        return precisions
    else:
        raise ValueError(f"Unsupported average: {average}")


def recall_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', labels: Optional[np.ndarray] = None,
    pos_label: int = 1
) -> Union[float, np.ndarray]:
    """
    Calcula el recall usando TP / (TP + FN).

    Parameters are similar to precision_score.

    Returns
    -------
    Union[float, np.ndarray]
        Recall único o vector de recalls.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    _, recalls = compute_precision_recall(cm)
    
    if average == 'binary':
        if labels.shape[0] != 2:
            raise ValueError("average='binary' requires binary classification")
        pos_idx = np.where(labels == pos_label)[0]
        if pos_idx.size == 0:
            raise ValueError(f"pos_label={pos_label} is not a valid label: {labels}")
        return recalls[pos_idx[0]]
    elif average == 'micro':
        tp_sum = np.sum(np.diag(cm))
        total = np.sum(cm)
        return safe_division(tp_sum, total)
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(recalls, weights=weights)
    elif average is None:
        return recalls
    else:
        raise ValueError(f"Unsupported average: {average}")


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', labels: Optional[np.ndarray] = None, pos_label: int = 1
) -> Union[float, np.ndarray]:
    """
    Calcula el F1 score como la media armónica entre precisión y recall.

    Parameters are similar to precision_score and recall_score.

    Returns
    -------
    Union[float, np.ndarray]
        F1 score único o vector de F1 scores.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # Calcular precisión y recall de forma vectorizada
    precisions = precision_score(y_true, y_pred, average=None, labels=labels)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels)
    
    # Calcular F1-score de forma robusta evitando divisiones por cero
    denom = precisions + recalls
    
    # Inicializar con ceros
    f1_scores = np.zeros_like(denom)
    
    # Usar np.divide con parámetros out y where para evitar warnings de división por cero
    np.divide(2 * precisions * recalls, denom, out=f1_scores, where=denom > 0)
    
    if average == 'binary':
        if labels.shape[0] != 2:
            raise ValueError("average='binary' requires binary classification")
        # Para binario se asume que la clase positiva es la segunda (índice 1)
        return f1_scores[1]
    elif average == 'micro':
        cm = confusion_matrix(y_true, y_pred, labels)
        tp_sum = np.sum(np.diag(cm))
        total = np.sum(cm)
        micro = safe_division(tp_sum, total)
        return micro
    elif average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        weights = np.array([np.sum(y_true == label) for label in labels])
        return np.average(f1_scores, weights=weights)
    elif average is None:
        return f1_scores
    else:
        raise ValueError(f"Unsupported average: {average}")


def _prepare_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara los datos para curvas ROC y de precisión-recall:
      - Convierte y_true a binario según pos_label.
      - Ordena y_score y y_true de mayor a menor.
      - Determina los índices donde cambia el valor de los scores.
    """
    y_true_bin = (y_true == pos_label).astype(int)
    desc_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true_bin[desc_indices]
    # Índices donde cambia el score
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true_sorted.size - 1]
    return y_true_sorted, y_score_sorted, threshold_idxs, desc_indices


def roc_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva ROC.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        fpr, tpr y thresholds.
    """
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    y_true_sorted, y_score_sorted, threshold_idxs, _ = _prepare_curve(y_true, y_score, pos_label)
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    # Usar el último valor de tps y fps para normalizar (evitar división por cero)
    tpr = tps / (tps[-1] if tps[-1] > 0 else 1)
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1)
    
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[y_score_sorted[0] + 1, y_score_sorted[threshold_idxs]]
    return fpr, tpr, thresholds


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva de precisión-recall.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        precisions, recalls y thresholds.
    """
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    y_true_sorted, y_score_sorted, threshold_idxs, _ = _prepare_curve(y_true, y_score, pos_label)
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    precisions = np.where((tps + fps) > 0, tps / (tps + fps), 0.0)
    recalls = tps / (tps[-1] if tps[-1] > 0 else 1)
    
    precisions = np.r_[1, precisions]
    recalls = np.r_[0, recalls]
    thresholds = np.r_[y_score_sorted[0] + 1, y_score_sorted[threshold_idxs]]
    return precisions, recalls, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula el Área Bajo la Curva (AUC) usando la regla del trapecio.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    if x.shape[0] < 2:
        raise ValueError("At least 2 points are required to compute AUC")
    order = np.argsort(x)
    x_ordered, y_ordered = x[order], y[order]
    return float(np.trapz(y_ordered, x_ordered))


def roc_auc_score(
    y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro', multi_class: str = 'ovr'
) -> float:
    """
    Calcula el ROC AUC para clasificación binaria o multi-clase.

    Parameters
    ----------
    average : str, default 'macro'
        Tipo de promedio ('micro', 'macro', 'weighted').
    multi_class : str, default 'ovr'
        Estrategia para multi-clase ('ovr' o 'ovo').

    Returns
    -------
    float
        ROC AUC score.
    """
    unique_classes = np.unique(y_true)
    n_classes = unique_classes.shape[0]
    
    if n_classes == 2:
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=unique_classes[1])
        return auc(fpr, tpr)
    else:
        if y_score.ndim == 1:
            raise ValueError("For multiclass classification, y_score must be a 2D array")
        if y_score.shape[1] != n_classes:
            raise ValueError(f"y_score has {y_score.shape[1]} columns, but y_true has {n_classes} classes")
        
        if multi_class == 'ovr':
            auc_scores = np.empty(n_classes)
            class_counts = np.empty(n_classes)
            for i, cls in enumerate(unique_classes):
                y_true_bin = (y_true == cls).astype(int)
                y_score_cls = y_score[:, i]
                fpr, tpr, _ = roc_curve(y_true_bin, y_score_cls, pos_label=1)
                auc_scores[i] = auc(fpr, tpr)
                class_counts[i] = np.sum(y_true == cls)
            
            if average == 'micro':
                y_true_bin = np.array([(y_true == cls).astype(int) for cls in unique_classes]).T.ravel()
                fpr, tpr, _ = roc_curve(y_true_bin, y_score.ravel(), pos_label=1)
                return auc(fpr, tpr)
            elif average == 'macro':
                return np.mean(auc_scores)
            elif average == 'weighted':
                return np.average(auc_scores, weights=class_counts)
            else:
                raise ValueError(f"Unsupported average: {average}")
        elif multi_class == 'ovo':
            n_pairs = n_classes * (n_classes - 1) // 2
            auc_scores = np.empty(n_pairs)
            pair_counts = np.empty(n_pairs)
            pair_idx = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    indices = np.where((y_true == unique_classes[i]) | (y_true == unique_classes[j]))[0]
                    y_true_ij = y_true[indices]
                    # Calibrar scores para pares
                    y_score_ij = np.column_stack([y_score[indices, i], y_score[indices, j]])
                    y_score_ij = y_score_ij / np.sum(y_score_ij, axis=1, keepdims=True)
                    fpr, tpr, _ = roc_curve(y_true_ij, y_score_ij[:, 0], pos_label=unique_classes[i])
                    auc_scores[pair_idx] = auc(fpr, tpr)
                    pair_counts[pair_idx] = len(indices)
                    pair_idx += 1
            if average == 'macro':
                return np.mean(auc_scores)
            elif average == 'weighted':
                return np.average(auc_scores, weights=pair_counts)
            else:
                raise ValueError(f"Unsupported average: {average} for multi_class='ovo'")
        else:
            raise ValueError(f"Unsupported multi_class: {multi_class}")


def average_precision_score(
    y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro', pos_label: int = 1
) -> float:
    """
    Calcula el average precision (AP) a partir de los scores de predicción.
    """
    unique_classes = np.unique(y_true)
    n_classes = unique_classes.shape[0]
    
    if n_classes == 2:
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        return np.sum((recall[1:] - recall[:-1]) * precision[1:])
    else:
        if y_score.ndim == 1:
            raise ValueError("For multiclass classification, y_score must be a 2D array")
        if y_score.shape[1] != n_classes:
            raise ValueError(f"y_score has {y_score.shape[1]} columns, but y_true has {n_classes} classes")
        
        ap_scores = np.empty(n_classes)
        class_counts = np.empty(n_classes)
        for i, cls in enumerate(unique_classes):
            y_true_bin = (y_true == cls).astype(int)
            y_score_cls = y_score[:, i]
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score_cls, pos_label=1)
            ap_scores[i] = np.sum((recall[1:] - recall[:-1]) * precision[1:])
            class_counts[i] = np.sum(y_true == cls)
        
        if average == 'micro':
            y_true_bin = np.array([(y_true == cls).astype(int) for cls in unique_classes]).T.ravel()
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score.ravel(), pos_label=1)
            return np.sum((recall[1:] - recall[:-1]) * precision[1:])
        elif average == 'macro':
            return np.mean(ap_scores)
        elif average == 'weighted':
            return np.average(ap_scores, weights=class_counts)
        else:
            raise ValueError(f"Unsupported average: {average}")


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