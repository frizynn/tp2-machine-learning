from typing import  Optional, Tuple, Union
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the confusion matrix from true and predicted values using vectorized operations.

    Parameters
    ----------
    y_true : np.ndarray
        Vector of true labels.
    y_pred : np.ndarray
        Vector of predicted labels.
    labels : Optional[np.ndarray]
        Array of labels to consider. If None, will be calculated as the unique union of y_true and y_pred.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_labels, n_labels).
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    labels = np.sort(labels)
    n_labels = labels.shape[0]
    
    # map labels to indices using search in sorted array
    y_true_idx = np.searchsorted(labels, y_true)
    y_pred_idx = np.searchsorted(labels, y_pred)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    np.add.at(cm, (y_true_idx, y_pred_idx), 1)
    return cm


def safe_division(numerator: float, denominator: float) -> float:
    """
    Perform safe division between two numbers, handling division by zero cases.

    Parameters
    ----------
    numerator : float
        The numerator in the division operation
    denominator : float
        The denominator in the division operation

    Returns
    -------
    float
        The result of the division if denominator is positive, 0.0 otherwise

    Examples
    --------
    >>> safe_division(10, 2)
    5.0
    >>> safe_division(10, 0)
    0.0
    """
    return numerator / denominator if denominator > 0 else 0.0

def compute_precision_recall(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision and recall arrays for each class using the confusion matrix in a vectorized way.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of precision and recall for each class.
    """
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    denom_precision = tp + fp
    denom_recall = tp + fn
    
    # using np.divide with 'out' parameters to handle zero divisions robustly
    precisions = np.zeros_like(tp)
    recalls = np.zeros_like(tp)
    
    np.divide(tp, denom_precision, out=precisions, where=denom_precision > 0)
    np.divide(tp, denom_recall, out=recalls, where=denom_recall > 0)
    
    return precisions, recalls


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score for classification predictions.
    
    Accuracy is defined as the fraction of correctly predicted samples over the total number of samples.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
        
    Returns
    -------
    float
        Accuracy score between 0 and 1, where 1 indicates perfect prediction.
        
    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
        
    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy_score(y_true, y_pred)
    0.75
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return np.mean(y_true == y_pred)

def precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary', labels: Optional[np.ndarray] = None,
    pos_label: int = 1
) -> Union[float, np.ndarray]:
    """
    Calculate precision using TP / (TP + FP).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Vectors of true and predicted labels.
    average : str, default 'binary'
        Type of averaging ('binary', 'micro', 'macro', 'weighted' or None).
    labels : Optional[np.ndarray]
        Set of labels to consider.
    pos_label : int, default 1
        Positive label (for binary classification).

    Returns
    -------
    Union[float, np.ndarray]
        Single precision if averaged or vector of precisions if average=None.
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
    Calculate recall (sensitivity) score for classification problems.

    Recall is defined as the ratio of true positives to the sum of true positives and false negatives.
    It measures the ability of the classifier to find all positive samples.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    average : str, default='binary'
        Averaging method to apply:
        - 'binary': Only report results for the positive class (pos_label).
        - 'micro': Calculate metrics globally by counting total true positives,
                   false negatives and false positives.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - 'weighted': Calculate metrics for each label, and find their average weighted
                     by support (the number of true instances for each label).
        - None: Return the score for each class.
    labels : Optional[np.ndarray], default=None
        The set of labels to include when average != 'binary'. If None, all labels
        present in y_true and y_pred are used.
    pos_label : int, default=1
        The label of the positive class. Only used when average='binary'.

    Returns
    -------
    Union[float, np.ndarray]
        Recall score of the positive class in binary classification or weighted average
        of the recall scores of each class for the multiclass task.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
        If average='binary' is used with non-binary classification.
        If pos_label is not present in the labels.
        If an unsupported average method is provided.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> recall_score(y_true, y_pred)
    0.5
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
    Calculate the F1 score as the harmonic mean of precision and recall.

    The F1 score is a measure of a test's accuracy that considers both precision and recall.
    It is particularly useful when dealing with imbalanced datasets.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier.
    average : str, default='binary'
        Averaging method to apply:
        - 'binary': Only report results for the positive class (pos_label).
        - 'micro': Calculate metrics globally by counting total true positives,
                   false negatives and false positives.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - 'weighted': Calculate metrics for each label, and find their average weighted
                     by support (the number of true instances for each label).
        - None: Return the score for each class.
    labels : Optional[np.ndarray], default=None
        The set of labels to include when average != 'binary'. If None, all labels
        present in y_true and y_pred are used.
    pos_label : int, default=1
        The label of the positive class. Only used when average='binary'.

    Returns
    -------
    Union[float, np.ndarray]
        F1 score of the positive class in binary classification or weighted average
        of the F1 scores of each class for the multiclass task.

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.
        If average='binary' is used with non-binary classification.
        If an unsupported average method is provided.

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> f1_score(y_true, y_pred)
    0.6666666666666666
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # calculate precision and recall in a vectorized way
    precisions = precision_score(y_true, y_pred, average=None, labels=labels)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels)
    
    # calculate F1-score robustly avoiding zero divisions
    denom = precisions + recalls
    
    f1_scores = np.zeros_like(denom)
    
    np.divide(2 * precisions * recalls, denom, out=f1_scores, where=denom > 0)
    
    if average == 'binary':
        if labels.shape[0] != 2:
            raise ValueError("average='binary' requires binary classification")
        # for binary classification, assume the positive class is the second one (index 1)
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
    Prepare data for ROC and precision-recall curves by preprocessing the input arrays.

    This helper function performs several preprocessing steps:
    1. Converts true labels to binary format based on the positive label
    2. Sorts scores and corresponding labels in descending order
    3. Identifies indices where score values change to determine thresholds

    Parameters
    ----------
    y_true : np.ndarray
        Array of true labels. Will be converted to binary format where 1 indicates the positive class.
    y_score : np.ndarray
        Array of prediction scores or probabilities. Higher values indicate higher confidence in positive class.
    pos_label : int
        The label value that represents the positive class in y_true.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - y_true_sorted : np.ndarray
            Binary true labels sorted in descending order of scores
        - y_score_sorted : np.ndarray
            Scores sorted in descending order
        - threshold_idxs : np.ndarray
            Indices where score values change, used to determine thresholds
        - desc_indices : np.ndarray
            Original indices that produced the sorted order

    Notes
    -----
    This function is used internally by roc_curve and precision_recall_curve functions
    to prepare data for curve computation. The sorting is done using a stable mergesort
    to ensure consistent results.
    """
    y_true_bin = (y_true == pos_label).astype(int)
    desc_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true_bin[desc_indices]
    
    # indices where score changes
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true_sorted.size - 1]
    return y_true_sorted, y_score_sorted, threshold_idxs, desc_indices

def roc_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Receiver Operating Characteristic (ROC) curve for binary classification.

    The ROC curve shows the tradeoff between the true positive rate (sensitivity) and the false positive rate
    (1-specificity) for different threshold values. A high area under the curve represents both high sensitivity
    and high specificity.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels. If labels are not binary, pos_label should be given.
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.
    pos_label : int, default=1
        The label of the positive class.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - fpr : np.ndarray
            False positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        - tpr : np.ndarray
            True positive rates such that element i is the true positive rate of predictions with score >= thresholds[i]
        - thresholds : np.ndarray
            Decreasing thresholds on the decision function used to compute fpr and tpr.

    Raises
    ------
    ValueError
        If y_true and y_score have different lengths.
    """
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    y_true_sorted, y_score_sorted, threshold_idxs, _ = _prepare_curve(y_true, y_score, pos_label)
    
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    # use the last value of tps and fps to normalize (avoid division by zero)
    tpr = tps / (tps[-1] if tps[-1] > 0 else 1)
    fpr = fps / (fps[-1] if fps[-1] > 0 else 1)
    
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[y_score_sorted[0] + 1, y_score_sorted[threshold_idxs]]
    return fpr, tpr, thresholds

def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the precision-recall curve for binary classification.

    The precision-recall curve shows the tradeoff between precision and recall for different
    threshold values. A high area under the curve represents both high recall and high precision,
    where high precision relates to a low false positive rate, and high recall relates to a
    low false negative rate.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels. If labels are not binary, pos_label should be given.
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.
    pos_label : int, default=1
        The label of the positive class.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - precisions : np.ndarray
            Precision values such that element i is the precision of predictions with score >= thresholds[i]
        - recalls : np.ndarray
            Recall values such that element i is the recall of predictions with score >= thresholds[i]
        - thresholds : np.ndarray
            Decreasing thresholds on the decision function used to compute precision and recall.

    Raises
    ------
    ValueError
        If y_true and y_score have different lengths.
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
    Calculate the Area Under the Curve (AUC) using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates of the points defining the curve.
    y : np.ndarray
        Array of y-coordinates of the points defining the curve.

    Returns
    -------
    float
        The area under the curve computed using the trapezoidal rule.

    Raises
    ------
    ValueError
        If x and y arrays have different lengths.
        If there are less than 2 points to compute the AUC.

    Notes
    -----
    The function first sorts the points by x-coordinate to ensure proper integration.
    The trapezoidal rule is used to approximate the integral of the curve.
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
    Calculate the ROC AUC for binary or multi-class classification.

    Parameters
    ----------
    average : str, default 'macro'
        Type of averaging ('micro', 'macro', 'weighted').
    multi_class : str, default 'ovr'
        Strategy for multi-class ('ovr' or 'ovo').

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
                # create a flattened version of the one-hot representation
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
            # one-vs-one approach
            n_pairs = n_classes * (n_classes - 1) // 2
            auc_scores = np.empty(n_pairs)
            pair_counts = np.empty(n_pairs)
            pair_idx = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    indices = np.where((y_true == unique_classes[i]) | (y_true == unique_classes[j]))[0]
                    y_true_ij = y_true[indices]
                    # calibrate scores for pairs
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
    Calculate the average precision (AP) from prediction scores.
    
    The average precision summarizes a precision-recall curve as the weighted mean of
    precisions achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels in range {0, 1} or {-1, 1} for binary classification,
        or array of class labels for multiclass classification.
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class
        or confidence values. For multiclass classification, must be a 2D array
        of shape (n_samples, n_classes).
    average : str, default='macro'
        Averaging method for multiclass classification:
        - 'micro': Calculate metrics globally by considering each element of the label
          indicator matrix as a label.
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - 'weighted': Calculate metrics for each label, and find their average weighted
          by support (the number of true instances for each label).
    pos_label : int, default=1
        The label of the positive class. Only used for binary classification.
        
    Returns
    -------
    float
        Average precision score.
        
    Raises
    ------
    ValueError
        If y_score is not a 2D array for multiclass classification.
        If the number of columns in y_score does not match the number of classes.
        If an unsupported averaging method is provided.
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
            # create a flattened version of the one-hot representation
            y_true_bin = np.array([(y_true == cls).astype(int) for cls in unique_classes]).T.ravel()
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score.ravel(), pos_label=1)
            return np.sum((recall[1:] - recall[:-1]) * precision[1:])
        elif average == 'macro':
            return np.mean(ap_scores)
        elif average == 'weighted':
            return np.average(ap_scores, weights=class_counts)
        else:
            raise ValueError(f"Unsupported average: {average}")

def compute_binary_curves(y_true: np.ndarray, y_score: np.ndarray, pos_label: int) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """
    Calculate ROC and Precision-Recall curves for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels in range {0, 1} or {-1, 1}
    y_score : np.ndarray
        Target scores, can either be probability estimates of the positive class
        or confidence values
    pos_label : int
        The label of the positive class

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]
        A tuple containing:
        - fpr : np.ndarray
            False positive rates
        - tpr : np.ndarray
            True positive rates
        - roc_auc_val : float
            Area under the ROC curve
        - precision_vals : np.ndarray
            Precision values
        - recall_vals : np.ndarray
            Recall values
        - pr_auc_val : float
            Area under the Precision-Recall curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc_val = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    pr_auc_val = auc(recall_vals, precision_vals)
    return fpr, tpr, roc_auc_val, precision_vals, recall_vals, pr_auc_val


def compute_multiclass_curves(y_true, y_pred_prob, classes):
    """
    Calculate ROC and Precision-Recall curves for each class in one-vs-rest mode.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred_prob : np.ndarray
        Predicted probabilities for each class, shape (n_samples, n_classes)
    classes : array-like
        List of unique class labels

    Returns
    -------
    tuple
        A tuple containing:
        - fpr : dict
            Dictionary of false positive rates for each class
        - tpr : dict
            Dictionary of true positive rates for each class
        - roc_auc_scores : np.ndarray
            ROC AUC scores for each class
        - precision_dict : dict
            Dictionary of precision values for each class
        - recall_dict : dict
            Dictionary of recall values for each class
        - pr_auc_scores : np.ndarray
            Precision-Recall AUC scores for each class
    """
    n_classes = len(classes)
    fpr, tpr = {}, {}
    roc_auc_scores = np.empty(n_classes)
    precision_dict, recall_dict = {}, {}
    pr_auc_scores = np.empty(n_classes)
    
    # for each class, binarize and calculate the corresponding curve
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
        roc_auc_scores[i] = auc(fpr[i], tpr[i])
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true_bin, y_pred_prob[:, i], pos_label=1)
        pr_auc_scores[i] = auc(recall_dict[i], precision_dict[i])
    return fpr, tpr, roc_auc_scores, precision_dict, recall_dict, pr_auc_scores