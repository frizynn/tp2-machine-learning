from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np

from .base import BaseModel, ModelConfig

@dataclass
class LogisticRegressionConfig(ModelConfig):
    """Configuration for Logistic Regression model

    Info
    -----
    - learning_rate: Learning rate for gradient descent
    - max_iter: Maximum number of iterations for gradient descent
    - tol: Tolerance for convergence
    - lambda_reg: L2 regularization strength
    - class_weight: Optional dictionary mapping class labels to weights.
      If provided, applies cost sensitivity during training.
    """
    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-4
    lambda_reg: float = 0.1  # L2 regularization strength
    random_state: int = 42
    verbose: bool = False
    class_weight: Optional[Dict[int, float]] = None  # Nuevo parámetro

class LogisticRegression(BaseModel):
    """
    Logistic Regression classifier with L2 regularization.
    
    Can optionally perform cost-sensitive learning if class_weight is provided
    in the configuration.
    
    Parameters
    ----------
    config : LogisticRegressionConfig
        Configuration parameters for the model.
    """
    
    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        cfg = config or LogisticRegressionConfig()
        super().__init__(cfg)
        self.config: LogisticRegressionConfig = cfg
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[Union[float, np.ndarray]] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid function element-wise with clipping for numerical stability.
        
        Parameters
        ----------
        z : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            Sigmoid output with values between 0 and 1
        """
        # clip values to avoid overflow in exp calculation
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Apply softmax function along axis 1 with numerical stability.
        
        Parameters
        ----------
        z : np.ndarray
            Input values of shape (m, n_classes)
            
        Returns
        -------
        np.ndarray
            Softmax probabilities of shape (m, n_classes)
        """
        # shift for numerical stability to prevent overflow
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute log-loss cost with L2 regularization.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values. For binary classification: shape (n_samples,)
            For multiclass: one-hot encoded matrix of shape (n_samples, n_classes)
            
        Returns
        -------
        float
            Computed cost (log loss with L2 regularization)
        """
        m = X.shape[0]
        if self.n_classes_ == 2:
            z = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(z)
            if self.config.class_weight is not None:
                # separate cost calculation for positive and negative samples with class weighting
                pos_idx = (y == 1)
                neg_idx = (y == 0)
                pos_cost = -self.config.class_weight.get(1, 1.0) * np.sum(y[pos_idx] * np.log(y_pred[pos_idx] + 1e-10))
                neg_cost = -self.config.class_weight.get(0, 1.0) * np.sum((1 - y[neg_idx]) * np.log(1 - y_pred[neg_idx] + 1e-10))
                cost = (pos_cost + neg_cost) / m
            else:
                # standard binary cross-entropy loss
                cost = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
        else:
            z = X.dot(self.weights.T) + self.bias  # shape: (m, n_classes)
            y_pred = self._softmax(z)
            if self.config.class_weight is not None:
                # weighted cost for multiclass with class weights
                cost = 0
                for class_idx in range(self.n_classes_):
                    class_mask = (np.argmax(y, axis=1) == class_idx)
                    if np.any(class_mask):
                        weight = self.config.class_weight.get(class_idx, 1.0)
                        class_cost = -np.sum(y[class_mask] * np.log(y_pred[class_mask] + 1e-10))
                        cost += weight * class_cost
                cost /= m
            else:
                # standard categorical cross-entropy loss
                cost = -np.mean(np.sum(y * np.log(y_pred + 1e-10), axis=1))
        
        # add l2 regularization term
        l2_reg = (self.config.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        return cost + l2_reg
    
    def _initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights and bias depending on binary or multi-class classification.
        """
        np.random.seed(self.config.random_state)
        if self.n_classes_ == 2:
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0.0
        else:
            self.weights = np.random.randn(self.n_classes_, n_features) * 0.01
            self.bias = np.zeros(self.n_classes_)
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Optimize weights using gradient descent with L2 regularization.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values. For binary classification: shape (n_samples,)
            For multiclass: one-hot encoded matrix of shape (n_samples, n_classes)
        """
        m = X.shape[0]
        prev_cost = float('inf')
        
        for iteration in range(self.config.max_iter):
            if self.n_classes_ == 2:
                z = X.dot(self.weights) + self.bias
                y_pred = self._sigmoid(z)
                error = y_pred - y  # shape: (m,)
                
                if self.config.class_weight is not None:
                    # apply class weights to error term based on true class
                    weighted_error = np.where(y == 1, error * self.config.class_weight.get(1, 1.0),
                                                error * self.config.class_weight.get(0, 1.0))
                    dw = (1/m) * np.dot(X.T, weighted_error) + (self.config.lambda_reg / m) * self.weights
                    db = np.sum(weighted_error) / m
                else:
                    # standard gradients for binary logistic regression
                    dw = (1/m) * np.dot(X.T, error) + (self.config.lambda_reg / m) * self.weights
                    db = np.sum(error) / m
            else:
                z = X.dot(self.weights.T) + self.bias  # shape: (m, n_classes)
                y_pred = self._softmax(z)
                error = y_pred - y  # shape: (m, n_classes)
                
                if self.config.class_weight is not None:
                    # apply class weights for multiclass case
                    weighted_error = np.copy(error)
                    true_classes = np.argmax(y, axis=1)
                    for i in range(m):
                        weighted_error[i] = error[i] * self.config.class_weight.get(true_classes[i], 1.0)
                    dw = (1/m) * np.dot(weighted_error.T, X) + (self.config.lambda_reg / m) * self.weights
                    db = np.sum(weighted_error, axis=0) / m
                else:
                    # standard gradients for multiclass logistic regression
                    dw = (1/m) * np.dot(error.T, X) + (self.config.lambda_reg / m) * self.weights
                    db = np.sum(error, axis=0) / m
            
            # update parameters with learning rate
            self.weights -= self.config.learning_rate * dw
            self.bias -= self.config.learning_rate * db
            
            # check for convergence
            current_cost = self._compute_cost(X, y)
            if self.config.verbose and (iteration % 100 == 0 or iteration == self.config.max_iter - 1):
                print(f"Iteration {iteration}, Cost: {current_cost:.6f}")
            
            if abs(prev_cost - current_cost) < self.config.tol:
                if self.config.verbose:
                    print(f"Converged at iteration {iteration} with cost {current_cost:.6f}")
                break
            prev_cost = current_cost
        else:
            if self.config.verbose:
                print(f"Reached max iterations ({self.config.max_iter}) without converging.")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model using gradient descent.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Inconsistent samples: X {X.shape[0]}, y {y.shape[0]}")
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("At least two classes are required.")
        
        # Remap labels to 0,1,..., n_classes-1
        class_map = {label: i for i, label in enumerate(self.classes_)}
        y_mapped = np.array([class_map[label] for label in y])
        
        self._initialize_weights(X.shape[1])
        
        if self.n_classes_ == 2:
            y_train = y_mapped.astype(np.float64)
        else:
            y_train = np.zeros((y_mapped.size, self.n_classes_))
            y_train[np.arange(y_mapped.size), y_mapped] = 1
        
        self._gradient_descent(X, y_train)
        self.is_fitted = True
        self._model_params = {
            'weights': self.weights,
            'bias': self.bias,
            'classes': self.classes_
        }
        return self
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D, got {X.ndim}D.")
        if X.shape[1] != self.weights.shape[-1]:
            raise ValueError(f"Feature mismatch: input has {X.shape[1]}, expected {self.weights.shape[-1]}.")
        
        if self.n_classes_ == 2:
            z = X.dot(self.weights) + self.bias
            prob_class1 = self._sigmoid(z)
            proba = np.column_stack((1 - prob_class1, prob_class1))
        else:
            z = X.dot(self.weights.T) + self.bias
            proba = self._softmax(z)
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples in X.
        """
        proba = self.predict_prob(X)
        if self.n_classes_ == 2:
            indices = (proba[:, 1] >= threshold).astype(int)
        else:
            indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
