from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .logistic_regression import LogisticRegression, LogisticRegressionConfig


@dataclass
class CostSensitiveLogisticRegressionConfig(LogisticRegressionConfig):
    """Configuration for Cost-Sensitive Logistic Regression model
    
    Info
    -----
    - learning_rate: Learning rate for gradient descent
    - max_iter: Maximum number of iterations for gradient descent
    - tol: Tolerance for convergence
    - lambda_reg: L2 regularization strength
    - class_weights: Dict mapping class labels to weights (optional)
    """
    class_weight: Optional[Dict[int, float]] = None


class CostSensitiveLogisticRegression(LogisticRegression):
    """
    Cost-Sensitive Logistic Regression classifier with L2 regularization.
    
    This implementation applies different weights to samples from different classes
    during training, which is effective for handling imbalanced datasets.
    
    Parameters
    ----------
    config : CostSensitiveLogisticRegressionConfig
        Configuration parameters for the model
    """
    
    def __init__(self, config: CostSensitiveLogisticRegressionConfig = None):
        super().__init__(config or CostSensitiveLogisticRegressionConfig())
        # Ensure config is an instance of CostSensitiveLogisticRegressionConfig
        self.config: CostSensitiveLogisticRegressionConfig = config or CostSensitiveLogisticRegressionConfig()
        self.class_weight = self.config.class_weight
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost function (log loss) with L2 regularization and class weighting.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values. Shape is (n_samples,) for binary or 
            (n_samples, n_classes) for one-hot encoded multi-class.
            
        Returns
        -------
        float
            Computed cost value.
        """
        m = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification cost (Log Loss)
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            # Apply cost weighting
            if self.class_weight is not None:
                # Split the cost calculation for positive and negative classes
                # Only apply weights to each class separately
                positive_idx = (y == 1)
                negative_idx = (y == 0)
                
                pos_cost = -self.class_weight[1] * np.sum(y[positive_idx] * np.log(y_pred[positive_idx] + 1e-10))
                neg_cost = -self.class_weight[0] * np.sum((1 - y[negative_idx]) * np.log(1 - y_pred[negative_idx] + 1e-10))
                
                cost = (pos_cost + neg_cost) / m
            else:
                # Standard cost calculation if no class weights provided
                cost = -1/m * np.sum(y * np.log(y_pred + 1e-10) + 
                                    (1 - y) * np.log(1 - y_pred + 1e-10))
        else:
            # Multi-class classification cost (Cross-Entropy)
            z = np.dot(X, self.weights.T) + self.bias
            y_pred = self._softmax(z)
            
            if self.class_weight is not None:
                # For multi-class case with one-hot encoding
                cost = 0
                for class_idx in range(self.n_classes_):
                    # Get samples of this class
                    class_mask = np.argmax(y, axis=1) == class_idx
                    if np.any(class_mask):
                        weight = self.class_weight[class_idx]
                        # Apply weight to cross-entropy loss for this class
                        class_cost = -np.sum(y[class_mask] * np.log(y_pred[class_mask] + 1e-10))
                        cost += weight * class_cost
                cost /= m
            else:
                # Standard cost calculation if no class weights provided
                cost = -1/m * np.sum(y * np.log(y_pred + 1e-10))
        
        # Add L2 regularization term (applied to weights, not bias)
        l2_reg = (self.config.lambda_reg / (2 * m)) * np.sum(np.square(self.weights))
        
        return cost + l2_reg
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform gradient descent optimization with class weighting.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values. Shape is (n_samples,) for binary or 
            (n_samples, n_classes) for one-hot encoded multi-class.
        """
        m, n = X.shape
        prev_cost = float('inf')
        
        # Ensure X and weights are numpy arrays
        X = np.asarray(X, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        
        for iteration in range(self.config.max_iter):
            if self.n_classes_ == 2:
                # --- Binary classification gradient ---
                z = np.dot(X, self.weights) + self.bias
                y_pred = self._sigmoid(z)
                
                # Error calculation (difference between prediction and actual)
                error = y_pred - y # Shape (m,)
                
                if self.class_weight is not None:
                    # Apply class weights directly to samples based on their true class
                    # This will increase the gradient magnitude for minority class samples
                    weighted_error = np.zeros_like(error)
                    
                    # Apply weights separately for positive and negative classes
                    positive_idx = (y == 1)
                    negative_idx = (y == 0)
                    
                    weighted_error[positive_idx] = error[positive_idx] * self.class_weight[1]
                    weighted_error[negative_idx] = error[negative_idx] * self.class_weight[0]
                    
                    # Gradient calculation with weights
                    dw = (1/m) * np.dot(X.T, weighted_error) + (self.config.lambda_reg / m) * self.weights
                    db = (1/m) * np.sum(weighted_error)
                else:
                    # Standard gradient calculation
                    dw = (1/m) * np.dot(X.T, error) + (self.config.lambda_reg / m) * self.weights
                    db = (1/m) * np.sum(error)
                
            else:
                # --- Multi-class classification gradient ---
                z = np.dot(X, self.weights.T) + self.bias # Shape (m, n_classes)
                y_pred = self._softmax(z) # Shape (m, n_classes)
                
                # Error calculation (difference between prediction and actual one-hot)
                error = y_pred - y # Shape (m, n_classes)
                
                if self.class_weight is not None:
                    # Apply class weights based on true class
                    weighted_error = np.copy(error)
                    
                    # For each sample, apply weight based on its true class
                    true_classes = np.argmax(y, axis=1)
                    for i in range(m):
                        class_idx = true_classes[i]
                        weight = self.class_weight[class_idx]
                        weighted_error[i] = error[i] * weight
                    
                    # Gradient calculation with weights
                    dw = (1/m) * np.dot(weighted_error.T, X) + (self.config.lambda_reg / m) * self.weights
                    db = (1/m) * np.sum(weighted_error, axis=0)
                else:
                    # Standard gradient calculation
                    dw = (1/m) * np.dot(error.T, X) + (self.config.lambda_reg / m) * self.weights
                    db = (1/m) * np.sum(error, axis=0)
            
            # --- Update weights and bias ---
            self.weights = self.weights - self.config.learning_rate * dw
            self.bias = self.bias - self.config.learning_rate * db
            
            # --- Check for convergence ---
            current_cost = self._compute_cost(X, y)
            if self.config.verbose and (iteration % 100 == 0 or iteration == self.config.max_iter - 1):
                print(f"Iteration {iteration}, Cost: {current_cost:.6f}")
            
            # Convergence condition: change in cost is less than tolerance
            if abs(prev_cost - current_cost) < self.config.tol:
                if self.config.verbose:
                    print(f"Converged at iteration {iteration} with cost {current_cost:.6f}")
                break
            
            prev_cost = current_cost
        
        if iteration == self.config.max_iter - 1 and self.config.verbose:
             print(f"Reached max iterations ({self.config.max_iter}) without converging.")
    
    def set_class_weight(self, class_weight: Dict[int, float]) -> None:
        """
        Set the class weights after model initialization.
        
        Parameters
        ----------
        class_weight : Dict[int, float]
            Dictionary mapping class indices to weights.
        """
        self.class_weight = class_weight
        self.config.class_weight = class_weight 