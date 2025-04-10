from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseModel, ModelConfig


@dataclass
class LDAConfig(ModelConfig):
    """Configuration for Linear Discriminant Analysis model
    
    Parameters
    ----------
    solver : str
        Solver method for LDA ('svd' or 'eigen')
    """
    solver: str = 'svd'  # Options: 'svd', 'eigen'


class LDA(BaseModel):
    """
    Linear Discriminant Analysis classifier implementation.
    
    LDA is a generative model that assumes each class is drawn from a Gaussian distribution
    with class-specific means but shared covariance matrix.
    
    Parameters
    ----------
    config : LDAConfig
        Configuration parameters for the model
    """
    
    def __init__(self, config: LDAConfig = None):
        super().__init__(config or LDAConfig())
        self.config: LDAConfig = config or LDAConfig()
        
        # Model parameters
        self.means_: Optional[np.ndarray] = None  # Mean vector for each class
        self.priors_: Optional[np.ndarray] = None  # Class priors (probabilities)
        self.covariance_: Optional[np.ndarray] = None  # Shared covariance matrix
        self.classes_: Optional[np.ndarray] = None  # Unique class labels
        self.n_classes_: Optional[int] = None  # Number of classes
        self.coef_: Optional[np.ndarray] = None  # Linear discriminant coefficients
        self.intercept_: Optional[np.ndarray] = None  # Intercept terms
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """
        Fit LDA model according to the training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : LDA
            Returns self
        """
        # Convert inputs to numpy arrays if needed
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Basic validation
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        
        # Get unique classes and count them
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        
        if self.n_classes_ < 2:
            raise ValueError("The number of classes must be at least 2")
        
        # Initialize storage for class means and covariance
        self.means_ = np.zeros((self.n_classes_, n_features))
        
        # Compute class prior probabilities and class means
        n_samples_per_class = np.zeros(self.n_classes_, dtype=np.int64)
        
        # Calculate means for each class
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            n_samples_per_class[idx] = X_class.shape[0]
            self.means_[idx] = np.mean(X_class, axis=0)
        
        # Compute priors (class probabilities)
        self.priors_ = n_samples_per_class / n_samples
        
        # Compute shared covariance matrix (within-class scatter matrix)
        self.covariance_ = np.zeros((n_features, n_features))
        
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            X_centered = X_class - self.means_[idx]
            self.covariance_ += X_centered.T @ X_centered
        
        # Normalize by total degrees of freedom
        self.covariance_ /= (n_samples - self.n_classes_)
        
        # Add a small regularization to avoid singular covariance matrix
        self.covariance_ += np.eye(n_features) * 1e-6
        
        # Handle different solvers
        if self.config.solver == 'svd':
            # SVD decomposition for numerical stability
            U, s, Vt = np.linalg.svd(self.covariance_)
            eigvals = s
            pseudo_inv = Vt.T @ np.diag(1 / s) @ U.T
            scaled_means = self.means_ @ pseudo_inv
        else:  # 'eigen' solver
            # Compute inverse of covariance matrix directly
            try:
                cov_inv = np.linalg.inv(self.covariance_)
                scaled_means = self.means_ @ cov_inv
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if covariance is singular
                cov_inv = np.linalg.pinv(self.covariance_)
                scaled_means = self.means_ @ cov_inv
        
        # Compute linear discriminant coefficients
        self.coef_ = scaled_means
        
        # Compute intercept terms
        self.intercept_ = -0.5 * np.sum(self.means_ * scaled_means, axis=1) + np.log(self.priors_)
        
        self.is_fitted = True
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Apply decision function to the given data.
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Decision function values for each class, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # Compute decision scores: X @ coef_.T + intercept_
        return X @ self.coef_.T + self.intercept_
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given data.
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # Get decision scores
        scores = self.decision_function(X)
        
        # Convert scores to probabilities using softmax
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # Get decision scores and find class with highest score
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        
        # Map indices back to original class labels
        return self.classes_[indices] 