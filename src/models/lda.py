from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseModel, ModelConfig


@dataclass
class LDAConfig(ModelConfig):
    """Configuration for Linear Discriminant Analysis (LDA).
    
    Parameters
    ----------
    solver : str
        Solver method for LDA ('svd' or 'eigen').
    """
    solver: str = 'svd'


class LDA(BaseModel):
    """
    Linear Discriminant Analysis (LDA) classifier.
    
    LDA assumes that each class follows a Gaussian distribution with class-specific means
    but shares the same covariance matrix (Σ) across all classes.
    From this assumption, we arrive at the linear form:
    
        δ_c(x) = γ_c + β_c^T x
    
    where:
        β_c = Σ^{-1} μ_c,
        γ_c = log π_c - 1/2 (μ_c^T Σ^{-1} μ_c).

    Parameters
    ----------
    config : LDAConfig
        Model configuration including solver and other hyperparameters.
    """
    
    def __init__(self, config: Optional[LDAConfig] = None):
        super().__init__(config or LDAConfig())
        self.config: LDAConfig = config or LDAConfig()
        
        self.means_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        
        self.beta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """
        Fits the LDA model to the training data (X, y).
        
        Parameters
        ----------
        X : np.ndarray
            Training matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).
            
        Returns
        -------
        self : LDA
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        
        if self.n_classes_ < 2:
            raise ValueError("The number of classes must be at least 2.")
        
        self.means_ = np.zeros((self.n_classes_, n_features))
        n_samples_per_class = np.zeros(self.n_classes_, dtype=np.int64)
        
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            n_samples_per_class[idx] = X_class.shape[0]
            self.means_[idx] = np.mean(X_class, axis=0)
        
        self.priors_ = n_samples_per_class / n_samples
        
        self.covariance_ = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            X_centered = X_class - self.means_[idx]
            self.covariance_ += X_centered.T @ X_centered
        
        self.covariance_ /= (n_samples - self.n_classes_)
        
        # add small regularization to avoid singular matrix
        self.covariance_ += np.eye(n_features) * 1e-6
        
        if self.config.solver == 'svd':
            # using svd for numerical stability
            U, s, Vt = np.linalg.svd(self.covariance_)
            pseudo_inv = Vt.T @ np.diag(1.0 / s) @ U.T
            scaled_means = self.means_ @ pseudo_inv
        else:  # 'eigen'
            try:
                cov_inv = np.linalg.inv(self.covariance_)
            except np.linalg.LinAlgError:
                # fallback to pseudo-inverse if matrix is singular
                cov_inv = np.linalg.pinv(self.covariance_)
            scaled_means = self.means_ @ cov_inv
        
        self.beta_ = scaled_means
        
        # compute γ_c using elementwise multiplication and sum
        self.gamma_ = np.log(self.priors_) - 0.5 * np.sum(self.means_ * scaled_means, axis=1)
        
        self.is_fitted = True
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the discriminant function for each class.
        
        δ_c(x) = γ_c + β_c^T x
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Discriminant function values for each class,
            with shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # δ_c(x) = X @ β_c^T + γ_c
        return X @ self.beta_.T + self.gamma_
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities using softmax.
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Probabilities for each class, with shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        scores = self.decision_function(X)
        # numerically stable softmax implementation
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class for each sample based on the highest δ_c(x).
        
        Parameters
        ----------
        X : np.ndarray
            Data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]
