from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np


@dataclass
class ModelConfig:
    """Base configuration class for ML models"""
    random_state: int = 42
    verbose: bool = False


class BaseModel(ABC):
    """Abstract base class for all machine learning models"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.is_fitted = False
        self._model_params: Dict[str, Any] = {}
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """Get the model parameters"""
        return self._model_params
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model on the given data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : BaseModel
            Trained model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, n_classes)
        """
        pass
    
    def _check_is_fitted(self) -> None:
        """Check if the model is fitted"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
    