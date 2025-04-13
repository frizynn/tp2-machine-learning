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
    solver: str = 'svd'  # Options: 'svd', 'eigen'


class LDA(BaseModel):
    """
    Linear Discriminant Analysis (LDA) classifier.
    
    LDA asume que cada clase se distribuye según una Gaussiana con media específica por clase,
    pero con una misma matriz de covarianza para todas las clases (Σ).
    A partir de esta suposición, llegamos a la forma lineal:
    
        δ_c(x) = γ_c + β_c^T x
    
    donde:
        β_c = Σ^{-1} μ_c,
        γ_c = log π_c - 1/2 (μ_c^T Σ^{-1} μ_c).

    Parameters
    ----------
    config : LDAConfig
        Configuración del modelo, que incluye el solver y otros hiperparámetros.
    """
    
    def __init__(self, config: Optional[LDAConfig] = None):
        super().__init__(config or LDAConfig())
        self.config: LDAConfig = config or LDAConfig()
        
        # Parámetros del modelo
        self.means_: Optional[np.ndarray] = None  # Medias por clase, shape (n_classes, n_features)
        self.priors_: Optional[np.ndarray] = None  # Prior de cada clase, shape (n_classes,)
        self.covariance_: Optional[np.ndarray] = None  # Matriz de covarianza compartida, shape (n_features, n_features)
        self.classes_: Optional[np.ndarray] = None  # Etiquetas únicas de clase
        self.n_classes_: Optional[int] = None      # Número de clases
        
        # β_c y γ_c en la notación del libro:
        self.beta_: Optional[np.ndarray] = None     # β_c para cada clase, shape (n_classes, n_features)
        self.gamma_: Optional[np.ndarray] = None    # γ_c para cada clase, shape (n_classes,)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
        """
        Ajusta el modelo LDA según los datos de entrenamiento (X, y).
        
        Parameters
        ----------
        X : np.ndarray
            Matriz de entrenamiento de shape (n_samples, n_features).
        y : np.ndarray
            Vector de etiquetas de shape (n_samples,).
            
        Returns
        -------
        self : LDA
            Devuelve la instancia de la clase entrenada.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X y y tienen formas incompatibles: {X.shape[0]} vs {y.shape[0]}")
        
        # Identificar clases y contarlas
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape
        
        if self.n_classes_ < 2:
            raise ValueError("El número de clases debe ser al menos 2.")
        
        # Inicializar medias y contadores
        self.means_ = np.zeros((self.n_classes_, n_features))
        n_samples_per_class = np.zeros(self.n_classes_, dtype=np.int64)
        
        # Calcular medias y número de muestras por clase
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            n_samples_per_class[idx] = X_class.shape[0]
            self.means_[idx] = np.mean(X_class, axis=0)
        
        # Calcular priors
        self.priors_ = n_samples_per_class / n_samples
        
        # Calcular la matriz de covarianza compartida
        self.covariance_ = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes_):
            X_class = X[y == cls]
            X_centered = X_class - self.means_[idx]
            self.covariance_ += X_centered.T @ X_centered
        
        # Normalizar por grados de libertad
        self.covariance_ /= (n_samples - self.n_classes_)
        
        # Pequeña regularización para evitar singulares
        self.covariance_ += np.eye(n_features) * 1e-6
        
        # Invertir o pseudo-invertir la matriz de covarianza según solver
        if self.config.solver == 'svd':
            U, s, Vt = np.linalg.svd(self.covariance_)
            pseudo_inv = Vt.T @ np.diag(1.0 / s) @ U.T
            # β_c = Σ^{-1} μ_c
            scaled_means = self.means_ @ pseudo_inv
        else:  # 'eigen'
            try:
                cov_inv = np.linalg.inv(self.covariance_)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(self.covariance_)
            scaled_means = self.means_ @ cov_inv
        
        # Asignamos β_c = scaled_means
        self.beta_ = scaled_means
        
        # γ_c = log π_c - 1/2 (μ_c^T Σ^{-1} μ_c)
        # Ojo que self.means_[idx] * scaled_means[idx] es elementwise,
        # sumando en axis=1 nos da μ_c^T Σ^{-1} μ_c.
        self.gamma_ = np.log(self.priors_) - 0.5 * np.sum(self.means_ * scaled_means, axis=1)
        
        self.is_fitted = True
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la función discriminante para cada clase.
        
        δ_c(x) = γ_c + β_c^T x
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Valores de la función discriminante para cada clase,
            con shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # δ_c(x) = X @ β_c^T + γ_c
        # Nota: self.beta_ tiene shape (n_classes, n_features), por eso se hace la transpuesta
        return X @ self.beta_.T + self.gamma_
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predice las probabilidades de cada clase usando softmax.
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Probabilidades de cada clase, con shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        scores = self.decision_function(X)
        # Softmax numéricamente estable
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase para cada muestra según la clase de mayor δ_c(x).
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Etiquetas de clase predichas, de shape (n_samples,).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        scores = self.decision_function(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]
