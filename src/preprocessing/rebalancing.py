from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

@dataclass
class RebalancingConfig:
    """
    Configuration for class rebalancing
    
    Attributes
    ----------
    random_state : int
        Seed for random number generation
    sampling_strategy : Union[str, Dict[int, int], float]
        Strategy to determine the number of samples per class:
        - "auto": Balance all classes to the size of the majority class
        - dict: Specify exact number of samples for each class
        - float: Specify the ratio of minority class samples relative to majority class
    """
    random_state: int = 42
    sampling_strategy: Union[str, Dict[int, int], float] = "auto"

class BaseRebalancer(ABC):
    """
    Abstract base class for rebalancing techniques
    
    Parameters
    ----------
    config : RebalancingConfig
        Configuration for the rebalancer
    """
    
    def __init__(self, config: RebalancingConfig = None):
        self.config = config or RebalancingConfig()
        self.sampling_ratio = None
        self.class_indices = None
    
    @abstractmethod
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit rebalancer and resample data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Resampled feature matrix and target vector
        """
        pass
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
    
    def _compute_sampling_ratio(self, y: np.ndarray) -> Dict[int, int]:
        """
        Compute sampling ratio for each class
        
        Parameters
        ----------
        y : np.ndarray
            Target vector
            
        Returns
        -------
        Dict[int, int]
            Sampling ratio for each class
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        # create a mapping of class labels to their indices
        self.class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
        
        if self.config.sampling_strategy == "auto":
            # balance all classes to match the majority class size
            majority_count = class_counts.max()
            return {cls: int(majority_count) for cls in unique_classes}
        elif isinstance(self.config.sampling_strategy, dict):
            # use user-provided sample counts for each class
            return self.config.sampling_strategy
        elif isinstance(self.config.sampling_strategy, float):
            # adjust minority classes to be a fraction of the majority class
            majority_count = class_counts.max()
            return {cls: int(majority_count) if cls == unique_classes[np.argmax(class_counts)]
                    else int(majority_count * self.config.sampling_strategy)
                    for cls in unique_classes}
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.config.sampling_strategy}")

class RandomUnderSampler(BaseRebalancer):
    """
    Random undersampling
    
    Randomly removes samples from the majority class to balance class distribution.
    
    Parameters
    ----------
    config : RebalancingConfig
        Configuration for the rebalancer
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_inputs(X, y)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_count = class_counts.min()
        X_resampled = []
        y_resampled = []
        np.random.seed(self.config.random_state)
        for cls in unique_classes:
            # get indices for current class and randomly select subset
            cls_indices = np.where(y == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=int(min_count), replace=False)
            X_resampled.append(X[selected_indices])
            y_resampled.append(y[selected_indices])
        return np.vstack(X_resampled), np.hstack(y_resampled)

class RandomOverSampler(BaseRebalancer):
    """
    Random oversampling
    
    Randomly duplicates samples from the minority class to balance class distribution.
    
    Parameters
    ----------
    config : RebalancingConfig
        Configuration for the rebalancer
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_inputs(X, y)
        self.sampling_ratio = self._compute_sampling_ratio(y)
        unique_classes = np.unique(y)
        X_resampled = []
        y_resampled = []
        np.random.seed(self.config.random_state)
        for cls in unique_classes:
            cls_indices = self.class_indices[cls]
            target_count = self.sampling_ratio[cls]
            if target_count > len(cls_indices):
                # if we need more samples than we have, duplicate with replacement
                n_samples_to_add = target_count - len(cls_indices)
                additional_indices = np.random.choice(cls_indices, size=int(n_samples_to_add), replace=True)
                selected_indices = np.hstack([cls_indices, additional_indices])
            else:
                selected_indices = cls_indices
            X_resampled.append(X[selected_indices])
            y_resampled.append(y[selected_indices])
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        # shuffle the resampled data to avoid blocks of the same class
        final_indices = np.random.permutation(len(y_resampled))
        return X_resampled[final_indices], y_resampled[final_indices]

@dataclass
class SMOTEConfig(RebalancingConfig):
    """
    Configuration for SMOTE algorithm
    
    Attributes
    ----------
    k_neighbors : int
        Number of nearest neighbors to use for generating synthetic samples
    """
    k_neighbors: int = 5


class SMOTE(BaseRebalancer):
    """
    Synthetic Minority Over-sampling Technique (SMOTE)
    
    Creates synthetic samples by interpolating between minority class samples.
    
    Parameters
    ----------
    config : SMOTEConfig
        Configuration for SMOTE
    """
    
    def __init__(self, config: SMOTEConfig = None):
        super().__init__(config or SMOTEConfig())
        self.config = config or SMOTEConfig()
    
    def _generate_synthetic_samples(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples using SMOTE in a vectorized fashion.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of minority class samples (shape: (n_samples, n_features))
        n_samples : int
            Number of synthetic samples to generate
            
        Returns
        -------
        np.ndarray
            Synthetic samples array of shape (n_samples, n_features)
        """
        # ensure X is a numpy array with dtype float64
        X = np.asarray(X, dtype=np.float64)
        n_minority_samples, n_features = X.shape
        
        # handle edge cases
        if n_minority_samples == 0:
            return np.empty((0, n_features))
        if n_minority_samples == 1:
            return np.repeat(X, n_samples, axis=0)
        
        # compute pairwise differences using broadcasting for vectorized distance calculation
        diff = X[None, :, :] - X[:, None, :]  # shape: (n_minority_samples, n_minority_samples, n_features)
        distances = np.linalg.norm(diff, axis=2)  # shape: (n_minority_samples, n_minority_samples)
        
        np.random.seed(self.config.random_state)
        synthetic_samples = np.empty((n_samples, n_features))
        # limit k to available samples minus the sample itself
        k = min(self.config.k_neighbors, n_minority_samples - 1)
        
        # generate synthetic samples one by one
        for count in range(n_samples):
            # randomly select a minority sample
            idx = np.random.randint(0, n_minority_samples)
            # find k nearest neighbors (exclude the sample itself, hence [1:k+1])
            neighbor_indices = np.argsort(distances[idx])[1:k+1]
            # randomly select one of the neighbors
            nn_idx = np.random.choice(neighbor_indices, 1)[0]
            # generate random gap for interpolation
            gap = np.random.random()
            # create synthetic sample by interpolation
            synthetic_samples[count] = X[idx] + gap * (X[nn_idx] - X[idx])
        
        return synthetic_samples
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # ensure X is a float64 array for numerical stability
        X = np.asarray(X, dtype=np.float64)
        self._validate_inputs(X, y)
        self.sampling_ratio = self._compute_sampling_ratio(y)
        unique_classes, _ = np.unique(y, return_counts=True)
        X_resampled = []
        y_resampled = []
        for cls in unique_classes:
            cls_indices = self.class_indices[cls]
            cls_samples = X[cls_indices]
            target_count = self.sampling_ratio[cls]
            if target_count > len(cls_indices):
                # if oversampling needed, keep original samples and generate synthetic ones
                X_resampled.append(cls_samples)
                y_resampled.append(np.full(len(cls_indices), cls))
                n_synthetic = target_count - len(cls_indices)
                synthetic_samples = self._generate_synthetic_samples(cls_samples, n_synthetic)
                X_resampled.append(synthetic_samples)
                y_resampled.append(np.full(n_synthetic, cls))
            else:
                # if undersampling needed, use RandomUnderSampler
                under_sampler = RandomUnderSampler(RebalancingConfig(
                    random_state=self.config.random_state,
                    sampling_strategy={cls: target_count}
                ))
                under_sampler.class_indices = {cls: cls_indices}
                X_cls, y_cls = under_sampler.fit_resample(cls_samples, np.full(len(cls_indices), cls))
                X_resampled.append(X_cls)
                y_resampled.append(y_cls)
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        return X_resampled, y_resampled
