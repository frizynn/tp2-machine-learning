from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from .base import BaseModel, ModelConfig


@dataclass
class RandomForestConfig(ModelConfig):
    """Configuration for Random Forest classifier
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of each tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    max_features : Union[int, float, str], optional
        Number of features to consider when looking for the best split:
        - If int, consider max_features features
        - If float, consider max_features * n_features features
        - If "sqrt", consider sqrt(n_features) features
        - If "log2", consider log2(n_features) features
        - If None, consider all features
    min_samples_split : int
        Minimum number of samples required to split an internal node
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node
    bootstrap : bool
        Whether to use bootstrap samples when building trees
    criterion : str
        Function to measure the quality of a split ('entropy' or 'gini')
    """
    n_estimators: int = 100
    max_depth: Optional[int] = None
    max_features: Union[int, float, str, None] = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    criterion: str = "entropy"  # 'entropy' or 'gini'


class DecisionTreeNode:
    """A node in the decision tree"""
    
    def __init__(self, depth: int = 0):
        self.feature_idx: Optional[int] = None  # Feature index for split
        self.threshold: Optional[float] = None  # Threshold value for split
        self.left: Optional['DecisionTreeNode'] = None  # Left subtree (<=)
        self.right: Optional['DecisionTreeNode'] = None  # Right subtree (>)
        self.is_leaf: bool = False  # Whether this node is a leaf
        self.class_distribution: Optional[np.ndarray] = None  # Class distribution at this node
        self.prediction: Optional[Union[int, float]] = None  # Predicted class for leaf nodes
        self.depth: int = depth  # Depth of the node in the tree


class DecisionTree:
    """
    Decision Tree classifier implementation using NumPy.
    
    This implementation supports both 'entropy' and 'gini' criteria for splits.
    
    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree
    min_samples_split : int
        Minimum number of samples required to split an internal node
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node
    max_features : int, float, str or None
        Number of features to consider when looking for the best split
    criterion : str
        Function to measure the quality of a split ('entropy' or 'gini')
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(
        self, 
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[int, float, str, None] = None,
        criterion: str = "entropy",
        random_state: int = 42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        
        self.n_features_: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.tree_: Optional[DecisionTreeNode] = None
        self.is_fitted: bool = False
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a label array.
        
        Parameters
        ----------
        y : np.ndarray
            Array of labels
            
        Returns
        -------
        float
            Entropy value
        """
        m = len(y)
        if m <= 1:
            return 0
        
        # Get probabilities for each class
        _, counts = np.unique(y, return_counts=True)
        probs = counts / m
        
        # Calculate entropy: -sum(p_i * log2(p_i))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity of a label array.
        
        Parameters
        ----------
        y : np.ndarray
            Array of labels
            
        Returns
        -------
        float
            Gini impurity value
        """
        m = len(y)
        if m <= 1:
            return 0
        
        # Get probabilities for each class
        _, counts = np.unique(y, return_counts=True)
        probs = counts / m
        
        # Calculate Gini impurity: 1 - sum(p_i^2)
        gini = 1 - np.sum(np.square(probs))
        return gini
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Calculate information gain from a split.
        
        Parameters
        ----------
        y : np.ndarray
            Parent node labels
        y_left : np.ndarray
            Left child node labels
        y_right : np.ndarray
            Right child node labels
            
        Returns
        -------
        float
            Information gain value
        """
        m = len(y)
        m_left, m_right = len(y_left), len(y_right)
        
        if m_left == 0 or m_right == 0:
            return 0
        
        # Calculate parent impurity
        if self.criterion == "entropy":
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)
        else:  # "gini"
            parent_impurity = self._gini(y)
            left_impurity = self._gini(y_left)
            right_impurity = self._gini(y_right)
        
        # Calculate weighted average impurity of children
        weighted_child_impurity = (m_left / m) * left_impurity + (m_right / m) * right_impurity
        
        # Information gain is the difference between parent and weighted child impurity
        return parent_impurity - weighted_child_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold for splitting.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target array of shape (n_samples,)
        feature_indices : np.ndarray
            Indices of features to consider for splitting
            
        Returns
        -------
        Tuple[Optional[int], Optional[float], float]
            (best_feature_idx, best_threshold, best_info_gain)
        """
        m, n = X.shape
        best_info_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        # If too few samples, don't split
        if m < self.min_samples_split:
            return None, None, 0
        
        # Check each feature
        for feature_idx in feature_indices:
            # Get unique values in this feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # If only one unique value, skip this feature
            if len(thresholds) <= 1:
                continue
            
            # Compute thresholds between consecutive unique values for better splits
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            # Try each threshold
            for threshold in thresholds:
                # Split data based on threshold
                left_idx = feature_values <= threshold
                right_idx = ~left_idx
                
                # Ensure minimum samples in each leaf
                if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                info_gain = self._information_gain(y, y[left_idx], y[right_idx])
                
                # Update best split if this one is better
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_info_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Recursively build a decision tree.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target array of shape (n_samples,)
        depth : int
            Current depth of the tree
            
        Returns
        -------
        DecisionTreeNode
            Root node of the built tree
        """
        node = DecisionTreeNode(depth=depth)
        m, n = X.shape
        
        # Calculate class distribution at this node
        unique_classes, counts = np.unique(y, return_counts=True)
        class_distribution = np.zeros(self.n_classes_)
        for cls, count in zip(unique_classes, counts):
            class_idx = np.where(self.classes_ == cls)[0][0]
            class_distribution[class_idx] = count
        node.class_distribution = class_distribution
        
        # Determine number of features to consider at this node
        if isinstance(self.max_features, int):
            n_features_to_consider = min(self.max_features, n)
        elif isinstance(self.max_features, float):
            n_features_to_consider = max(1, int(self.max_features * n))
        elif self.max_features == "sqrt":
            n_features_to_consider = max(1, int(np.sqrt(n)))
        elif self.max_features == "log2":
            n_features_to_consider = max(1, int(np.log2(n)))
        else:  # None or other values
            n_features_to_consider = n
        
        # Randomly select feature indices to consider
        np.random.seed(self.random_state + depth)  # Change seed at each depth for more randomness
        feature_indices = np.random.choice(n, size=n_features_to_consider, replace=False)
        
        # Check stopping criteria
        if (
            depth >= self.max_depth if self.max_depth else False
            or m < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # Make this a leaf node
            node.is_leaf = True
            # Most common class as prediction
            node.prediction = np.argmax(class_distribution)
            node.prediction = self.classes_[node.prediction]  # Convert back to original class
            return node
        
        # Find best split
        best_feature_idx, best_threshold, best_info_gain = self._best_split(X, y, feature_indices)
        
        # If no good split found, make leaf node
        if best_feature_idx is None or best_info_gain <= 0:
            node.is_leaf = True
            node.prediction = np.argmax(class_distribution)
            node.prediction = self.classes_[node.prediction]
            return node
        
        # Create split based on best feature and threshold
        left_idx = X[:, best_feature_idx] <= best_threshold
        right_idx = ~left_idx
        
        # Create child nodes
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return node
    
    def _predict_tree(self, X: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        """
        Make predictions using a tree for a batch of samples.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        node : DecisionTreeNode
            Root node of the tree
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for i in range(m):
            current_node = node
            
            # Traverse the tree until a leaf node is reached
            while not current_node.is_leaf:
                if X[i, current_node.feature_idx] <= current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            
            predictions[i] = current_node.prediction
        
        return predictions
    
    def _predict_proba_tree(self, X: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        """
        Predict class probabilities using a tree for a batch of samples.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        node : DecisionTreeNode
            Root node of the tree
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        m = X.shape[0]
        probas = np.zeros((m, self.n_classes_))
        
        for i in range(m):
            current_node = node
            
            # Traverse the tree until a leaf node is reached
            while not current_node.is_leaf:
                if X[i, current_node.feature_idx] <= current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            
            # Use class distribution at leaf node as probabilities
            probas[i] = current_node.class_distribution / np.sum(current_node.class_distribution)
        
        return probas
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Build a decision tree from the training data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target array of shape (n_samples,)
            
        Returns
        -------
        self : DecisionTree
            Fitted estimator
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Check input shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        
        # Save number of features
        self.n_features_ = X.shape[1]
        
        # Get unique classes and count them
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Build the tree
        self.tree_ = self._build_tree(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("The model is not fitted yet. Call fit before predict.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Check feature shape
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but DecisionTree is expecting {self.n_features_} features.")
        
        return self._predict_tree(X, self.tree_)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("The model is not fitted yet. Call fit before predict_proba.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Check feature shape
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but DecisionTree is expecting {self.n_features_} features.")
        
        return self._predict_proba_tree(X, self.tree_)


class RandomForest(BaseModel):
    """
    Random Forest classifier implementation using NumPy.
    
    A random forest is an ensemble of decision trees, trained with the bagging method,
    where each tree is trained on a random subset of the training data and features.
    
    Parameters
    ----------
    config : RandomForestConfig
        Configuration parameters for the random forest
    """
    
    def __init__(self, config: RandomForestConfig = None):
        super().__init__(config or RandomForestConfig())
        self.config: RandomForestConfig = config or RandomForestConfig()
        
        # Model parameters
        self.trees_: List[DecisionTree] = []  # List of decision trees
        self.classes_: Optional[np.ndarray] = None  # Unique class labels
        self.n_classes_: Optional[int] = None  # Number of classes
        self.n_features_: Optional[int] = None  # Number of features
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Build a forest of trees from the training data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target array of shape (n_samples,)
            
        Returns
        -------
        self : RandomForest
            Fitted estimator
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Check input shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        
        # Save number of features and classes
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Get sample size for bootstrapping
        n_samples = X.shape[0]
        
        # Initialize the forest
        self.trees_ = []
        
        # Build each tree in the forest
        for i in range(self.config.n_estimators):
            # Create a new tree with current config
            tree = DecisionTree(
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                criterion=self.config.criterion,
                random_state=self.config.random_state + i  # Different seed for each tree
            )
            
            # Bootstrap sample if requested
            if self.config.bootstrap:
                # Sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap, y_bootstrap = X[indices], y[indices]
                tree.fit(X_bootstrap, y_bootstrap)
            else:
                # Use all data for each tree (still uses feature subsampling)
                tree.fit(X, y)
            
            # Add tree to forest
            self.trees_.append(tree)
            
            if self.config.verbose and (i % 10 == 0 or i == self.config.n_estimators - 1):
                print(f"Built tree {i+1}/{self.config.n_estimators}")
        
        self.is_fitted = True
        return self
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # Check feature shape
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but RandomForest is expecting {self.n_features_} features.")
        
        # Get predictions from all trees
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        # Aggregate predictions from all trees
        for tree in self.trees_:
            tree_probas = tree.predict_proba(X)
            probas += tree_probas
        
        # Average predictions
        probas /= len(self.trees_)
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        
        # Get probabilities
        probas = self.predict_prob(X)
        
        # Get class with highest probability
        y_pred_idx = np.argmax(probas, axis=1)
        
        # Map indices back to original classes
        return self.classes_[y_pred_idx]
    
    def feature_importances(self) -> np.ndarray:
        """
        Calculate feature importances based on mean decrease in impurity.
        
        Returns
        -------
        np.ndarray
            Feature importances of shape (n_features,)
        """
        self._check_is_fitted()
        
        # This is a simplified approach that would need more complete implementation
        # in a production-level Random Forest. In a full implementation, this would
        # track impurity decreases at each node weighted by number of samples.
        # Here we simply count feature usage frequency as a proxy.
        
        # Initialize feature importance array
        feature_importances = np.zeros(self.n_features_)
        
        # Count features used in split nodes
        for tree in self.trees_:
            self._count_feature_usage(tree.tree_, feature_importances)
        
        # Normalize to sum to 1
        if np.sum(feature_importances) > 0:
            feature_importances /= np.sum(feature_importances)
        
        return feature_importances
    
    def _count_feature_usage(self, node: DecisionTreeNode, importances: np.ndarray):
        """
        Helper method to count feature usage in a tree.
        
        Parameters
        ----------
        node : DecisionTreeNode
            Current node in the tree
        importances : np.ndarray
            Array to update with feature usage counts
        """
        if node is None or node.is_leaf:
            return
        
        # Increment count for this feature
        importances[node.feature_idx] += 1
        
        # Recurse on children
        self._count_feature_usage(node.left, importances)
        self._count_feature_usage(node.right, importances) 