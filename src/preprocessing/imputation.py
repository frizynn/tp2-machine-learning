import numpy as np
import pandas as pd

class KNNImputer:
    """
    Imputer for completing missing values using k-Nearest Neighbors.
    
    This class implements KNN imputation, which replaces missing values by finding 
    the k nearest neighbors in the feature space and using their values (weighted or not)
    to estimate the missing value.
    
    The algorithm works by:
    1. Normalizing numerical features to ensure they're on the same scale
    2. Finding k nearest neighbors for each sample with missing values
    3. Imputing missing values using either uniform weighting or distance-based weighting
    4. For categorical variables, using mode imputation
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for imputation.
    weights : str, default='uniform'
        Weight function: 'uniform' for equal weights or 'distance' to weight by inverse distance.
    metric : str, default='euclidean'
        Distance metric to use (currently only 'euclidean' is supported).
    return_df : bool, default=True
        Whether to return a pandas DataFrame or a numpy array.
        
    Attributes
    ----------
    fitted : bool
        Whether the imputer has been fitted to data.
    X_reference : pd.DataFrame
        Reference dataset after fitting.
    X_norm_reference : pd.DataFrame
        Normalized reference dataset for numeric features.
    numeric_cols : pd.Index
        Names of numeric columns in the training data.
    categorical_cols : pd.Index
        Names of categorical columns in the training data.
    means : pd.Series
        Mean values for each numeric feature used for normalization.
    stds : pd.Series
        Standard deviation values for each numeric feature used for normalization.
    cat_modes : dict
        Mode values for each categorical feature used for imputation.
    """
    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", return_df=True):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.return_df = return_df
        self.fitted = False
        self.X_reference = None
        self.X_norm_reference = None
        self.numeric_cols = None
        self.categorical_cols = None

    def _normalize_data(self, X):
        """
        Normalize the data to zero mean and unit variance.
        
        This helps ensure all features contribute equally to distance calculations
        regardless of their original scale.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing numeric features to normalize.
            
        Returns
        -------
        pd.DataFrame
            Normalized data with mean=0 and std=1.
        """
        if not hasattr(self, "means"):
            self.means = X.mean()
            # replace zero std with 1 to avoid division by zero
            self.stds = X.std().replace(0, 1)
        
        return (X - self.means) / self.stds

    def _vectorized_distances(self, X, sample):
        """
        Calculate the Euclidean distance between a sample and each row of X in a vectorized way,
        using only the columns where both samples have data.
        
        This handles missing values by only using dimensions where both the sample and
        reference points have non-missing values for distance calculation.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Reference dataset to compare against.
        sample : pd.Series or np.ndarray
            Sample to calculate distances from.
            
        Returns
        -------
        np.ndarray
            Array of distances from the sample to each row in X.
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        sample_values = sample.values if isinstance(sample, pd.Series) else sample
        
        # create mask for positions where both X and sample have valid values
        valid_mask = ~(np.isnan(X_values) | np.isnan(sample_values))
        valid_counts = np.sum(valid_mask, axis=1)
        
        # calculate differences, setting invalid positions to zero
        diff = np.where(valid_mask, X_values - sample_values, 0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        # set distance to infinity if no valid dimensions exist
        distances[valid_counts == 0] = np.inf
        
        return distances

    def _find_neighbors(self, X, sample):
        """
        Find the indices of the k nearest neighbors (using vectorized operations).
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Reference dataset to find neighbors in.
        sample : pd.Series or np.ndarray
            Sample to find neighbors for.
            
        Returns
        -------
        list
            List of indices for the k nearest neighbors.
        """
        distances = self._vectorized_distances(X, sample)
        neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        return neighbor_indices.tolist()

    def _calculate_weights(self, X, sample, neighbors):
        """
        Calculate the weights for neighbors based on distance.
        
        For 'uniform' weighting, all neighbors have equal weight.
        For 'distance' weighting, neighbors are weighted by inverse distance.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Reference dataset.
        sample : pd.Series or np.ndarray
            Sample to calculate weights for.
        neighbors : list
            List of indices of neighbor points.
            
        Returns
        -------
        np.ndarray
            Array of weights for each neighbor, summing to 1.
        """
        if self.weights == "uniform":
            return np.ones(len(neighbors))
        
        distances_all = self._vectorized_distances(X, sample)
        distances = distances_all[neighbors]
        
        # avoid division by zero by replacing zeros with small value
        distances = np.where(distances == 0, 1e-10, distances)
        w = 1.0 / distances
        return w / np.sum(w)

    def fit(self, X, verbose=False):
        """
        Fit the imputer to the reference data.
        
        This stores the reference dataset, identifies numeric and categorical columns,
        normalizes numeric features, and calculates modes for categorical features.
        
        Parameters
        ----------
        X : pd.DataFrame
            The reference dataset to fit the imputer.
        verbose : bool, default=False
            If True, prints additional information.
            
        Returns
        -------
        self : KNNImputer
            The fitted imputer.
        """
        self.X_reference = X.copy()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if not X[self.numeric_cols].empty:
            self.X_norm_reference = self._normalize_data(X[self.numeric_cols])
        
        self.cat_modes = {col: X[col].mode()[0] for col in self.categorical_cols}
        self.fitted = True
        return self

    def transform(self, X, verbose=False):
        """
        Transform (impute) the data using the fitted imputer.
        
        This replaces missing values in numeric features using KNN imputation
        and missing values in categorical features using mode imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The dataset with missing values to impute.
        verbose : bool, default=False
            If True, prints additional information.
            
        Returns
        -------
        pd.DataFrame or np.ndarray
            The imputed dataset, as DataFrame or ndarray depending on return_df.
        """
        if not self.fitted:
            raise ValueError("The imputer must be fitted before calling transform.")
            
        X_imputed = X.copy()
        X_numeric = X[self.numeric_cols].copy() if not self.numeric_cols.empty else pd.DataFrame()

        if not X_numeric.empty:
            X_norm = self._normalize_data(X_numeric)
            missing_idx = X_numeric.index[X_numeric.isna().any(axis=1)]
            
            for idx in missing_idx:
                missing_features = X_numeric.columns[X_numeric.loc[idx].isna()].tolist()
                norm_sample = X_norm.loc[idx]
                neighbors = self._find_neighbors(self.X_norm_reference, norm_sample)
                if not neighbors:
                    continue
                
                weights = self._calculate_weights(self.X_norm_reference, norm_sample, neighbors)
                
                for feature in missing_features:
                    neighbor_values = self.X_reference.loc[neighbors, feature].dropna()
                    
                    if neighbor_values.empty:
                        imputed_value = X_numeric[feature].mean()
                    else:
                        if self.weights == "uniform":
                            imputed_value = neighbor_values.mean()
                        else:
                            # identify which neighbors have valid values for this feature
                            valid_neighbor_indices = [i for i, n_idx in enumerate(neighbors) if n_idx in neighbor_values.index]
                            
                            if not valid_neighbor_indices:
                                imputed_value = neighbor_values.mean()
                            else:
                                valid_weights = weights[valid_neighbor_indices]
                                
                                # handle potential numerical instability
                                if valid_weights.sum() > 1e-9:
                                    normalized_weights = valid_weights / valid_weights.sum()
                                    imputed_value = np.dot(neighbor_values.values, normalized_weights)
                                else:
                                    imputed_value = neighbor_values.mean()
                    
                    if not np.isnan(imputed_value):
                        X_imputed.loc[idx, feature] = imputed_value

        # handle categorical columns with simple mode imputation
        for col in self.categorical_cols:
            if X[col].isna().any():
                X_imputed[col] = X_imputed[col].fillna(self.cat_modes[col])
        
        return X_imputed if self.return_df else X_imputed.values

    def fit_transform(self, X, verbose=False):
        """
        Fit the imputer and transform the data in one step.
        
        This is more efficient than calling fit() and transform() separately
        when imputing the same dataset that is used for fitting.
        
        Parameters
        ----------
        X : pd.DataFrame
            The dataset with missing values to fit and impute.
        verbose : bool, default=False
            If True, prints additional information.
            
        Returns
        -------
        pd.DataFrame or np.ndarray
            The imputed dataset, as DataFrame or ndarray depending on return_df.
        """
        return self.fit(X, verbose).transform(X, verbose)
