import numpy as np
import pandas as pd

class KNNImputer:
    """
    Imputer para completar valores faltantes usando k-Nearest Neighbors.

    Parámetros:
    -----------
    n_neighbors : int, default=5
        Número de vecinos a usar para imputación.
    weights : str, default='uniform'
        Función de ponderación: 'uniform' para pesos iguales o 'distance' para ponderar por la inversa de la distancia.
    metric : str, default='euclidean'
        Métrica de distancia a usar (actualmente solo se soporta 'euclidean').
    return_df : bool, default=True
        Si se devuelve un DataFrame de pandas o un array numpy.
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
        Normaliza los datos a media cero y varianza unitaria.
        """
        if not hasattr(self, "means"):
            self.means = X.mean()
            self.stds = X.std().replace(0, 1)
        return (X - self.means) / self.stds

    def _vectorized_distances(self, X, sample):
        """
        Calcula de forma vectorizada la distancia euclidiana entre un sample y cada fila de X,
        usando solo las columnas en las que ambas muestras tienen datos.
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        sample_values = sample.values if isinstance(sample, pd.Series) else sample
        valid_mask = ~(np.isnan(X_values) | np.isnan(sample_values))
        valid_counts = np.sum(valid_mask, axis=1)
        diff = np.where(valid_mask, X_values - sample_values, 0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        distances[valid_counts == 0] = np.inf
        return distances

    def _find_neighbors(self, X, sample):
        """
        Encuentra los índices de los k vecinos más cercanos (usando operaciones vectorizadas).
        """
        distances = self._vectorized_distances(X, sample)
        neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        return neighbor_indices.tolist()

    def _calculate_weights(self, X, sample, neighbors):
        """
        Calcula las ponderaciones para los vecinos en base a la distancia.
        """
        if self.weights == "uniform":
            return np.ones(len(neighbors))
        distances_all = self._vectorized_distances(X, sample)
        distances = distances_all[neighbors]
        distances = np.where(distances == 0, 1e-10, distances)
        w = 1.0 / distances
        return w / np.sum(w)

    def fit(self, X, verbose=False):
        """
        Ajusta el imputer a los datos de referencia.
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
        Transforma (imputa) los datos usando el imputer ajustado.
        """
        if not self.fitted:
            raise ValueError("El imputer debe ser ajustado (fit) antes de llamar a transform.")
            
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
                            valid_neighbor_indices = [i for i, n_idx in enumerate(neighbors) if n_idx in neighbor_values.index]
                            if not valid_neighbor_indices:
                                imputed_value = neighbor_values.mean()
                            else:
                                valid_weights = weights[valid_neighbor_indices]
                                if valid_weights.sum() > 1e-9:
                                    normalized_weights = valid_weights / valid_weights.sum()
                                    imputed_value = np.dot(neighbor_values.values, normalized_weights)
                                else:
                                    imputed_value = neighbor_values.mean()
                    if not np.isnan(imputed_value):
                        X_imputed.loc[idx, feature] = imputed_value

        for col in self.categorical_cols:
            if X[col].isna().any():
                X_imputed[col] = X_imputed[col].fillna(self.cat_modes[col])
        
        return X_imputed if self.return_df else X_imputed.values

    def fit_transform(self, X, verbose=False):
        """
        Ajusta e imputa a la vez los datos.
        """
        return self.fit(X, verbose).transform(X, verbose)
