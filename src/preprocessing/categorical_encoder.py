import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class CategoricalEncoder:
    """
    Encoder para features categóricos usando one-hot encoding con seguimiento de categorías.
    
    Parámetros:
        categorical_columns : Optional[List[str]]
            Lista de columnas a tratar como categóricas. Si es None se detectan automáticamente.
        drop_first : bool, default False
            Si se elimina la primera categoría en cada feature para evitar multicolinealidad.
    """
    def __init__(self, categorical_columns: Optional[List[str]] = None, drop_first: bool = False):
        self.categorical_columns = categorical_columns
        self.drop_first = drop_first
        self.categorical_mappings: Dict[str, np.ndarray] = {}
        self.encoded_columns: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> 'CategoricalEncoder':
        # Si no se especifica, se detectan las columnas categóricas por tipos
        if self.categorical_columns is None:
            self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_columns:
            # Se almacena el conjunto de categorías detectadas usando pd.unique
            self.categorical_mappings[col] = pd.unique(X[col]) if col in X.columns else np.array([])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder no está ajustado. Llama a 'fit' antes de 'transform'.")
        
        # Se obtienen las columnas no categóricas
        non_cat = X.drop(columns=self.categorical_columns, errors='ignore')
        encoded_list = []

        # Se transforma cada columna categórica
        for col in self.categorical_columns:
            if col in X.columns:
                # Se asignan las categorías aprendidas; las categorías desconocidas se codificarán como NaN
                cat_col = pd.Categorical(X[col], categories=self.categorical_mappings[col])
                dummies = pd.get_dummies(cat_col, prefix=col, drop_first=self.drop_first)
            else:
                # Si la columna no está en X, se generan las columnas dummy con valores 0
                temp_cat = pd.Categorical(self.categorical_mappings[col], categories=self.categorical_mappings[col])
                dummies = pd.get_dummies(temp_cat, prefix=col, drop_first=self.drop_first)
                dummies = pd.DataFrame(0, index=X.index, columns=dummies.columns)
            
            encoded_list.append(dummies)
            # Se acumulan los nombres de las columnas codificadas sin duplicados
            self.encoded_columns.extend([c for c in dummies.columns if c not in self.encoded_columns])
        
        # Se concatenan las columnas no categóricas y las dummies a lo largo de las columnas
        return pd.concat([non_cat] + encoded_list, axis=1)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Encoder no está ajustado. Llama a 'fit' antes de usar 'get_feature_names'.")
        return self.encoded_columns
