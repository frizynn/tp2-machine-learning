import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class CategoricalEncoder:
    """
    Encoder for categorical features using one-hot encoding with category tracking.
    
    Parameters
    ----------
    categorical_columns : Optional[List[str]]
        List of columns to treat as categorical. If None, automatically detected.
    drop_first : bool, default False
        Whether to drop the first category in each feature to avoid multicollinearity.
    """
    def __init__(self, categorical_columns: Optional[List[str]] = None, drop_first: bool = False):
        self.categorical_columns = categorical_columns
        self.drop_first = drop_first
        self.categorical_mappings: Dict[str, np.ndarray] = {}
        self.encoded_columns: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> 'CategoricalEncoder':
        # detect categorical columns by type if not specified
        if self.categorical_columns is None:
            self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_columns:
            # store the set of detected categories using pd.unique
            self.categorical_mappings[col] = pd.unique(X[col]) if col in X.columns else np.array([])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call 'fit' before 'transform'.")
        
        # get non-categorical columns
        non_cat = X.drop(columns=self.categorical_columns, errors='ignore')
        encoded_list = []

        # transform each categorical column
        for col in self.categorical_columns:
            if col in X.columns:
                # assign learned categories; unknown categories will be encoded as NaN
                cat_col = pd.Categorical(X[col], categories=self.categorical_mappings[col])
                dummies = pd.get_dummies(cat_col, prefix=col, drop_first=self.drop_first)
            else:
                # if column is not in X, generate dummy columns with 0 values
                temp_cat = pd.Categorical(self.categorical_mappings[col], categories=self.categorical_mappings[col])
                dummies = pd.get_dummies(temp_cat, prefix=col, drop_first=self.drop_first)
                dummies = pd.DataFrame(0, index=X.index, columns=dummies.columns)
            
            encoded_list.append(dummies)
            # accumulate encoded column names without duplicates
            self.encoded_columns.extend([c for c in dummies.columns if c not in self.encoded_columns])
        
        # concatenate non-categorical columns and dummies along the columns
        return pd.concat([non_cat] + encoded_list, axis=1)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with features to encode.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features.
        """
        return self.fit(X).transform(X)
    
    @staticmethod
    def encode_categorical(X: pd.DataFrame, categorical_columns: Optional[List[str]] = None, drop_first: bool = False) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with features to encode.
        categorical_columns : Optional[List[str]], default None
            List of columns to treat as categorical. If None, automatically detected.
        drop_first : bool, default False
            Whether to drop the first category in each feature to avoid multicollinearity.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features.
        """
        if categorical_columns is None:
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_columns:
            return X
        encoder = CategoricalEncoder(categorical_columns=categorical_columns, drop_first=drop_first)
        return encoder.fit_transform(X)
