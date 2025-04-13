import pandas as pd
from typing import List, Optional, Dict, Tuple, Union


def replace_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                         method: str = 'winsorize', inplace: bool = False, return_params: bool = False,
                         params: Optional[Dict[str, Dict[str, float]]] = None, target_column: Optional[str] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Replaces outliers in a DataFrame using the IQR method.
    
    Args:
        df: Pandas DataFrame
        columns: List of columns to analyze. If None, all numeric columns are analyzed.
        method: Method to replace outliers ('winsorize', 'median', 'mean', 'nearest_valid')
            - winsorize: Caps values at lower/upper bounds
            - median: Replaces outliers with median value
            - mean: Replaces outliers with mean value
            - nearest_valid: Replaces outliers with closest valid boundary
        inplace: If True, modifies the original DataFrame.
        return_params: If True, also returns the calculated IQR parameters.
        params: Pre-calculated IQR parameters (optional).
        target_column: Target column to exclude from outlier analysis.
        
    Returns:
        DataFrame with replaced outliers and, optionally, the IQR parameters.
    """
    if not inplace:
        df = df.copy()
    
    if columns is None:
        # exclude target column from analysis if provided
        if target_column is not None:
            numeric_columns = df.select_dtypes(include=['number']).columns
            columns = [col for col in numeric_columns if col != target_column]
        else:
            columns = df.select_dtypes(include=['number']).columns
    elif target_column is not None:
        # if specific columns are provided, ensure target column is excluded
        columns = [col for col in columns if col != target_column]
    
    if params is None:
        params = {}
    
    for column in columns:
        # use existing parameters if available
        if column in params:
            lower_bound = params[column]['lower_bound']
            upper_bound = params[column]['upper_bound']
        else:
            # calculate iqr bounds using 1.5*iqr rule
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if return_params:
                params[column] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
        
        # create masks to identify outliers
        lower_mask = df[column] < lower_bound
        upper_mask = df[column] > upper_bound
        
        # apply the selected replacement method to outliers
        if method == 'winsorize':
            df.loc[lower_mask, column] = lower_bound
            df.loc[upper_mask, column] = upper_bound
        elif method == 'median':
            median_value = df[column].median()
            df.loc[lower_mask | upper_mask, column] = median_value
        elif method == 'mean':
            mean_value = df[column].mean()
            df.loc[lower_mask | upper_mask, column] = mean_value
        elif method == 'nearest_valid':
            if any(lower_mask):
                df.loc[lower_mask, column] = lower_bound
            if any(upper_mask):
                df.loc[upper_mask, column] = upper_bound
    
    if return_params:
        return df, params
    return df
