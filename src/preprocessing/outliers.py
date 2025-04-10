import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Set, Tuple, Union
import os

def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                        return_indices: bool = False) -> Union[Dict, Tuple[Dict, Set]]:
    """
    Detecta outliers en un DataFrame usando el método IQR.
    
    Args:
        df: DataFrame de pandas
        columns: Lista de columnas a analizar. Si es None, se analizan todas las columnas numéricas.
        return_indices: Si es True, devuelve los índices de las filas con outliers.
        
    Returns:
        dict: Diccionario con las columnas como claves y los valores outliers como valores.
        Si return_indices=True, devuelve también un conjunto con los índices de las filas con outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    outliers = {}
    outlier_indices = set()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outliers[column] = column_outliers
        
        if return_indices:
            outlier_indices.update(column_outliers.index)
    
    if return_indices:
        return outliers, outlier_indices
    return outliers

def replace_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                         method: str = 'winsorize', inplace: bool = False, return_params: bool = False,
                         params: Optional[Dict[str, Dict[str, float]]] = None, target_column: Optional[str] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Reemplaza outliers en un DataFrame usando el método IQR.
    
    Args:
        df: DataFrame de pandas
        columns: Lista de columnas a analizar. Si es None, se analizan todas las columnas numéricas.
        method: Método para reemplazar outliers ('winsorize', 'median', 'mean', 'nearest_valid')
        inplace: Si es True, modifica el DataFrame original.
        return_params: Si es True, devuelve también los parámetros IQR calculados.
        params: Parámetros IQR precalculados (opcional).
        target_column: Columna objetivo que se excluirá del análisis de outliers.
        
    Returns:
        DataFrame con outliers reemplazados y, opcionalmente, los parámetros IQR.
    """
    if not inplace:
        df = df.copy()
    
    if columns is None:
        # Excluir la columna objetivo del análisis si se proporciona
        if target_column is not None:
            numeric_columns = df.select_dtypes(include=['number']).columns
            columns = [col for col in numeric_columns if col != target_column]
        else:
            columns = df.select_dtypes(include=['number']).columns
    elif target_column is not None:
        # Si se proporcionan columnas específicas, asegurarse de excluir la columna objetivo
        columns = [col for col in columns if col != target_column]
    
    if params is None:
        params = {}
    
    for column in columns:
        # Si ya se tienen parámetros para la columna, úsalos
        if column in params:
            lower_bound = params[column]['lower_bound']
            upper_bound = params[column]['upper_bound']
        else:
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
        
        # Crear máscaras para detectar outliers
        lower_mask = df[column] < lower_bound
        upper_mask = df[column] > upper_bound
        
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

def get_iqr_bounds(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calcula los límites IQR para cada columna en el DataFrame.
    
    Args:
        df: DataFrame de pandas
        columns: Lista de columnas a analizar. Si es None, se analizan todas las columnas numéricas.
        
    Returns:
        Dict: Diccionario con las columnas como claves y un diccionario con los límites como valores.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    bounds = {}
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        bounds[column] = {
            'lower_bound': Q1 - 1.5 * IQR,
            'upper_bound': Q3 + 1.5 * IQR,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
    
    return bounds
