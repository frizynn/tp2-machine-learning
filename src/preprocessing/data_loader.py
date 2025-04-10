from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .categorical_encoder import CategoricalEncoder
import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    """Configuration for dataset splitting"""
    test_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""
    data_dir: Path
    target_column: str
    split_config: SplitConfig = field(default_factory=SplitConfig)


class DataLoader:
    """
    Class for loading and preparing datasets for machine learning.
    
    Parameters
    ----------
    config : DatasetConfig
        Configuration for dataset loading and preprocessing.
    """
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.target_column = config.target_column

        # Dataframes and processed data
        self.df_dev = None
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None

        self.mean = None
        self.std = None

    def load_data(self, train_file: Optional[str] = None, valid_file: Optional[str] = None,
                  test_file: Optional[str] = None, dev_file: Optional[str] = None,
                  splitted: bool = True) -> 'DataLoader':
        """
        Load data from CSV files.
        
        Parameters
        ----------
        train_file : str, optional
            Path to the training dataset.
        valid_file : str, optional
            Path to the validation dataset.
        test_file : str, optional
            Path to the test dataset.
        dev_file : str, optional
            Path to the development dataset.
        splitted : bool, default=True
            If True, load data as already splitted into training, validation and test sets.
            
        Returns
        -------
        self : DataLoader
            Instance with loaded data.
        """
        if splitted:
            if train_file is None:
                raise ValueError("train_file must be provided when splitted=True")
            self.df_train = pd.read_csv(self.data_dir / train_file)
            if valid_file:
                self.df_valid = pd.read_csv(self.data_dir / valid_file)
            if test_file:
                self.df_test = pd.read_csv(self.data_dir / test_file)
        else:
            if dev_file:
                self.df_dev = pd.read_csv(self.data_dir / dev_file)
            elif train_file:
                self.df_train = pd.read_csv(self.data_dir / train_file)
                self.df_dev = self.df_train.copy()
            else:
                raise ValueError("Either train_file or dev_file must be provided when splitted=False")
            if test_file:
                self.df_test = pd.read_csv(self.data_dir / test_file)
        return self

    def train_test_split(self, return_splitted: bool = True, random_state: int = 42,
                         return_numpy: bool = False, encode_categorical: bool = True,
                         normalize: bool = False, is_test: bool = False
                         ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]:
        """
        Split the data into training and testing sets.
        
        Parameters
        ----------
        return_splitted : bool, default=True
            If True, return X_train, X_test, y_train, y_test.
            Otherwise, return X and y (full dataset).
        random_state : int, default=42
            Random seed for reproducibility.
        return_numpy : bool, default=False
            If True, return numpy arrays instead of pandas objects.
        encode_categorical : bool, default=True
            If True, encode categorical features.
        normalize : bool, default=False
            If True, normalize the data.
        is_test : bool, default=False
            If True, use stored mean and std (from training) for normalization.
        
        Returns
        -------
        Depending on return_splitted:
            - If True: X_train, X_test, y_train, y_test.
            - Otherwise: X, y.
        """
        if self.df_dev is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        # Prepare features and target from the development set
        X = self.df_dev.drop(columns=[self.target_column])
        y = self.df_dev[self.target_column]

        if encode_categorical:
            X = DataLoader.encode_categorical(X)

        self.feature_names = X.columns.tolist()

        if not return_splitted:
            X = self._normalize(X) if normalize and not (is_test and self.mean is not None and self.std is not None) \
                else (self._normalize(X, self.mean, self.std) if normalize else X)
            return (X.to_numpy(), y.to_numpy()) if return_numpy else (X, y)

        X_train, X_test, y_train, y_test = self._train_test_split(
            X, y,
            test_size=self.config.split_config.test_size,
            random_state=random_state,
            shuffle=self.config.split_config.shuffle
        )

        if normalize:
            X_train = self._normalize(X_train)
            X_test = self._normalize(X_test, self.mean, self.std)

        if return_numpy:
            X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
            y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

        return X_train, X_test, y_train, y_test

    def _normalize(self, X: pd.DataFrame, mean: Optional[pd.Series] = None,
                   std: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Normalize the data using z-score normalization (mean=0, std=1).
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to normalize.
        mean : Optional[pd.Series], default=None
            Mean for each feature. If None, calculated from X.
        std : Optional[pd.Series], default=None
            Standard deviation for each feature. If None, calculated from X.
            
        Returns
        -------
        pd.DataFrame
            Normalized data.
        """
        if mean is None or std is None:
            self.mean = X.mean()
            self.std = X.std().replace(0, 1)
        else:
            self.mean, self.std = mean, std
        return (X - self.mean) / self.std

    @staticmethod
    def encode_categorical(X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with features to encode.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features.
        """
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if not categorical_columns:
            return X
        encoder = CategoricalEncoder(categorical_columns=categorical_columns, drop_first=True)
        return encoder.fit_transform(X)

    def _train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float,
                          random_state: int, shuffle: bool
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets using a reproducible permutation.
        """
        n_samples = len(X)
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
        n_test = int(n_samples * test_size)
        test_indices, train_indices = indices[:n_test], indices[n_test:]
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

    def get_pandas_data(self, splitted: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Retrieve data as pandas DataFrames and Series.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test as pandas objects.
        """
        if self.df_dev is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        if self.df_test is None:
            return self.df_dev
        if splitted:
            return self.df_train, self.df_valid, self.df_test
        return self.df_dev, self.df_test

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns
        -------
        List[str]
            Feature names.
        """
        return self.feature_names

    def get_class_names(self) -> List[str]:
        """
        Get the list of class names.
        
        Returns
        -------
        List[str]
            Class names.
        """
        return self.class_names

    def save_processed_data(self, df_train_dir: Optional[Path] = None,
                            df_valid_dir: Optional[Path] = None,
                            df_test_dir: Optional[Path] = None,
                            df_dev_dir: Optional[Path] = None,
                            verbose: bool = False) -> None:
        """
        Save the processed data to CSV files.
        
        Parameters
        ----------
        df_train_dir : Optional[Path], default=None
            Path to save the training data.
        df_valid_dir : Optional[Path], default=None
            Path to save the validation data.
        df_test_dir : Optional[Path], default=None
            Path to save the test data.
        df_dev_dir : Optional[Path], default=None
            Path to save the development data.
        """
        if not any([df_train_dir, df_valid_dir, df_test_dir, df_dev_dir]):
            raise ValueError("Provide at least one path to save data (train, valid, test, or dev).")
            
        datasets = [
            (self.df_train, df_train_dir, "Training"),
            (self.df_valid, df_valid_dir, "Validation"),
            (self.df_test, df_test_dir, "Test"),
            (self.df_dev, df_dev_dir, "Development")
        ]
        for df, path, name in datasets:
            if path is not None:
                if df is None:
                    raise ValueError(f"{name} data not loaded yet. Call load_data() first.")
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                if verbose:
                    print(f"{name} data saved to {path}")
                if name == "Training" and self.class_names is not None:
                    classes_path = path.parent / "class_names.txt"
                    with open(classes_path, "w") as f:
                        for class_name in self.class_names:
                            f.write(f"{class_name}\n")
                    if verbose:
                        print(f"Class names saved to {classes_path}")

    def update(self, df_train: pd.DataFrame = None,
               df_valid: pd.DataFrame = None,
               df_test: pd.DataFrame = None,
               df_dev: pd.DataFrame = None) -> 'DataLoader':
        """
        Update the datasets with new data.

        Parameters
        ----------
        df_train : pd.DataFrame, optional
            New training data.
        df_valid : pd.DataFrame, optional
            New validation data.
        df_test : pd.DataFrame, optional
            New test data.
        df_dev : pd.DataFrame, optional
            New development data.

        Returns
        -------
        self : DataLoader
        """
        if df_train is not None:
            self.df_train = df_train
        if df_valid is not None:
            self.df_valid = df_valid
        if df_test is not None:
            self.df_test = df_test
        if df_dev is not None:
            self.df_dev = df_dev
        return self

    def get_processed_test_data(self, return_numpy: bool = True,
                                encode_categorical: bool = True,
                                normalize: bool = False
                                ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Process test data and prepare it for prediction.
        
        Parameters
        ----------
        return_numpy : bool, default=True
            If True, return numpy arrays.
        encode_categorical : bool, default=True
            If True, encode categorical features.
        normalize : bool, default=False
            If True, normalize data using stored training statistics.
            
        Returns
        -------
        Tuple with test features and target.
        """
        if self.df_test is None:
            raise ValueError("Test data not loaded yet. Call load_data() with test_file parameter first.")
        X_test = self.df_test.drop(columns=[self.target_column])
        y_test = self.df_test[self.target_column]
        if encode_categorical:
            X_test = DataLoader.encode_categorical(X_test)
        if normalize:
            if self.mean is None or self.std is None:
                raise ValueError("Normalization parameters not found. Ensure training data was processed with normalize=True.")
            X_test = self._normalize(X_test, self.mean, self.std)
        if return_numpy:
            return X_test.to_numpy(), y_test.to_numpy()
        return X_test, y_test
