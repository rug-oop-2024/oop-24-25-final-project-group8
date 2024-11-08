from typing import List, Tuple, Dict
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeaturePreprocessor:
    """
    A class to preprocess features in a dataset.

    Attributes:
        features (List[Feature]): List of features to preprocess.
        dataset (Dataset): Dataset object containing the data.
    """
    
    def __init__(self, features: List[Feature], dataset: Dataset) -> None:
        """
        Initialize the FeaturePreprocessor with features and dataset.

        Args:
            features (List[Feature]): List of features to preprocess.
            dataset (Dataset): Dataset containing raw data to preprocess.
        """
        self.features = features
        self.dataset = dataset

    def preprocess(self) -> List[Tuple[str, np.ndarray, Dict[str, object]]]:
        """
        Preprocess all features based on their types (categorical or numerical).

        Returns:
            List[Tuple[str, np.ndarray, Dict[str, object]]]: A list of tuples with the feature name, 
            preprocessed data array, and an artifact dictionary for each feature.
        """
        results = []
        raw_data = self.dataset.read()

        for feature in self.features:
            if feature.type == "categorical":
                processed_feature = self._process_categorical_feature(feature, raw_data)
            elif feature.type == "numerical":
                processed_feature = self._process_numerical_feature(feature, raw_data)
            else:
                raise ValueError(f"Unsupported feature type: {feature.type}")
                
            results.append(processed_feature)

        # Sort results for consistency
        results.sort(key=lambda x: x[0])
        return results

    def _process_categorical_feature(self, feature: Feature, data: pd.DataFrame) -> Tuple[str, np.ndarray, Dict[str, object]]:
        """
        Preprocess a categorical feature using one-hot encoding.

        Args:
            feature (Feature): The categorical feature to preprocess.
            data (pd.DataFrame): The raw dataset.

        Returns:
            Tuple[str, np.ndarray, Dict[str, object]]: The feature name, encoded array, and encoder artifact.
        """
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(data[feature.name].values.reshape(-1, 1)).toarray()
        artifact = {"type": "OneHotEncoder", "params": encoder.get_params()}
        return feature.name, encoded_data, artifact

    def _process_numerical_feature(self, feature: Feature, data: pd.DataFrame) -> Tuple[str, np.ndarray, Dict[str, object]]:
        """
        Preprocess a numerical feature using standard scaling.

        Args:
            feature (Feature): The numerical feature to preprocess.
            data (pd.DataFrame): The raw dataset.

        Returns:
            Tuple[str, np.ndarray, Dict[str, object]]: The feature name, scaled array, and scaler artifact.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[feature.name].values.reshape(-1, 1))
        artifact = {"type": "StandardScaler", "params": scaler.get_params()}
        return feature.name, scaled_data, artifact