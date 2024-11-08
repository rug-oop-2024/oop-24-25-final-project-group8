from typing import List, Tuple, Dict
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeaturePreprocessor:
    def __call__(self, features: List[Feature], dataset: Dataset) -> List[Tuple[str, np.ndarray, dict]]:
        """
        Preprocess features.
        
        Args:
            features (List[Feature]): List of features.
            dataset (Dataset): Dataset object.

        Returns:
            List[Tuple[str, np.ndarray, dict]]: List of preprocessed features. Each tuple contains 
            the feature name, transformed numpy array, and an artifact dictionary.
        """
        results = []
        raw = dataset.read()  # Assumes Dataset has a `read()` method returning a DataFrame
        for feature in features:
            if feature.type == "categorical":
                encoder = OneHotEncoder()
                data = encoder.fit_transform(raw[feature.name].values.reshape(-1, 1)).toarray()
                artifact = {"type": "OneHotEncoder", "encoder": encoder}
                results.append((feature.name, data, artifact))
            elif feature.type == "numerical":
                scaler = StandardScaler()
                data = scaler.fit_transform(raw[feature.name].values.reshape(-1, 1))
                artifact = {"type": "StandardScaler", "scaler": scaler}
                results.append((feature.name, data, artifact))
            else:
                raise ValueError(f"Unsupported feature type: {feature.type}")
        
        # Sort for consistency by feature name
        results.sort(key=lambda x: x[0])
        return results