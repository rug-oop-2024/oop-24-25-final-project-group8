from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

class FeatureTypeDetector:
    """
    A class to detect types of features in a dataset.

    Attributes:
        dataset (Dataset): The dataset to analyze for feature types.
    """
    
    def __init__(self, dataset: Dataset) -> None:
        """
        Initializes the FeatureTypeDetector with a dataset.

        Args:
            dataset (Dataset): The dataset containing the features.
        """
        self.dataset = dataset

    def detect_feature_types(self) -> List[Feature]:
        """
        Detects feature types in the dataset. Assumes only categorical and numerical features.

        Returns:
            List[Feature]: A list of features with their detected types.
            
        Raises:
            ValueError: If a feature has unrecognized data types or no non-None values.
        """
        df = self.dataset.read()
        features = []

        for feature_name in df.columns:
            feature_type = self._determine_feature_type(df[feature_name], feature_name)
            feature = Feature(name=feature_name, type=feature_type)
            features.append(feature)
        
        return features

    def _determine_feature_type(self, column_data, feature_name: str) -> str:
        """
        Determines the type of a single feature based on its values.

        Args:
            column_data (pd.Series): The column data from the dataset.
            feature_name (str): The name of the feature.

        Returns:
            str: The type of the feature, either 'numerical' or 'categorical'.
            
        Raises:
            ValueError: If no non-None values exist in the feature or an unrecognized data type is found.
        """
        for value in column_data:
            if value is not None:
                if isinstance(value, (int, float)):
                    return 'numerical'
                elif isinstance(value, str):
                    return 'categorical'
                else:
                    raise ValueError(f"Unrecognized data type for feature {feature_name}: {type(value)}")
        raise ValueError(f"Feature {feature_name} contains no non-None values to determine type")
