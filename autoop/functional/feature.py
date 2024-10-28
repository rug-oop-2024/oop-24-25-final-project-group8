
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    # Read the dataset which is defined as a pandas df
    df = dataset.read()
    
    features = []
    
    # For column in the df
    for feature_name in df.columns:
        # Get the first non-NaN value in the column
        for value in df[feature_name]:
            if value is not None:
                # Check the type of the value
                if isinstance(value, (int, float)):
                    feature_type = 'numerical'
                elif isinstance(value, str):
                    feature_type = 'categorical'
                else:
                    raise ValueError(f"Unrecognized data type for feature {feature_name}: {type(value)}")
                break
        else:
            raise ValueError(f"Feature {feature_name} contains no non-None values to determine type")
        
        # Create the Feature object with the detected type
        feature = Feature(name=feature_name, type=feature_type)
        features.append(feature)
    
    return features