from typing import List, Dict
import pickle
import os
import numpy as np
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.feature_preprocessor import FeaturePreprocessor


class Pipeline(Artifact):
    
    def __init__(self, 
                 metrics: List[Metric],
                 dataset: Dataset, 
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 version: str = "1.0.0",
                 name: str = "pipeline_config") -> None:
        """
        Initializes the Pipeline object with specified parameters.
        
        Args:
            metrics (List[Metric]): A list of metrics to evaluate model performance.
            dataset (Dataset): The dataset to be used in the pipeline.
            model (Model): The machine learning model to train and evaluate.
            input_features (List[Feature]): Features for model input.
            target_feature (Feature): The target feature for prediction.
            split (float): Proportion of data to use for training (0 < split < 1).
            version (str): Version identifier for the pipeline.
            name (str): Name of the pipeline configuration.
        """
        super().__init__(name=name, type="pipeline", version=version)
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split if split <= 1 else split / 100  # Ensure split is between 0 and 1
        
        # Validate target and model compatibility
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """ Returns human readable representation of pipeline"""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    def save(self) -> None:
        """Saves the Pipeline object and metadata as a pickle file."""
        super().save()
        self.asset_path = f"{self.name}v{self.id}.pkl"
        data_path = os.path.join("assets", "objects", self.asset_path)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Pipeline":
        """Load the Pipeline object from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Retrieve artifacts generated during pipeline execution."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type == "OneHotEncoder":
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            elif artifact_type == "StandardScaler":
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts
    
    def _register_artifact(self, name: str, artifact) -> None:
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess target and input features."""
        # Preprocess the target feature
        target_preprocessor = FeaturePreprocessor()
        target_result = target_preprocessor([self._target_feature], self._dataset)[0]
        target_feature_name, target_data, target_artifact = target_result
        self._register_artifact(target_feature_name, target_artifact)
        self._output_vector = target_data

        # Preprocess input features
        input_preprocessor = FeaturePreprocessor()
        input_results = input_preprocessor(self._input_features, self._dataset)
        
        # Store processed input feature data
        self._input_vectors = []
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
            self._input_vectors.append(data)

    def _split_data(self) -> None:
        """Split data into training and testing sets."""
        split_idx = int(self._split * len(self._output_vector))
        self._train_X = [vector[:split_idx] for vector in self._input_vectors]
        self._test_X = [vector[split_idx:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split_idx]
        self._test_y = self._output_vector[split_idx:]
        if len(self._test_y) == 0:
            print("Warning: Test set is empty. Adjust the split ratio or check data size.")

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Combine multiple feature vectors into a single array."""
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model using the training dataset."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, X: np.ndarray, Y: np.ndarray) -> List[tuple]:
        """Evaluate the model on given data and compute metrics."""
        metrics_results = []
        predictions = self._model.predict(X)
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        if self.model.type == "classification" and Y.ndim > 1:
            num_classes = Y.shape[1]
            predictions = predictions.astype(int)
            predictions = np.eye(num_classes)[predictions]
        for metric in self._metrics:
            result = metric(predictions, Y)
            metrics_results.append((metric, result))
        return metrics_results
    
    def execute(self) -> Dict[str, List[tuple]]:
        """Execute the pipeline, including preprocessing, training, and evaluation."""
        self._preprocess_features()
        self._split_data()
        self._train()
        train_X = self._compact_vectors(self._train_X)
        train_y = self._train_y
        train_metrics_results = self._evaluate(train_X, train_y)
        test_X = self._compact_vectors(self._test_X)
        test_y = self._test_y
        test_metrics_results = self._evaluate(test_X, test_y)
        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
        }
