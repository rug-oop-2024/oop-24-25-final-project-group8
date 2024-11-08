from typing import List, Dict
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.feature_preprocessor import FeaturePreprocessor
import numpy as np
import os


class Pipeline(Artifact):
    
    def __init__(self, 
                 metrics: List[Metric],
                 dataset: Dataset, 
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 version="1.0.0",
                 name="pipeline_config",
                 ):
        super().__init__(name=name, type="pipeline", version=version)
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self):
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
            """
            Saves both the metadata (via Artifact) and the entire Pipeline object as a pickle file.
            """
            # Call the parent class's save method to save metadata
            super().save()

            # Path to save the pipeline object itself
            self.asset_path = f"{self.name}v{self.id}.pkl"
            data_path = os.path.join("assets", "objects", self.asset_path)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            # Serialize the entire Pipeline object as a pickle file
            with open(data_path, 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """Load the Pipeline object from a pickle file."""
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    @property
    def model(self):
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts
    
    def _register_artifact(self, name: str, artifact):
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        # Preprocess target feature
        target_preprocessor = FeaturePreprocessor([self._target_feature], self._dataset)
        target_feature_name, target_data, target_artifact = target_preprocessor.preprocess()[0]
        
        # Register the target artifact
        self._register_artifact(target_feature_name, target_artifact)
        self._output_vector = target_data
        
        # Preprocess input features
        input_preprocessor = FeaturePreprocessor(self._input_features, self._dataset)
        input_results = input_preprocessor.preprocess()
        
        # Register each input feature artifact and collect data for input vectors
        self._input_vectors = []
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
            self._input_vectors.append(data)

    def _split_data(self):
        # Split the data into training and testing sets
        split = float(self._split/100)
        self._train_X = [vector[:int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

        # Debugging statements to check split sizes
        if len(self._test_y) == 0:
            print("Warning: Test set is empty. Adjust the split ratio or check data size.")

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self):
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, X: np.ndarray, Y: np.ndarray) -> List[tuple]:
        """
        Evaluate the model on the given data and compute metrics.
        :param X: Input data (features).
        :param Y: Target data (ground truth).
        :return: List of tuples containing the metric and the result.
        """
        metrics_results = []

        # Generate predictions
        predictions = self._model.predict(X)

        # Ensure `predictions` is a 1D array of class labels
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)

        # If Y is one-hot encoded, convert `predictions` to one-hot
        if Y.ndim > 1:  # Y is one-hot encoded
            num_classes = Y.shape[1]
            predictions = np.eye(num_classes)[predictions]  # Convert to one-hot encoding

        for metric in self._metrics:
            result = metric(predictions, Y)
            metrics_results.append((metric, result))

        return metrics_results
    
    def execute(self) -> Dict[str, List[tuple]]:
        """
        Execute the machine learning pipeline by performing preprocessing, 
        splitting the data, training the model, and evaluating the model 
        on both the training and test datasets.
        
        The pipeline performs the following steps:
        - Preprocesses the input features and target feature.
        - Splits the data into training and testing sets based on the specified split ratio.
        - Trains the model on the training data.
        - Evaluates the model's performance on both the training and test datasets.
        - Returns the evaluation metrics for both sets.
        
        Returns:
            Dict[str, List[tuple]]: A dictionary containing the evaluation metrics
            for the training and test sets. Each entry in the dictionary has the 
            following structure:
                - "train_metrics": List of tuples, each containing a metric and its result for the training set.
                - "test_metrics": List of tuples, each containing a metric and its result for the test set.
        """
        # Preprocess and split data
        self._preprocess_features()
        self._split_data()

        # Train the model
        self._train()

        # Evaluate on training data
        train_X = self._compact_vectors(self._train_X)
        train_y = self._train_y
        train_metrics_results = self._evaluate(train_X, train_y)

        # Evaluate on test data (evaluation data)
        test_X = self._compact_vectors(self._test_X)
        test_y = self._test_y
        test_metrics_results = self._evaluate(test_X, test_y)

        # Return the metrics for both training and test sets
        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
        }
        

    