from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from autoop.core.ml.artifact import Artifact
from typing import Dict, Optional, Any


class Model(Artifact, ABC):
    """
    Represents a generic Machine Learning model.
    This class serves as a base class for both classification and regression tasks.
    """

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the model with optional hyperparameters.

        Args:
            hyperparameters (Optional[Dict[str, Any]]): A dictionary of hyperparameters for the model.
        """
        super().__init__(type="model")  # Pass `type` to Artifact initializer
        self._hyperparameters = hyperparameters or {}
        self._parameters = {}  # Initialize _parameters to store model-specific parameters

        # Automatically populate hyperparameters with defaults defined in subclass fields
        for name, field in self.__class__.__fields__.items():
            self._hyperparameters[name] = self._hyperparameters.get(name, field.default)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the observations and ground truth data.

        Args:
            observations (np.ndarray): Input data or features the model will learn from.
            ground_truth (np.ndarray): Actual target values used for training.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model for the given observations.

        Args:
            observations (np.ndarray): Input data for which predictions are to be made.

        Returns:
            np.ndarray: Predictions generated by the model.
        """
        pass

    def _validate_input(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Validates that input data is of the correct shape and dimensionality.

        Args:
            observations (np.ndarray): Input data for training or predicting.
            ground_truth (np.ndarray): Actual values used during training.

        Raises:
            ValueError: If dimensions or number of samples/features do not match.
        """
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                f"Mismatch in number of samples: Observations have {observations.shape[0]} samples, "
                f"but ground truth has {ground_truth.shape[0]} samples."
            )
        if observations.ndim != 2:
            raise ValueError("Observations should be a 2-D array.")

        num_samples, num_features = observations.shape
        if num_samples < 1 or num_features < 1:
            raise ValueError("The input data must have at least one sample and one feature.")

        if num_features > num_samples:
            print(
                "Warning: Detected more features than samples! "
                "Are you sure that samples are rows and features are columns?"
            )

        self._parameters["num_features"] = num_features

    def _validate_num_features(self, observations: np.ndarray) -> None:
        """
        Ensures the number of features matches between training and prediction.

        Args:
            observations (np.ndarray): The input data for predictions.

        Raises:
            ValueError: If the feature count during prediction doesn't match training.
        """
        num_features = self._parameters.get("num_features")
        if num_features is not None and observations.shape[1] != num_features:
            raise ValueError(
                f"Number of dimensions from fitting the data ({num_features}) does not match input "
                f"observations ({observations.shape[1]})."
            )

    @abstractmethod
    def _validate_fit(self) -> None:
        """Validates if the model has been fitted properly."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Retrieve the model parameters.

        Returns:
            Dict[str, Any]: The model's parameters.
        """
        return deepcopy(self._parameters)

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """
        Retrieve the model hyperparameters.

        Returns:
            Dict[str, Any]: The model's hyperparameters.
        """
        return deepcopy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Sets the model hyperparameters.

        Args:
            hyperparameters (Dict[str, Any]): Dictionary of hyperparameters.
        """
        self._hyperparameters = deepcopy(hyperparameters)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Sets the model parameters (e.g., weights, biases).

        Args:
            params (Dict[str, Any]): Dictionary of parameters.
        """
        self._parameters = deepcopy(params)

    def save(self, file_path: str) -> None:
        """
        Save the model's parameters and hyperparameters to a file.

        Args:
            file_path (str): Path to save the model data.
        """
        model_data = {
            "parameters": self._parameters,
            "hyperparameters": self._hyperparameters,
        }
        super().save(data=model_data, artifact_type="model", filename=file_path)

    def load(self, file_path: str) -> None:
        """
        Load the model's parameters and hyperparameters from a file.

        Args:
            file_path (str): Path from which to load the model data.
        """
        super().load(file_path)
        self._parameters = self.data.get("parameters", {})
        self._hyperparameters = self.data.get("hyperparameters", {})
