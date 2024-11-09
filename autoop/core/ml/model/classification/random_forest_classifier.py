from sklearn.ensemble import RandomForestClassifier
from pydantic import PrivateAttr, Field
from typing import Literal
import numpy as np
from autoop.core.ml.model import Model


class RandomForestClassifierModel(Model):
    """
    Wrapper around scikit-learn's RandomForestClassifier, integrating it with the Model
    interface.
    Provides methods for fitting the model and making predictions, while storing
    model-specific hyperparameters and fitted parameters.
    """

    # Private attribute to hold the model instance
    _model: RandomForestClassifier = PrivateAttr(default_factory=RandomForestClassifier)

    n_estimators: int = Field(
        default=100,
        ge=1,
        description="The number of trees in the forest; must be >= 1.",
    )
    criterion: Literal["gini", "entropy"] = Field(
        default="gini",
        description="The function to measure the quality of a split ('gini', 'entropy'",
    )

    def __init__(self, **data) -> None:
        """
        Initialize the RandomForestClassifierModel with specified hyperparameters.

        Args:
            data: Keyword arguments representing model hyperparameters.
                  Expected keys include 'n_estimators' and 'criterion'.
        """
        super().__init__(**data)
        # Initialize _hyperparameters with fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "random forest classifier"
        self.type = "classification"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Random Forest Classifier model to the provided data.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).

        Raises:
            ValueError: If there is a mismatch in dimensions between observations and
            ground truth.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize and fit the RandomForestClassifier with specified hyperparameters
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
        )
        self._model.fit(observations, ground_truth)

        # Store the fitted model's estimators in _parameters for future validation
        self._parameters["estimators_"] = self._model.estimators_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest Classifier model.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features) for which predictions are to be made.

        Returns:
            np.ndarray: Predicted class labels for the provided observations.

        Raises:
            ValueError: If the model has not been fitted or if the input features do not
            match the expected dimensions from training.
        """
        # Validate that the model is fitted and that input dimensions match training
        # dimensions
        self._validate_fit()
        super()._validate_num_features(observations)

        return self._model.predict(observations)

    def _validate_fit(self) -> None:
        """
        Check if the model has been fitted by verifying the presence of learned
        estimators.

        Raises:
            ValueError: If the model has not been trained, indicated by the absence of
            estimators.
        """
        if not hasattr(self._model, "estimators_"):
            raise ValueError("The model has not been fitted!")
