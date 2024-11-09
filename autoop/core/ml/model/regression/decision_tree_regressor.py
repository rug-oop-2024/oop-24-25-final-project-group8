from sklearn.tree import DecisionTreeRegressor
from pydantic import PrivateAttr, Field
from typing import Literal
import numpy as np
from autoop.core.ml.model import Model


class DecisionTreeRegressorModel(Model):
    """
    Wrapper around scikit-learn's DecisionTreeRegressor, integrating it with the Model
    interface.
    Provides methods for fitting the model and making predictions, while storing
    model-specific
    hyperparameters and fitted parameters.
    """

    _model: DecisionTreeRegressor = PrivateAttr()

    criterion: Literal["absolute_error", "squared_error", "poisson", "friedman_mse"] = (
        Field(
            default="friedman_mse",
            description="The function to measure the quality of a split.",
        )
    )

    def __init__(self, **data) -> None:
        """
        Initialize the DecisionTreeRegressorModel with specified hyperparameters.

        Args:
            data: Keyword arguments representing model hyperparameters.
                  Expected keys include 'criterion'.
        """
        super().__init__(**data)
        # Initialize _hyperparameters with only fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "decision tree regressor"
        self.type = "regression"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Decision Tree Regressor model to the provided data.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).

        Raises:
            ValueError: If there is a mismatch in dimensions between observations and
            ground truth.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize and fit the DecisionTreeRegressor with specified criterion
        self._model = DecisionTreeRegressor(criterion=self.criterion)
        self._model.fit(observations, ground_truth)

        # Store the fitted model's tree attribute in _parameters for future validation
        self._parameters["tree_"] = self._model.tree_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Decision Tree Regressor model.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features) for which predictions are to be made.

        Returns:
            np.ndarray: Predicted values for the provided observations.

        Raises:
            ValueError: If the model has not been fitted or if the input features
            do not match the expected dimensions from training.
        """
        # Validate that the model is fitted and that input dimensions match training
        # dimensions
        self._validate_fit()
        super()._validate_num_features(observations)

        return self._model.predict(observations)

    def _validate_fit(self) -> None:
        """
        Check if the model has been fitted by verifying the presence of a decision tree.

        Raises:
            ValueError: If the model has not been trained, indicated by the absence of
            a tree.
        """
        if not hasattr(self._model, "tree_"):
            raise ValueError("The model has not been trained yet!")
