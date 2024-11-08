from sklearn.tree import DecisionTreeRegressor

from pydantic import PrivateAttr, Field
from typing import Literal, Optional

import numpy as np
from autoop.core.ml.model import Model

class DecisionTreeRegressorModel(Model):
    """
    Wrapper around the DecisionTreeRegressor from scikit-learn.
    """

    _model: DecisionTreeRegressor = PrivateAttr()

    criterion: Literal['absolute_error', 'squared_error', 'poisson', 'friedman_mse'] = Field(default='friedman_mse', description="The function to measure the quality of a split.")

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize _hyperparameters with only fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "decision tree regressor"
        self.type = "regression"
        self._parameters ={}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Decision Tree Regressor model to the provided data.
        :param observations: Input data (features).
        :param ground_truth: Target values.
        """
        super()._validate_input(observations, ground_truth)

        self._model = DecisionTreeRegressor(
            criterion=self.criterion,
        )

        self._model.fit(observations, ground_truth)

        self._parameters["tree_"] = self._model.tree_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Use the model to make predictions on new observations.
        :param observations: Input data to predict on.
        :return: Predicted values.
        """

        self._validate_fit()
        super()._validate_num_features(observations)

        return self._model.predict(observations)

    def _validate_fit(self) -> None:
        """
        Validate that the model has been fitted.
        Raises:
            ValueError: If the model tree is missing.
        """
        if not hasattr(self._model, 'tree_'):
            raise ValueError("The model has not been trained yet!")
