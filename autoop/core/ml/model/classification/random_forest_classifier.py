from sklearn.ensemble import RandomForestClassifier

from pydantic import PrivateAttr, Field
from typing import Literal, Optional

import numpy as np
from autoop.core.ml.model import Model

class RandomForestClassifierModel(Model):
    """
    Wrapper around the RandomForestClassifier from scikit-learn.
    """

    # Store the underlying scikit-learn model
    _model: RandomForestClassifier = PrivateAttr(default_factory=RandomForestClassifier)

    n_estimators: int = Field(default=100, ge=1, description="The number of trees in the forest, must be >= 1.")
    criterion: Literal['gini', 'entropy'] = Field(default='gini', description="The function to measure the quality of a split ('gini', 'entropy').")

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize _hyperparameters with only fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "random forest classifier"
        self.type = "classification"
        self._parameters ={}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Random Forest model to the provided data.
        :param observations: Input data (features).
        :param ground_truth: Target values.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize RandomForestClassifier with hyperparameters
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
        )

        self._model.fit(observations, ground_truth)

        self._parameters["estimators_"] = self._model.estimators_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Use the model to make predictions on new observations.
        :param observations: Input data to predict on.
        :return: Predicted values.
        """
        self._validate_fit()
        super()._validate_num_features(observations)

        predictions = self._model.predict(observations)
        return predictions

    def _validate_fit(self) -> None:
        """
        Validate that the model has been fitted.

        Raises:
            ValueError: If model estimators are missing.
        """
        if not hasattr(self._model, 'estimators_'):
            raise ValueError("The model has not been fitted!")