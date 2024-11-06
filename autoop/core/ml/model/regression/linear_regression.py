from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pydantic import PrivateAttr, Field
from typing import Optional

import numpy as np
from autoop.core.ml.model import Model

class LinearRegressionModel(Model):
    """
    Wrapper around the LinearRegression from scikit-learn.
    """
    
    _model: LinearRegression = PrivateAttr()

    fit_intercept: bool = Field(default=True, description="Whether to calculate the intercept for the model.")
    normalize: Optional[bool] = Field(default=False, description="Whether to normalize the input features.")
    copy_X: bool = Field(default=True, description="Whether to copy X (input data) or overwrite it.")

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize _hyperparameters with only fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "linear regression"
        self.type = "regression"
        self._parameters ={}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the provided data.
        :param observations: Input data (features).
        :param ground_truth: Target values.
        """
        self._parameters = {}
        super()._validate_input(observations, ground_truth)

        if self.normalize:
            self._model = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X))
        else:
            self._model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X)

        self._model.fit(observations, ground_truth)

        self._parameters["coef_"] = self._model.coef_
        self._parameters["intercept_"] = self._model.intercept_

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
            ValueError: If model coefficients are missing.
        """
        if not hasattr(self._model, 'coef_'):
            raise ValueError("The model has not been trained yet!")