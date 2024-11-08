from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pydantic import PrivateAttr, Field
from typing import Optional
import numpy as np
from autoop.core.ml.model import Model

class LinearRegressionModel(Model):
    """
    Wrapper around the LinearRegression from scikit-learn, integrating it with the Model interface.
    Provides methods for fitting the model and making predictions, while storing model-specific 
    hyperparameters and fitted parameters.
    """

    _model: LinearRegression = PrivateAttr()

    fit_intercept: bool = Field(default=True, description="Whether to calculate the intercept for the model.")
    normalize: Optional[bool] = Field(default=False, description="Whether to normalize the input features.")
    copy_X: bool = Field(default=True, description="Whether to copy X (input data) or overwrite it.")

    def __init__(self, **data) -> None:
        """
        Initialize the LinearRegressionModel with specified hyperparameters.
        
        Args:
            data: Keyword arguments representing model hyperparameters.
                  Expected keys include 'fit_intercept', 'normalize', and 'copy_X'.
        """
        super().__init__(**data)
        # Initialize _hyperparameters with fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "linear regression"
        self.type = "regression"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the provided data.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).
        
        Raises:
            ValueError: If there is a mismatch in dimensions between observations and ground truth.
        """
        self._parameters = {}
        super()._validate_input(observations, ground_truth)

        # Create the LinearRegression model, using normalization if specified
        if self.normalize:
            self._model = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X))
        else:
            self._model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=self.copy_X)

        # Fit the model and store parameters
        self._model.fit(observations, ground_truth)

        # Extract and store the fitted coefficients and intercept
        self._parameters["coef_"] = self._model.coef_
        self._parameters["intercept_"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Linear Regression model.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features) 
                                       for which predictions are to be made.
        
        Returns:
            np.ndarray: Predicted values for the provided observations.
        
        Raises:
            ValueError: If the model has not been fitted or if the input features do not match 
                        the expected dimensions from training.
        """
        # Validate that the model is fitted and that input dimensions match training dimensions
        self._validate_fit()
        super()._validate_num_features(observations)

        return self._model.predict(observations)

    def _validate_fit(self) -> None:
        """
        Check if the model has been fitted by verifying the presence of coefficients.
        
        Raises:
            ValueError: If the model has not been trained, indicated by missing coefficients.
        """
        if not hasattr(self._model, 'coef_'):
            raise ValueError("The model has not been trained yet!")
