from sklearn.ensemble import RandomForestRegressor
from pydantic import PrivateAttr, Field
from typing import Literal
import numpy as np
from autoop.core.ml.model import Model

class RandomForestRegressorModel(Model):
    """
    Wrapper around the RandomForestRegressor from scikit-learn.
    Provides methods to fit the model to data and make predictions, while managing 
    model hyperparameters and fitted parameters within the Model framework.
    """

    _model: RandomForestRegressor = PrivateAttr()

    n_estimators: int = Field(default=100, ge=1, description="The number of trees in the forest, must be >= 1.")
    criterion: Literal['absolute_error', 'poisson', 'friedman_mse', 'squared_error'] = Field(default='friedman_mse', description="The function to measure the quality of a split ('mse', 'mae').")

    def __init__(self, **data) -> None:
        """
        Initialize the RandomForestRegressorModel with specified hyperparameters.
        
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
        self.name = "random forest regressor"
        self.type = "regression"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Random Forest Regressor model to the provided data.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).
        
        Raises:
            ValueError: If there is a mismatch in dimensions between observations and ground truth.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize and fit the RandomForestRegressor with specified hyperparameters
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
        )
        self._model.fit(observations, ground_truth)

        # Store the fitted model's estimators in _parameters for future validation
        self._parameters["estimators_"] = self._model.estimators_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Use the trained model to make predictions on new observations.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features) 
                                       for which predictions are to be made.
        
        Returns:
            np.ndarray: Predicted values, with shape (n_samples,).
        
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
        Check if the model has been fitted by verifying the presence of estimators.
        
        Raises:
            ValueError: If the model has not been trained, indicated by missing estimators.
        """
        if not hasattr(self._model, 'estimators_'):
            raise ValueError("The model has not been trained yet!")