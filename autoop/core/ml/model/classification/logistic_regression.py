from sklearn.linear_model import LogisticRegression

from pydantic import PrivateAttr, Field, model_validator
from typing import Literal, Dict, Any
import numpy as np
from autoop.core.ml.model import Model

class LogisticRegressionModel(Model):
    """
    Wrapper around the Logistic Regression model from scikit-learn.
    """

    # Store the underlying scikit-learn model
    _model: LogisticRegression = PrivateAttr(default_factory=LogisticRegression)

    penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = Field(default='l2', description="Regularization type.")
    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength, must be > 0.")
    max_iter: int = Field(default=100, ge=1, description="Maximum number of iterations, must be >= 1.")
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = Field(default='lbfgs', description="Optimization solver.")

    # Validator to check if 'penalty' is compatible with 'solver'
    @model_validator(mode="after")
    def check_penalty_and_solver(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the 'penalty' is compatible with the chosen 'solver'.
        
        Args:
            values (Dict[str, Any]): Dictionary containing the validated values of the fields.
        
        Returns:
            Dict[str, Any]: The dictionary of validated values.
        
        Raises:
            ValueError: If the 'penalty' is 'l1' and the 'solver' is not 'liblinear' or 'saga'.
        """
        penalty = values.get('penalty')
        solver = values.get('solver')

        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            raise ValueError(f"penalty '{penalty}' is only supported by solvers 'liblinear' and 'saga'.")

        return values

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Logistic Regression model to the data.
        :param observations: Input data (features).
        :param ground_truth: Target values.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize the LogisticRegression model with the hyperparameters
        self._model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver
        )

        # Fit the model to the data
        self._model.fit(observations, ground_truth)

        # Store the learned parameters
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict values using the fitted Logistic Regression model.
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
            ValueError: If model parameters 'coefficients' or 'intercept' are missing.
        """
        if "coefficients" not in self._parameters or "intercept" not in self._parameters:
            raise ValueError("The model has not been fitted!")