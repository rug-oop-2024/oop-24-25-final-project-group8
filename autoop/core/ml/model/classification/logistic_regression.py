from sklearn.linear_model import LogisticRegression
from pydantic import PrivateAttr, Field, model_validator
from typing import Literal, Dict, Any
import numpy as np
from autoop.core.ml.model import Model


class LogisticRegressionModel(Model):
    """
    Wrapper around scikit-learn's LogisticRegression, integrating it with the Model
    interface.
    Provides methods for fitting the model and making predictions, while storing
    model-specific hyperparameters and fitted parameters.
    """

    _model: LogisticRegression = PrivateAttr(default_factory=LogisticRegression)

    penalty: Literal["l1", "l2", "elasticnet", "none"] = Field(
        default="l2", description="Regularization type."
    )
    C: float = Field(
        default=1.0, gt=0, description="Inverse regularization strength; must be > 0."
    )
    max_iter: int = Field(
        default=100, ge=1, description="Maximum number of iterations; must be >= 1."
    )
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = Field(
        default="lbfgs", description="Optimization solver."
    )

    @model_validator(mode="after")
    def check_penalty_and_solver(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the 'penalty' is compatible with the chosen 'solver'.

        Args:
            values (Dict[str, Any]): Dictionary containing the validated values of the
            fields.

        Returns:
            Dict[str, Any]: The dictionary of validated values.

        Raises:
            ValueError: If the 'penalty' is 'l1' and the 'solver' is not 'liblinear' or
            'saga'.
        """
        penalty = getattr(values, "penalty", None)
        solver = getattr(values, "solver", "lbfgs")

        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            raise ValueError(
                f"penalty '{penalty}' is only supported by solvers 'liblinear' and"
                f"'saga'."
            )

        return values

    def __init__(self, **data) -> None:
        """
        Initialize the LogisticRegressionModel with specified hyperparameters.

        Args:
            data: Keyword arguments representing model hyperparameters.
                  Expected keys include 'penalty', 'C', 'max_iter', and 'solver'.
        """
        super().__init__(**data)
        # Initialize _hyperparameters with fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "logistic regression"
        self.type = "classification"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Logistic Regression model to the provided data.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).

        Raises:
            ValueError: If there is a mismatch in dimensions between observations and
            ground truth.
        """
        super()._validate_input(observations, ground_truth)

        # Convert one-hot encoded ground_truth to class labels if necessary
        if ground_truth.ndim > 1 and ground_truth.shape[1] > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        # Initialize the LogisticRegression model with the specified hyperparameters
        self._model = LogisticRegression(
            penalty=self.penalty, C=self.C, max_iter=self.max_iter, solver=self.solver
        )

        # Fit the model to the data
        self._model.fit(observations, ground_truth)

        # Store the learned parameters
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the fitted Logistic Regression model.

        Args:
            observations (np.ndarray): Input data (features) with shape
            (n_samples, n_features) for which predictions are to be made.

        Returns:
            np.ndarray: Predicted class labels for the provided observations.

        Raises:
            ValueError: If the model has not been fitted or if the input features do
            not match the expected dimensions from training.
        """
        self._validate_fit()
        super()._validate_num_features(observations)

        return self._model.predict(observations)

    def _validate_fit(self) -> None:
        """
        Check if the model has been fitted by verifying the presence of learned
        parameters.

        Raises:
            ValueError: If model parameters 'coefficients' or 'intercept' are missing.
        """
        if (
            "coefficients" not in self._parameters
            or "intercept" not in self._parameters
        ):
            raise ValueError("The model has not been fitted!")
