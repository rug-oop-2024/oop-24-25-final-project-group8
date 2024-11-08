from sklearn.tree import DecisionTreeClassifier
from pydantic import PrivateAttr, Field
from typing import Literal
import numpy as np
from autoop.core.ml.model import Model

class DecisionTreeClassifierModel(Model):
    """
    Wrapper around scikit-learn's DecisionTreeClassifier, integrating it with the Model interface.
    Provides methods for fitting the model and making predictions, while storing model-specific 
    hyperparameters and fitted parameters.
    """

    _model: DecisionTreeClassifier = PrivateAttr(default_factory=DecisionTreeClassifier)

    criterion: Literal['gini', 'entropy'] = Field(
        default='gini', description="The function to measure the quality of a split ('gini', 'entropy')."
    )
    splitter: Literal['best', 'random'] = Field(
        default='best', description="The strategy used to split at each node ('best', 'random')."
    )

    def __init__(self, **data) -> None:
        """
        Initialize the DecisionTreeClassifierModel with specified hyperparameters.
        
        Args:
            data: Keyword arguments representing model hyperparameters.
                  Expected keys include 'criterion' and 'splitter'.
        """
        super().__init__(**data)
        # Initialize _hyperparameters with fields defined in this class
        self._hyperparameters = {
            field_name: getattr(self, field_name)
            for field_name in self.__fields__.keys()
            if field_name in self.__annotations__
        }
        self.name = "decision tree classifier"
        self.type = "classification"
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Decision Tree Classifier model to the provided data.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features).
            ground_truth (np.ndarray): Target values with shape (n_samples,).
        
        Raises:
            ValueError: If there is a mismatch in dimensions between observations and ground truth.
        """
        super()._validate_input(observations, ground_truth)

        # Initialize and fit the DecisionTreeClassifier with specified hyperparameters
        self._model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
        )
        self._model.fit(observations, ground_truth)

        # Store the fitted model's tree attribute in _parameters for future validation
        self._parameters["tree_"] = self._model.tree_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Decision Tree Classifier model.
        
        Args:
            observations (np.ndarray): Input data (features) with shape (n_samples, n_features) 
                                       for which predictions are to be made.
        
        Returns:
            np.ndarray: Predicted class labels for the provided observations.
        
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
        Check if the model has been fitted by verifying the presence of a decision tree.
        
        Raises:
            ValueError: If the model has not been trained, indicated by the absence of a tree.
        """
        if not hasattr(self._model, 'tree_'):
            raise ValueError("The model has not been fitted!")
