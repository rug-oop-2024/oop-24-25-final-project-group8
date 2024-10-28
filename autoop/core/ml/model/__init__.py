from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.linear_regression import LinearRegressionModel
from autoop.core.ml.model.regression.decision_tree_regressor import DecisionTreeRegressorModel
from autoop.core.ml.model.regression.random_forest_regressor import RandomForestRegressorModel
from autoop.core.ml.model.classification.logistic_regression import LogisticRegressionModel
from autoop.core.ml.model.classification.decision_tree_classifier import DecisionTreeClassifierModel
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifierModel


REGRESSION_MODELS = [
    'linear_regression',
    'decision_tree_regressor',
    'random_forest_regressor'
]

CLASSIFICATION_MODELS = [
    'logistic_regression',
    'decision_tree_classifier',
    'random_forest_classifier'
]

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.
    
    Args:
        model_name (str): The name of the model to retrieve.
        
    Returns:
        Model: The model class corresponding to the model name.
        
    Raises:
        ValueError: If the model_name is not found in any models list.
    """
    
    if model_name == 'lasso_regression':
        return LinearRegressionModel()
    elif model_name == 'decision_tree_regressor':
        return DecisionTreeRegressorModel()
    elif model_name == 'random_forest_regressor':
        return RandomForestRegressorModel()
    elif model_name == 'logistic_regression':
        return LogisticRegressionModel()
    elif model_name == 'decision_tree_classifier':
        return DecisionTreeClassifierModel()
    elif model_name == 'random_forest_classifier':
        return RandomForestClassifierModel()
    else:
        raise ValueError(f"Model '{model_name}' not found. Available models are: "
                         f"{', '.join(REGRESSION_MODELS + CLASSIFICATION_MODELS)}")
