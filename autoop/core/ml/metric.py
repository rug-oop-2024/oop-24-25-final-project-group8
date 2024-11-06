from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "root_mean_squared_error",
    "mean_absolute_error"
] # add the names (in strings) of the metrics you implement

def get_metric(name: str):
    """
    Factory function to get a metric by name.
    
    Args:
        name (str): The name of the metric to retrieve.
    
    Returns:
        Metric: An instance of a Metric subclass.
    
    Raises:
        ValueError: If the metric name is not recognized.
    """

    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    else:
        raise ValueError(f"Metric '{name}' is not implemented. Available metrics: {METRICS}")

class Metric(ABC):
    """
    Base class for all metrics. Metrics take ground truth and predictions as input and return a real number.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the metric.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Model predictions.
        
        Returns:
            float: The computed metric value.
        """
        pass


class MeanSquaredError(Metric):
    """
    Mean Squared Error metric.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE) between ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Model predictions.

        Returns:
            float: The mean squared error value.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

class Accuracy(Metric):
    """
    Accuracy metric for multi-class classification.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes accuracy as the ratio of correctly predicted instances.

        Args:
            y_true (np.ndarray): Ground truth labels (can be one-hot encoded).
            y_pred (np.ndarray): Model predictions (can be one-hot encoded).

        Returns:
            float: The accuracy value.
        """

        # Calculate accuracy
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions
        
        return accuracy
    
class Precision(Metric):
    """
    Precision metric for multi-class classification using macro-averaging.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes precision for multi-class classification using macro-averaging.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Model predictions.
        
        Returns:
            float: The macro-averaged precision value.
        """

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            precisions.append(precision)

        # Macro-averaged precision
        return np.mean(precisions)

class Recall(Metric):
    """
    Recall metric for multi-class classification using macro-averaging.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes recall for multi-class classification using macro-averaging.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Model predictions.
        
        Returns:
            float: The macro-averaged recall value.
        """

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))
            
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            recalls.append(recall)

        # Macro-averaged recall
        return np.mean(recalls)

class F1Score(Metric):
    """
    F1 Score metric for multi-class classification using macro-averaging.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes F1 Score for multi-class classification using macro-averaging.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Model predictions.
        
        Returns:
            float: The macro-averaged F1 score.
        """
        precision_metric = Precision()
        recall_metric = Recall()
        
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        
        # F1 score is the mean of precision and recall
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error metric for regression.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error (RMSE) between ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Model predictions.

        Returns:
            float: The root mean squared error value.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error metric for regression.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error (MAE) between ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Model predictions.

        Returns:
            float: The mean absolute error value.
        """
        mae = np.mean(np.abs(y_true - y_pred))
        return mae