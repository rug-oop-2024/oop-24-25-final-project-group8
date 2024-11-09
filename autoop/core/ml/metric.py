from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "root_mean_squared_error",
    "mean_absolute_error",
]


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
    metrics = {
        "mean_squared_error": MeanSquaredError,
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
        "f1_score": F1Score,
        "root_mean_squared_error": RootMeanSquaredError,
        "mean_absolute_error": MeanAbsoluteError,
    }

    if name not in metrics:
        raise ValueError(
            f"Metric '{name}' is not implemented. Available metrics: {METRICS}"
        )

    return metrics[name]()


class Metric(ABC):
    """
    Base class for all metrics. Metrics take ground truth and predictions as input and
    return a real number.
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
    """Mean Squared Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    """Accuracy metric for multi-class classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        return np.mean(y_true == y_pred)


class Precision(Metric):
    """Precision metric for multi-class classification using macro-averaging."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            precisions.append(precision)

        return np.mean(precisions)


class Recall(Metric):
    """Recall metric for multi-class classification using macro-averaging."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []

        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            recalls.append(recall)

        return np.mean(recalls)


class F1Score(Metric):
    """F1 Score metric for multi-class classification using macro-averaging."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision_metric = Precision()
        recall_metric = Recall()
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class RootMeanSquaredError(Metric):
    """Root Mean Squared Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
