from sklearn.datasets import fetch_openml
import unittest
import pandas as pd
import numpy as np

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.linear_regression import LinearRegressionModel
from autoop.core.ml.metric import MeanSquaredError

class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=LinearRegressionModel(),
            input_features=list(filter(lambda x: x.name != "age", self.features)),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(self.pipeline._train_X[0].shape[0], int(0.8 * self.ds_size))
        self.assertEqual(self.pipeline._test_X[0].shape[0], self.ds_size - int(0.8 * self.ds_size))

    def test_train(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()

        # Ensure that the test data is in the correct format (e.g., a single NumPy array)
        X_test = self.pipeline._compact_vectors(self.pipeline._test_X)
        y_test = self.pipeline._test_y

        # Pass the test set for evaluation
        metrics_results = self.pipeline._evaluate(X_test, y_test)

        # Check that metrics_results is not None and has at least one result
        self.assertIsNotNone(metrics_results)
        self.assertGreater(len(metrics_results), 0)

        # Ensure that the results contain valid metric results
        for metric, result in metrics_results:
            self.assertIsInstance(result, float)  # Assuming metric results are floats