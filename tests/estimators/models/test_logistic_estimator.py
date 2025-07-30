import unittest

import tensorflow as tf

from estimators.models.logistic_estimator import LogisticEstimator
from estimators.utils import create_input_signature


class TestLogisticEstimator(unittest.TestCase):
    """Unit and regression tests for the LogisticEstimator."""

    def setUp(self):
        """Set up a LogisticEstimator instance for testing."""
        self.config = {
            "input_variables": ["x1", "x2"],
            "coefficients": {"Intercept": 0.5, "x1": 1.5, "x2": -1.0},
        }
        self.input_signature = create_input_signature(["x1", "x2"])
        self.estimator = LogisticEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """Test `predict` with valid batch input."""
        input_data = {"x1": tf.ones((3, 1)), "x2": tf.ones((3, 1))}
        prediction = self.estimator.predict(input_data)
        self.assertEqual(prediction.shape, (3, 1))
        self.assertEqual(prediction.dtype, tf.float32)

    def test_predict_regression(self):
        """Test that the estimator correctly calculates the logit."""
        input_data = {
            "x1": tf.constant([[1.0], [2.0], [-1.0]], dtype=tf.float32),
            "x2": tf.constant([[0.5], [1.0], [0.0]], dtype=tf.float32),
        }

        # --- Calculate the expected output using Python/TF logic ---
        intercept = self.config["coefficients"]["Intercept"]
        w1 = self.config["coefficients"]["x1"]
        w2 = self.config["coefficients"]["x2"]

        x1 = input_data["x1"]
        x2 = input_data["x2"]

        expected_logit = (x1 * w1) + (x2 * w2) + intercept
        # Expected:
        # Sample 1: (1.0 * 1.5) + (0.5 * -1.0) + 0.5 = 1.5
        # Sample 2: (2.0 * 1.5) + (1.0 * -1.0) + 0.5 = 2.5
        # Sample 3: (-1.0 * 1.5) + (0.0 * -1.0) + 0.5 = -1.0

        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(
            prediction_tensor,
            expected_logit,
            rtol=1e-6,
            message="Logistic estimator did not calculate logit correctly.",
        )


if __name__ == "__main__":
    unittest.main()
