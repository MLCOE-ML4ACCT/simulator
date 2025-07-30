import unittest

import tensorflow as tf

from estimators.models.hs_estimator import HSEstimator
from estimators.utils import create_input_signature


class TestHSEstimator(unittest.TestCase):
    """Unit and regression tests for the HSEstimator."""

    def setUp(self):
        """Set up an HSEstimator instance for testing."""
        self.config = {
            "input_variables": ["x1", "x2"],
            "coefficients": {"Intercept": 10.0, "x1": 2.0, "x2": -3.0},
        }
        self.input_signature = create_input_signature(["x1", "x2"])
        self.estimator = HSEstimator(
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
        """Test that the estimator correctly calculates the linear prediction."""
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

        expected_prediction = (x1 * w1) + (x2 * w2) + intercept
        # Expected:
        # Sample 1: (1.0 * 2.0) + (0.5 * -3.0) + 10.0 = 10.5
        # Sample 2: (2.0 * 2.0) + (1.0 * -3.0) + 10.0 = 11.0
        # Sample 3: (-1.0 * 2.0) + (0.0 * -3.0) + 10.0 = 8.0

        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(
            prediction_tensor,
            expected_prediction,
            rtol=1e-6,
            message="HS estimator did not calculate prediction correctly.",
        )


if __name__ == "__main__":
    unittest.main()
