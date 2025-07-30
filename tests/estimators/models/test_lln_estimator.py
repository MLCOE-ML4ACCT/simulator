import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from estimators.models.lln_estimator import LLNEstimator
from estimators.utils import create_input_signature


class TestLLNEstimator(unittest.TestCase):
    """Unit and regression tests for the LLNEstimator."""

    def setUp(self):
        """Set up an LLNEstimator instance for testing."""
        # Use non-zero coefficients for a more meaningful test setup.
        self.config = {
            "steps": [
                {  # Probability model
                    "input_variables": ["p1"],
                    "coefficients": {"Intercept": -0.5, "p1": 1.0},
                },
                {  # Level model
                    "input_variables": ["l1"],
                    "coefficients": {"Intercept": 10.0, "l1": 5.0},
                },
            ]
        }
        self.input_signature = create_input_signature(["p1", "l1"])
        self.estimator = LLNEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """Test `predict` with valid batch input."""
        input_data = {"p1": tf.ones((3, 1)), "l1": tf.ones((3, 1))}
        prediction = self.estimator.predict(input_data)
        self.assertEqual(prediction.shape, (3, 1))
        self.assertEqual(prediction.dtype, tf.float32)

    @patch("estimators.models.lln_estimator.tf.random.uniform")
    def test_predict_with_mocked_randomness(self, mock_tf_uniform):
        """Test `predict` with mocked randomness and sub-models."""
        # Mock the internal models to return deterministic values
        mock_levels = tf.constant([[10.0], [20.0], [30.0]], dtype=tf.float32)
        self.estimator.level_model.predict = MagicMock(return_value=mock_levels)

        # Mock logit to get P_hat = sigmoid(0) = 0.5
        mock_eta = tf.constant([[0.0], [0.0], [0.0]], dtype=tf.float32)
        self.estimator.probability_model.predict = MagicMock(return_value=mock_eta)

        # Mock the random draw
        mock_u = tf.constant([[0.4], [0.6], [0.45]], dtype=tf.float32)
        mock_tf_uniform.return_value = mock_u

        # --- Calculate the expected output using the estimator's logic ---
        p_hat = tf.math.sigmoid(mock_eta)
        should_report_level = tf.cast(p_hat > mock_u, dtype=tf.float32)
        expected_output = should_report_level * mock_levels
        # Expected: P_hat=0.5
        # Sample 1: 0.5 > 0.4 -> 1. Output = 1 * 10.0 = 10.0
        # Sample 2: 0.5 > 0.6 -> 0. Output = 0 * 20.0 = 0.0
        # Sample 3: 0.5 > 0.45 -> 1. Output = 1 * 30.0 = 30.0

        input_data = {"p1": tf.zeros((3, 1)), "l1": tf.zeros((3, 1))}
        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(prediction_tensor, expected_output, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
