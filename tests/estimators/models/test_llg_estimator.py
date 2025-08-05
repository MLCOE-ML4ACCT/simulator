import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from estimators.models.llg_estimator import LLGEstimator
from estimators.utils import create_input_signature


class TestLLGEstimator(unittest.TestCase):
    """Unit and regression tests for the LLGEstimator."""

    def setUp(self):
        """Set up a LLGEstimator instance for testing."""
        # A simplified config mimicking a real two-step LLG model.
        self.config = {
            "steps": [
                {  # Step 1: Probability model (Logistic)
                    "input_variables": ["p1", "p2"],
                    "coefficients": {"Intercept": -0.2, "p1": 0.5, "p2": -0.5},
                },
                {  # Step 2: Level model (Huber-Schweppes)
                    "input_variables": ["l1", "l2"],
                    "coefficients": {"Intercept": 100, "l1": 10, "l2": 20},
                },
            ]
        }

        # The overall signature includes all unique inputs from all steps.
        self.input_signature = create_input_signature(["p1", "p2", "l1", "l2"])

        self.estimator = LLGEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """
        Test Case: Test `predict` with valid batch input.
        Verifies the output shape and type are correct.
        """
        # Create a sample input packet with 3 samples
        input_data = {
            "p1": tf.ones((3, 1), dtype=tf.float32),
            "p2": tf.ones((3, 1), dtype=tf.float32),
            "l1": tf.ones((3, 1), dtype=tf.float32),
            "l2": tf.ones((3, 1), dtype=tf.float32),
        }
        num_samples = tf.shape(input_data["p1"])[0]

        # Execute the prediction
        prediction_tensor = self.estimator.predict(input_data)

        # Assertions
        self.assertIsInstance(prediction_tensor, tf.Tensor)
        self.assertEqual(prediction_tensor.shape[0], num_samples)
        self.assertEqual(prediction_tensor.dtype, tf.float32)

    @patch("estimators.models.llg_estimator.tf.random.uniform")
    def test_predict_with_mocked_randomness(self, mock_tf_uniform):
        """
        Test Case: Test `predict` with mocked randomness and sub-models.
        This isolates the LLGEstimator's combination logic.
        """
        # Mock the internal models to return deterministic values
        mock_levels = tf.constant([[10.0], [20.0], [30.0]], dtype=tf.float32)
        self.estimator.level_model.predict = MagicMock(return_value=mock_levels)

        # This logit value corresponds to P_hat ~= 0.5
        # logit = log(-log(1-P_hat)) => log(-log(0.5)) => -0.3665
        mock_logit = tf.constant([[-0.3665], [-0.3665], [-0.3665]], dtype=tf.float32)
        self.estimator.probability_model.predict = MagicMock(return_value=mock_logit)

        # Mock the random uniform draw.
        mock_u = tf.constant([[0.4], [0.6], [0.45]], dtype=tf.float32)
        mock_tf_uniform.return_value = mock_u

        # --- Calculate the expected output using Python/TF logic ---
        # This makes the test self-verifying and removes "magic numbers".
        p_hat = 1.0 - tf.math.exp(-tf.math.exp(mock_logit))
        should_report_level = tf.cast(p_hat > mock_u, dtype=tf.float32)
        expected_output = mock_levels * should_report_level
        # Expected calculation:
        # P_hat for all samples is ~0.5.
        # Sample 1: 0.5 > 0.4 -> should_report = 1. Output = 10.0 * 1 = 10.0
        # Sample 2: 0.5 > 0.6 -> should_report = 0. Output = 20.0 * 0 = 0.0
        # Sample 3: 0.5 > 0.45 -> should_report = 1. Output = 30.0 * 1 = 30.0

        # Create a dummy input packet (values don't matter as sub-models are mocked)
        input_data = {
            "p1": tf.zeros((3, 1)),
            "p2": tf.zeros((3, 1)),
            "l1": tf.zeros((3, 1)),
            "l2": tf.zeros((3, 1)),
        }

        # Execute the prediction
        prediction_tensor = self.estimator.predict(input_data)

        # Assert that the prediction exactly matches the calculated expected output
        tf.debugging.assert_near(
            prediction_tensor,
            expected_output,
            rtol=1e-6,
            message="Prediction with mocked randomness is incorrect.",
        )


if __name__ == "__main__":
    unittest.main()
