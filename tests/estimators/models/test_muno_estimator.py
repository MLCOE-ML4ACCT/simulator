import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from estimators.models.muno_estimator import MUNOEstimator
from estimators.utils import create_input_signature


class TestMUNOEstimator(unittest.TestCase):
    """Unit and regression tests for the MUNOEstimator."""

    def setUp(self):
        """Set up a MUNOEstimator instance for testing."""
        # This config now correctly mirrors the structure of a real MUNO config
        # like 'sma_config.py', especially the 'coefficients' for the
        # multinomial probability model.
        self.config = {
            "steps": [
                {  # Step 1: Probability model (Multinomial)
                    "input_variables": ["p1"],
                    "coefficients": {
                        "Intercept": [-1.5, 0.8],  # Two intercepts for two logits
                        "p1": 0.0,  # Single coefficient per variable
                    },
                },
                {  # Step 2: Positive level model (HS)
                    "input_variables": ["l_pos"],
                    "coefficients": {"Intercept": 100.0, "l_pos": 10.0},
                },
                {  # Step 3: Negative level model (HS)
                    "input_variables": ["l_neg"],
                    "coefficients": {"Intercept": -100.0, "l_neg": -10.0},
                },
            ]
        }
        self.input_signature = create_input_signature(["p1", "l_pos", "l_neg"])
        self.estimator = MUNOEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """Test `predict` with valid batch input."""
        input_data = {
            "p1": tf.ones((3, 1)),
            "l_pos": tf.ones((3, 1)),
            "l_neg": tf.ones((3, 1)),
        }
        prediction = self.estimator.predict(input_data)
        self.assertEqual(prediction.shape, (3, 1))
        self.assertEqual(prediction.dtype, tf.float32)

    @patch("estimators.models.muno_estimator.tf.random.uniform")
    def test_predict_with_mocked_randomness(self, mock_tf_uniform):
        """Test `predict` with mocked randomness and sub-models."""
        # Mock the internal models to return deterministic values
        mock_pos_levels = tf.constant([[10.0], [20.0], [30.0]], dtype=tf.float32)
        mock_neg_levels = tf.constant([[-5.0], [-15.0], [-25.0]], dtype=tf.float32)
        self.estimator.positive_level_model.predict = MagicMock(
            return_value=mock_pos_levels
        )
        self.estimator.negative_level_model.predict = MagicMock(
            return_value=mock_neg_levels
        )

        # Mock logits for P(state<=neg) and P(state<=zero)
        # These correspond to P_hat1 ~= 0.2 and P_hat2 ~= 0.7
        mock_logits = tf.constant(
            [[-1.5, 0.8], [-1.5, 0.8], [-1.5, 0.8]], dtype=tf.float32
        )
        self.estimator.probability_model.predict = MagicMock(return_value=mock_logits)

        # Mock the random draw to select each state once
        mock_u = tf.constant([[0.1], [0.5], [0.9]], dtype=tf.float32)
        mock_tf_uniform.return_value = mock_u

        # --- Calculate the expected output using Python/TF logic ---
        eta1 = mock_logits[:, 0:1]
        eta2 = mock_logits[:, 1:2]
        eta2 = tf.maximum(eta1, eta2)
        p_hat1 = 1.0 - tf.math.exp(-tf.math.exp(eta1))
        p_hat2 = 1.0 - tf.math.exp(-tf.math.exp(eta2))

        is_negative = tf.cast(mock_u < p_hat1, dtype=tf.float32)
        is_zero = tf.cast((mock_u >= p_hat1) & (mock_u < p_hat2), dtype=tf.float32)
        is_positive = tf.cast(mock_u >= p_hat2, dtype=tf.float32)

        expected_output = (is_positive * mock_pos_levels) + (
            is_negative * mock_neg_levels
        )
        # Expected:
        # P_hat1 ~= 0.2, P_hat2 ~= 0.7
        # Sample 1: U=0.1 < 0.2 -> Negative state -> -5.0
        # Sample 2: U=0.5 (between 0.2, 0.7) -> Zero state -> 0.0
        # Sample 3: U=0.9 > 0.7 -> Positive state -> 30.0

        # Dummy input (sub-models are mocked)
        input_data = {
            "p1": tf.zeros((3, 1)),
            "l_pos": tf.zeros((3, 1)),
            "l_neg": tf.zeros((3, 1)),
        }

        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(
            prediction_tensor,
            expected_output,
            rtol=1e-6,
            message="Prediction with mocked randomness is incorrect.",
        )


if __name__ == "__main__":
    unittest.main()
