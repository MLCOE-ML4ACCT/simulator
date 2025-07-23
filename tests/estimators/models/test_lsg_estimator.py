import unittest
from unittest.mock import MagicMock, patch

import tensorflow as tf

from estimators.models.lsg_estimator import LSGEstimator
from estimators.utils import create_input_signature


class TestLSGEstimator(unittest.TestCase):
    """Unit and regression tests for the LSGEstimator."""

    def setUp(self):
        """Set up an LSGEstimator instance for testing."""
        # This config mimics the 4-step structure of dofa_config.py
        self.config = {
            "steps": [
                {"input_variables": ["p_pos"]},  # 0: pos_prob_config
                {"input_variables": ["p_neg"]},  # 1: neg_prob_config
                {"input_variables": ["l_pos"]},  # 2: pos_level_config
                {"input_variables": ["l_neg"]},  # 3: neg_level_config
            ]
        }
        # Add dummy coefficients to satisfy the sub-model initializers
        for step in self.config["steps"]:
            step["coefficients"] = {"Intercept": 0, step["input_variables"][0]: 0}

        self.input_signature = create_input_signature(
            ["p_pos", "p_neg", "l_pos", "l_neg"]
        )
        self.estimator = LSGEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """Test `predict` with valid batch input."""
        input_data = {
            "p_pos": tf.ones((3, 1)),
            "p_neg": tf.ones((3, 1)),
            "l_pos": tf.ones((3, 1)),
            "l_neg": tf.ones((3, 1)),
        }
        prediction = self.estimator.predict(input_data)
        self.assertEqual(prediction.shape, (3, 1))
        self.assertEqual(prediction.dtype, tf.float32)

    @patch("estimators.models.lsg_estimator.tf.random.uniform")
    def test_predict_with_mocked_randomness(self, mock_tf_uniform):
        """Test `predict` with mocked randomness and all sub-models."""
        # Mock the internal models to return deterministic values
        mock_pos_levels = tf.constant([[10.0], [20.0], [30.0]], dtype=tf.float32)
        mock_neg_levels = tf.constant([[-5.0], [-15.0], [-25.0]], dtype=tf.float32)
        self.estimator.positive_level_model.predict = MagicMock(
            return_value=mock_pos_levels
        )
        self.estimator.negative_level_model.predict = MagicMock(
            return_value=mock_neg_levels
        )

        # Mock logits to get specific probabilities
        # P_pos ~= 0.5, P_neg ~= 0.3
        mock_eta_pos = tf.constant([[-0.3665], [-0.3665], [-0.3665]], dtype=tf.float32)
        mock_eta_neg = tf.constant([[-1.1], [-1.1], [-1.1]], dtype=tf.float32)
        self.estimator.pos_probability_model.predict = MagicMock(
            return_value=mock_eta_pos
        )
        self.estimator.neg_probability_model.predict = MagicMock(
            return_value=mock_eta_neg
        )

        # Mock the random draw to select each state
        mock_u = tf.constant([[0.2], [0.7], [0.95]], dtype=tf.float32)
        mock_tf_uniform.return_value = mock_u

        # --- Calculate the expected output using the estimator's logic ---
        p_hat1 = 1.0 - tf.math.exp(-tf.math.exp(mock_eta_pos))  # Prob Negative
        p_hat2 = 1.0 - tf.math.exp(-tf.math.exp(mock_eta_neg))  # Prob Positive

        # The LSG estimator has a unique scaling step
        total_prob = p_hat1 + p_hat2

        is_positive = tf.cast(mock_u < p_hat1, dtype=tf.float32)
        is_negative = tf.cast(
            (mock_u >= p_hat1),
            dtype=tf.float32,
        )

        expected_output = (is_positive * mock_pos_levels) + (
            is_negative * mock_neg_levels
        )

        input_data = {k: tf.zeros((3, 1)) for k in ["p_pos", "p_neg", "l_pos", "l_neg"]}
        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(prediction_tensor, expected_output, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
