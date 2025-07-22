import unittest

import tensorflow as tf

from estimators.models.multinomial_estimator import MultinomialEstimator
from estimators.utils import create_input_signature


class TestMultinomialEstimator(unittest.TestCase):
    """Unit and regression tests for the MultinomialEstimator."""

    def setUp(self):
        """Set up a MultinomialEstimator instance for testing."""
        self.config = {
            "input_variables": ["x1", "x2"],
            "coefficients": {
                "Intercept": [0.5, 1.5],  # Two intercepts for two logits
                "x1": 2.0,
                "x2": -1.0,
            },
        }
        self.input_signature = create_input_signature(["x1", "x2"])
        self.estimator = MultinomialEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """Test `predict` with valid batch input."""
        input_data = {"x1": tf.ones((3, 1)), "x2": tf.ones((3, 1))}
        prediction = self.estimator.predict(input_data)
        self.assertEqual(prediction.shape, (3, 2))  # Expect 2 logits
        self.assertEqual(prediction.dtype, tf.float32)

    def test_predict_regression(self):
        """Test that the estimator correctly calculates the two logits."""
        input_data = {
            "x1": tf.constant([[1.0], [2.0], [-1.0]], dtype=tf.float32),
            "x2": tf.constant([[0.5], [1.0], [0.0]], dtype=tf.float32),
        }

        # --- Calculate the expected output using Python/TF logic ---
        intercepts = self.config["coefficients"]["Intercept"]
        w1 = self.config["coefficients"]["x1"]
        w2 = self.config["coefficients"]["x2"]

        x1 = input_data["x1"]
        x2 = input_data["x2"]

        base_logit = (x1 * w1) + (x2 * w2)

        expected_logit1 = base_logit + intercepts[0]
        expected_logit2 = base_logit + intercepts[1]

        expected_output = tf.concat([expected_logit1, expected_logit2], axis=1)
        # Expected base_logit: [1.5, 3.0, -2.0]
        # Expected logit1: [2.0, 3.5, -1.5]
        # Expected logit2: [3.0, 4.5, -0.5]

        prediction_tensor = self.estimator.predict(input_data)

        tf.debugging.assert_near(
            prediction_tensor,
            expected_output,
            rtol=1e-6,
            message="Multinomial estimator did not calculate logits correctly.",
        )


if __name__ == "__main__":
    unittest.main()
