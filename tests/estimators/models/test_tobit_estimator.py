import unittest
from unittest.mock import patch

import tensorflow as tf

from estimators.models.tobit_estimator import TobitEstimator
from estimators.utils import create_input_signature


class TestTobitEstimator(unittest.TestCase):
    """Unit tests for the TobitEstimator."""

    def setUp(self):
        """Set up a TobitEstimator instance for testing."""
        self.config = {
            "scale": 1.0,
            "steps": [
                {
                    "input_variables": ["x1", "x2"],
                    "coefficients": {
                        "Intercept": 0.5,
                        "x1": 1.5,
                        "x2": -1.0,
                    },
                }
            ],
        }

        # Use the utility to create the input signature
        self.input_signature = create_input_signature(
            self.config["steps"][0]["input_variables"]
        )

        self.estimator = TobitEstimator(
            config=self.config,
            input_signature=self.input_signature,
        )

    def test_predict_valid_input(self):
        """
        Test Case 1.1.1: Test `predict` with valid input.
        Verify that the predict method returns a result with the expected
        shape and data type when given a valid input tensor.
        """
        # Create a sample input packet
        input_data = {
            "x1": tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
            "x2": tf.constant([[0.5], [1.0], [1.5]], dtype=tf.float32),
        }
        num_samples = tf.shape(input_data["x1"])[0]

        # Execute the prediction
        prediction_tensor = self.estimator.predict(input_data)

        # Assertions
        self.assertIsInstance(
            prediction_tensor, tf.Tensor, "Prediction should be a TensorFlow Tensor."
        )
        self.assertEqual(
            prediction_tensor.shape[0],
            num_samples,
            "Prediction should have the same number of rows as input.",
        )
        self.assertEqual(
            prediction_tensor.dtype, tf.float32, "Prediction dtype should be float32."
        )

    def test_predict_non_negative_output(self):
        """
        Test Case 1.1.2: Test `predict` with inputs that could lead to negative results.
        Verify that the Tobit model correctly censors the output at 0.
        """
        # Create input data where the deterministic part (X'B) is negative
        input_data = {
            "x1": tf.constant(
                [[-1.0], [-2.0]], dtype=tf.float32
            ),  # Large negative values
            "x2": tf.constant([[1.0], [2.0]], dtype=tf.float32),
        }

        # Execute the prediction
        prediction_tensor = self.estimator.predict(input_data)

        # Assertions
        self.assertTrue(
            tf.reduce_all(prediction_tensor >= 0),
            "All predictions should be non-negative.",
        )

    def test_predict_with_missing_vars_raises_error(self):
        """
        Test Case 1.1.3: Test `predict` with missing variables.
        Verify that a KeyError is raised if a required variable is missing.
        """
        # Input data is missing 'x2'
        input_data = {
            "x1": tf.constant([[1.0]], dtype=tf.float32),
        }

        # Expect a KeyError because the logic inside _predict_logic will fail
        # to find the key 'x2' in the input packet.
        with self.assertRaises(TypeError):
            self.estimator.predict(input_data)

    @patch("estimators.models.tobit_estimator.tf.random.uniform")
    def test_predict_with_mocked_randomness(self, mock_tf_uniform):
        """
        Test Case 2.1: TobitEstimator prediction with mocked randomness.
        Ensures the prediction logic is correct by replacing the random
        error generation with a fixed, deterministic value. This test
        uses a batch of 3 samples.
        """
        # Configure the mock to return 0.5 for each sample in the batch.
        # This removes the stochastic element for a predictable test.
        mock_return_value = tf.constant([[0.5], [0.5], [0.5]], dtype=tf.float32)
        mock_tf_uniform.return_value = mock_return_value

        # Define a fixed input packet with 3 samples
        input_data = {
            "x1": tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
            "x2": tf.constant([[0.5], [1.0], [1.5]], dtype=tf.float32),
        }

        # Expected output is now fully deterministic and easy to verify.
        # With the error term being zero, prediction is just the deterministic
        # part (X'B), censored at zero.
        # Sample 1: 0.5 + (1.0 * 1.5) + (0.5 * -1.0) = 1.5
        # Sample 2: 0.5 + (2.0 * 1.5) + (1.0 * -1.0) = 2.5
        # Sample 3: 0.5 + (3.0 * 1.5) + (1.5 * -1.0) = 3.5
        expected_output = tf.constant([[1.5], [2.5], [3.5]], dtype=tf.float32)

        # Execute the prediction
        prediction_tensor = self.estimator.predict(input_data)

        # Assert that the prediction exactly matches the expected output
        tf.debugging.assert_near(
            prediction_tensor,
            expected_output,
            rtol=1e-6,
            message="Prediction with mocked randomness is incorrect for batch input.",
        )


if __name__ == "__main__":
    unittest.main()
