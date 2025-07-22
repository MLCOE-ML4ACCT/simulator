import unittest
from unittest.mock import patch

import tensorflow as tf

from estimators.factory import EstimatorFactory
from estimators.models.llg_estimator import LLGEstimator
from estimators.models.lln_estimator import LLNEstimator
from estimators.models.lsg_estimator import LSGEstimator
from estimators.models.muno_estimator import MUNOEstimator
from estimators.models.tobit_estimator import TobitEstimator


class TestEstimatorFactory(unittest.TestCase):
    """Integration tests for the EstimatorFactory."""

    def setUp(self):
        """Set up the factory for testing."""
        # Using the default config directory
        self.factory = EstimatorFactory()

    def test_create_tobit_estimator(self):
        """
        Test Case 3.1: Test `EstimatorFactory` with `TobitEstimator`.
        Verify that the factory can successfully create a TobitEstimator
        and that the created instance can make a prediction.
        """
        # The 'IMA' variable is configured to use the 'TOBIT' method
        # in estimators/configs/ima_config.py
        variable_name = "IMA"

        # 1. Create the estimator using the factory
        estimator = self.factory.get_estimator(variable_name)

        # 2. Assert that the correct type of estimator was created
        self.assertIsInstance(
            estimator,
            TobitEstimator,
            "Factory should create a TobitEstimator for IMA.",
        )

        # 3. Create a valid input packet that EXACTLY matches the estimator's
        #    compiled signature. The user is correct: providing a partial
        #    packet will cause a TypeError when calling the tf.function.
        config = self.factory.configs[variable_name]
        required_inputs = self.factory._get_required_inputs_from_config(config)

        input_data = {
            key: tf.constant([[0.0]], dtype=tf.float32) for key in required_inputs
        }

        # 4. Execute a prediction
        try:
            prediction = estimator.predict(input_data)
            # Check that the prediction runs without errors and returns a tensor
            self.assertIsInstance(prediction, tf.Tensor)
            self.assertEqual(prediction.shape, (1, 1))
        except Exception as e:
            self.fail(f"Prediction failed for factory-created TobitEstimator: {e}")

    def test_input_signature_validation(self):
        """
        Tests that the tf.function input signature correctly validates input.
        """
        # 1. Define a simple, self-contained config for a Tobit model.
        mock_config = {
            "method": "TOBIT",
            "scale": 1.0,
            "steps": [
                {
                    "input_variables": ["sig_test_var1", "sig_test_var2"],
                    "coefficients": {
                        "Intercept": 1.0,
                        "sig_test_var1": 1.0,
                        "sig_test_var2": 1.0,
                    },
                }
            ],
        }

        # 2. Use patch.dict to temporarily add this config to the factory.
        # This makes the test independent of the actual config files.
        with patch.dict(self.factory.configs, {"TEST_SIGNATURE": mock_config}):
            estimator = self.factory.get_estimator("TEST_SIGNATURE")

            # 3. Test Case 1: A correct input packet should succeed.
            correct_input = {
                "sig_test_var1": tf.constant([[1.0]], dtype=tf.float32),
                "sig_test_var2": tf.constant([[1.0]], dtype=tf.float32),
            }
            try:
                estimator.predict(correct_input)
            except Exception as e:
                self.fail(f"Predict failed with correct signature: {e}")

            # 4. Test Case 2: An incorrect packet (missing a key) should fail.
            # tf.function raises a TypeError when the input dict keys do not
            # exactly match the keys in the TensorSpec signature.
            incorrect_input = {"sig_test_var1": tf.constant([[1.0]], dtype=tf.float32)}
            with self.assertRaises(TypeError):
                estimator.predict(incorrect_input)

    def test_create_llg_estimator(self):
        """
        Test Case: Test `EstimatorFactory` with `LLGEstimator`.
        Verify that the factory can successfully create an LLGEstimator
        and that the created instance can make a prediction.
        """

        variable_name = "EDEPMA"

        # 1. Create the estimator using the factory
        estimator = self.factory.get_estimator(variable_name)

        # 2. Assert that the correct type of estimator was created
        # Note: We check for LLGEstimator as LLN is mapped to it.
        self.assertIsInstance(
            estimator,
            LLGEstimator,
            "Factory should create an LLGEstimator for EDEPMA.",
        )

        # 3. Create a valid input packet that matches the estimator's signature
        config = self.factory.configs[variable_name]
        required_inputs = self.factory._get_required_inputs_from_config(config)

        input_data = {
            key: tf.constant([[0.0]], dtype=tf.float32) for key in required_inputs
        }

        # 4. Execute a prediction
        try:
            prediction = estimator.predict(input_data)
            self.assertIsInstance(prediction, tf.Tensor)
            self.assertEqual(prediction.shape, (1, 1))
        except Exception as e:
            self.fail(f"Prediction failed for factory-created LLGEstimator: {e}")

    def test_create_muno_estimator(self):
        """
        Test Case: Test `EstimatorFactory` with `MUNOEstimator`.
        Verify that the factory can successfully create a MUNOEstimator
        using the correct 'SMA' configuration.
        """
        # The 'SMA' variable is configured to use the 'MUNO' method.
        variable_name = "SMA"

        # 1. Create the estimator using the factory
        estimator = self.factory.get_estimator(variable_name)

        # 2. Assert that the correct type of estimator was created
        self.assertIsInstance(
            estimator,
            MUNOEstimator,
            "Factory should create a MUNOEstimator for SMA.",
        )

        # 3. Create a valid input packet that matches the estimator's signature
        config = self.factory.configs[variable_name]
        required_inputs = self.factory._get_required_inputs_from_config(config)

        input_data = {
            key: tf.constant([[0.0]], dtype=tf.float32) for key in required_inputs
        }

        # 4. Execute a prediction
        try:
            prediction = estimator.predict(input_data)
            self.assertIsInstance(prediction, tf.Tensor)
            self.assertEqual(prediction.shape, (1, 1))
        except Exception as e:
            self.fail(f"Prediction failed for factory-created MUNOEstimator: {e}")

    def test_create_lsg_estimator(self):
        """
        Test Case: Test `EstimatorFactory` with `LSGEstimator`.
        Verify that the factory can successfully create an LSGEstimator.
        """
        variable_name = "DOFA"

        estimator = self.factory.get_estimator(variable_name)

        self.assertIsInstance(
            estimator,
            LSGEstimator,
            "Factory should create an LSGEstimator for DOFA.",
        )

        config = self.factory.configs[variable_name]
        required_inputs = self.factory._get_required_inputs_from_config(config)

        input_data = {
            key: tf.constant([[0.0]], dtype=tf.float32) for key in required_inputs
        }

        try:
            prediction = estimator.predict(input_data)
            self.assertIsInstance(prediction, tf.Tensor)
            self.assertEqual(prediction.shape, (1, 1))
        except Exception as e:
            self.fail(f"Prediction failed for factory-created LSGEstimator: {e}")

    def test_create_lln_estimator(self):
        """
        Test Case: Test `EstimatorFactory` with `LLNEstimator`.
        Verify that the factory can successfully create an LLNEstimator.
        """
        variable_name = "DLL"

        estimator = self.factory.get_estimator(variable_name)

        self.assertIsInstance(
            estimator,
            LLNEstimator,
            "Factory should create an LLNEstimator for DLL.",
        )

        config = self.factory.configs[variable_name]
        required_inputs = self.factory._get_required_inputs_from_config(config)

        input_data = {
            key: tf.constant([[0.0]], dtype=tf.float32) for key in required_inputs
        }

        try:
            prediction = estimator.predict(input_data)
            self.assertIsInstance(prediction, tf.Tensor)
            self.assertEqual(prediction.shape, (1, 1))
        except Exception as e:
            self.fail(f"Prediction failed for factory-created LLNEstimator: {e}")


if __name__ == "__main__":
    unittest.main()
