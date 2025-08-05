"""
Configuration creation example for the EstimatorFactory.

This script demonstrates how to create and use a mock configuration
with the EstimatorFactory, showing input order independence.
"""

import os

from estimators.utils import create_input_signature

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

from estimators.factory import EstimatorFactory


def run_config_example():
    """
    Demonstrates how to create a mock configuration and use it with the EstimatorFactory.
    Shows that input order doesn't matter as long as the keys match.
    """
    print("--- Running: Config Creation Example ---")

    mock_config = {
        "method": "HS",
        "scale": 1.0,
        "steps": [
            {
                "input_variables": ["var1", "var2"],
                "coefficients": {
                    "Intercept": 1.0,
                    "var1": 1.0,
                    "var2": 2.0,
                },
            }
        ],
    }

    factory = EstimatorFactory(num_firms=3)
    # We load the mock config into the factory's configs dictionary
    # Usually this would be loaded from a file, but here we simulate it.
    factory.configs["TEST_SIGNATURE"] = mock_config

    # Get the estimator for the test signature
    estimator = factory.get_estimator("TEST_SIGNATURE")

    # Create a dummy input packet that matches the estimator's signature
    inputs = {
        "var1": tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
        "var2": tf.constant([[4.0], [5.0], [6.0]], dtype=tf.float32),
    }

    # Run the prediction
    predictions = estimator.predict(inputs)
    print(f"Predictions for TEST_SIGNATURE: {predictions.numpy()}")

    # The order of the inputs can be different from the order in the config,
    # but the keys must match exactly.
    inputs_2 = {
        "var2": tf.constant([[4.0], [5.0], [6.0]], dtype=tf.float32),
        "var1": tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
    }
    predictions_2 = estimator.predict(inputs_2)
    print(
        f"Predictions for TEST_SIGNATURE with reordered inputs: {predictions_2.numpy()}"
    )
    assert np.array_equal(
        predictions.numpy(), predictions_2.numpy()
    ), "Predictions should be the same regardless of input order."

    print("SUCCESS: Confirmed input order independence works correctly!")
    print("--- Finished: Config Creation Example ---\n")


if __name__ == "__main__":
    run_config_example()
