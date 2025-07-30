"""
Basic usage example of the EstimatorFactory.

This script demonstrates the fundamental workflow:
1. Initialize the factory
2. Get an estimator for a specific variable
3. Create dummy input data
4. Run a prediction
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from estimators.factory import EstimatorFactory


def run_basic_example():
    """
    Demonstrates the basic workflow of using the EstimatorFactory:
    1. Initialize the factory.
    2. Get an estimator for a specific variable (e.g., "DCA").
    3. Create dummy input data matching the estimator's requirements.
    4. Run a prediction.
    """
    print("--- Running: Basic Usage Example ---")
    try:
        # 1. Initialize the factory for a simulated population of 100 firms.
        # The factory automatically discovers and loads all configurations from the
        # default directory ('estimators/configs').
        factory = EstimatorFactory(num_firms=100)
        print("EstimatorFactory initialized successfully.")

        # 2. Request an estimator for the 'DCA' variable.
        # The factory finds the 'dca_config.py', determines the required inputs,
        # selects the correct estimator class (e.g., TobitEstimator), and
        # returns a compiled, ready-to-use instance.
        dca_estimator = factory.get_estimator("DCA")
        print("Successfully created estimator for 'DCA'.")

        # 3. Get the list of required input variables for this specific estimator.
        print("This estimator needs to take the following input variables:")
        required_inputs = dca_estimator.get_input_var_set()
        print(f"Required inputs for DCA estimator: {required_inputs}")

        # Create a sample data packet (a dictionary of Tensors).
        # In a real simulation, this data would be loaded from a dataset.
        # Here, we generate random data for demonstration.
        dummy_data_packet = {
            var: tf.random.uniform((100, 1), dtype=tf.float32)
            for var in required_inputs
        }
        print(f"Created dummy data packet with {len(required_inputs)} input variables.")

        # 4. Execute the prediction.
        # The `predict` method is a tf.function, optimized for performance.
        predictions = dca_estimator.predict(dummy_data_packet)
        print(f"DCA predictions generated with shape: {predictions.shape}")
        print(f"First 5 predictions: {predictions.numpy()[:5]}")

    except Exception as e:
        print(f"An error occurred during the basic example: {e}")
    finally:
        print("--- Finished: Basic Usage Example ---\n")


if __name__ == "__main__":
    run_basic_example()
