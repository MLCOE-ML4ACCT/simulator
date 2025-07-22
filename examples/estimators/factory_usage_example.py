"""
This script provides a comprehensive example of how to use the EstimatorFactory
to create and run estimators for different flow variables. It also demonstrates
how to use the debugging utilities to inspect the internal workings of an estimator.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

from estimators.debug_utils import debug_tf_input_signature
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


def run_mismatch_key_example():
    """
    Demonstrates how to use the `debug_estimator_prediction` utility to
    inspect the intermediate outputs of a multi-step estimator like 'IMA'.
    """
    print("--- Running: Debug Utils Example ---")
    try:
        # Initialize the factory, this time for 5 firms for clearer debugging.
        factory = EstimatorFactory(num_firms=5)
        print("EstimatorFactory initialized for debugging.")

        # Get the 'IMA' estimator, which is a multi-step model.
        ima_estimator = factory.get_estimator("IMA")
        print("Successfully created estimator for 'IMA'.")

        # Generate dummy data for the 'IMA' estimator.
        ima_required_inputs = ima_estimator.get_input_var_set()
        ima_dummy_data = {
            var: tf.constant(0, shape=[5, 1], dtype=tf.float32)
            for var in ima_required_inputs
        }
        # pop one of the keys to simulate a mismatch
        ima_dummy_data.pop("sumcasht_1", None)

        # add one extra key to simulate a mismatch
        ima_dummy_data["extra_key"] = tf.constant(1, shape=[5, 1], dtype=tf.float32)

        # Use the debug utility to run the prediction and capture intermediate results.
        # This function executes the estimator's logic step-by-step and returns
        # a dictionary containing the output of each defined step.
        _ = ima_estimator.predict(ima_dummy_data)
        print("IMA predictions generated successfully.")
    except TypeError as e:
        debug_tf_input_signature(e, ima_dummy_data)
    finally:
        print("--- Finished: Debug Utils Example ---")


def run_mismatch_shape_example():
    """
    Demonstrates how to use the `debug_estimator_prediction` utility to
    inspect the intermediate outputs of a single-step estimator like 'DCA'.
    """
    print("--- Running: Debug Utils Example for Mismatch Shape ---")
    try:
        # Initialize the factory, this time for 5 firms for clearer debugging.
        factory = EstimatorFactory(num_firms=5)
        print("EstimatorFactory initialized for debugging.")

        # Get the 'DCA' estimator, which is a single-step model.
        dca_estimator = factory.get_estimator("DCA")
        print("Successfully created estimator for 'DCA'.")

        # Generate dummy data for the 'DCA' estimator.
        dca_required_inputs = dca_estimator.get_input_var_set()
        dca_dummy_data = {
            var: tf.constant(0, shape=[5, 1], dtype=tf.float32)
            for var in dca_required_inputs
        }
        # Intentionally create a shape mismatch by changing one variable's shape.
        dca_dummy_data["sumcasht_1"] = tf.constant(0, shape=[5, 2], dtype=tf.float32)

        # Use the debug utility to run the prediction and capture intermediate results.
        _ = dca_estimator.predict(dca_dummy_data)
        print("DCA predictions generated successfully.")
    except TypeError as e:
        debug_tf_input_signature(e, dca_dummy_data)
    finally:
        print("--- Finished: Debug Utils Example for Mismatch Shape ---")


if __name__ == "__main__":
    run_basic_example()
    print()
    run_mismatch_key_example()
    print()
    run_mismatch_shape_example()
