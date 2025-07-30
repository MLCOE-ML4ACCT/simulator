"""
Debugging examples for the EstimatorFactory.

This script demonstrates how to use debugging utilities to handle
common input errors like key mismatches and shape mismatches.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from estimators.debug_utils import debug_tf_input_signature
from estimators.factory import EstimatorFactory


def run_mismatch_key_example():
    """
    Demonstrates how to use the `debug_tf_input_signature` utility to
    debug when input keys don't match the estimator's expected signature.
    """
    print("--- Running: Key Mismatch Debug Example ---")
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
        print("--- Finished: Key Mismatch Debug Example ---\n")


def run_mismatch_shape_example():
    """
    Demonstrates how to use the `debug_tf_input_signature` utility to
    debug when input shapes don't match the estimator's expected signature.
    """
    print("--- Running: Shape Mismatch Debug Example ---")
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
        print("--- Finished: Shape Mismatch Debug Example ---\n")


if __name__ == "__main__":
    run_mismatch_key_example()
    print()
    run_mismatch_shape_example()
