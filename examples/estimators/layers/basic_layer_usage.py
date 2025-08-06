"""
Basic usage example of a custom TensorFlow layer.

This script demonstrates the fundamental workflow:
1. Instantiate the layer
2. Load the weights from its configuration
3. Create a dummy input tensor
4. Run a prediction
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from estimators.layers.dca_layer import DCALayer
from estimators.configs.t7_dca_config import DCA_CONFIG


def run_basic_layer_example():
    """
    Demonstrates the basic workflow of using a custom TensorFlow layer.
    """
    print("--- Running: Basic Layer Usage Example ---")
    try:
        # 1. Instantiate the specific layer for 'dca'
        dca_layer = DCALayer()
        print("DCALayer instantiated successfully.")

        # 2. Build the layer by calling it on dummy data. This creates the weight tensors.
        dummy_input = {name: tf.zeros((13, 1)) for name in dca_layer.feature_names}
        _ = dca_layer(dummy_input)

        # 3. Load the weights from the configuration
        dca_layer.load_weights_from_cfg(DCA_CONFIG)
        print("Weights loaded successfully from config.")

        # 4. Create a test input
        test_input = {name: tf.random.uniform((3, 1)) for name in dca_layer.feature_names}
        print(f"Created test input with {len(dca_layer.feature_names)} variables.")

        # 5. Run a prediction
        prediction = dca_layer(test_input)
        print(f"Prediction for test input: {prediction.numpy()}")

    except Exception as e:
        print(f"An error occurred during the basic layer example: {e}")
    finally:
        print("--- Finished: Basic Layer Usage Example ---")


if __name__ == "__main__":
    run_basic_layer_example()
