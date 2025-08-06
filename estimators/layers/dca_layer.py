import numpy as np
import tensorflow as tf

from estimators.base_layer.hs_layer import HSLayer
from estimators.configs.t7_dca_config import DCA_CONFIG


class DCALayer(tf.keras.layers.Layer):
    """Dedicated estimator for the 'dca' variable."""

    def __init__(self, **kwargs):
        """Initializes DCALayer with feature names."""
        self.feature_names = [
            "sumcasht_1",
            "diffcasht_1",
            "EDEPMAt",
            "EDEPMAt2",
            "SMAt",
            "IMAt",
            "EDEPBUt",
            "EDEPBUt2",
            "IBUt",
            "IBUt2",
            "IBUt3",
            "dclt_1",
            "ddmpat_1",
            "ddmpat_12",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        super().__init__(**kwargs)
        self.hs_layer = HSLayer()

    def build(self, input_shape):
        """Builds the layer weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        num_features = len(self.feature_names)
        tensor_input_shape = tf.TensorShape((None, num_features))
        self.hs_layer.build(tensor_input_shape)
        super().build(input_shape)

    def _assemble_tensor(self, inputs):
        """Converts input dict to an ordered tensor.

        Args:
            inputs: Dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: Concatenated tensor of shape (batch_size, num_features).
        """
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.feature_names
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        """Runs the layer on input data.

        Args:
            inputs: Dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: Output tensor after applying the layer.
        """
        x_tensor = self._assemble_tensor(inputs)
        return self.hs_layer(x_tensor)

    def load_weights_from_cfg(self, cfg):
        """Loads weights from a configuration dictionary.

        Args:
            cfg: Configuration dictionary with coefficients.
        """
        coefficients = cfg["steps"][0]["coefficients"]
        weights = []
        for name in self.feature_names:
            if name in coefficients:
                weights.append(coefficients[name])
            else:
                raise ValueError(f"Coefficient for {name} not found in config.")
        weights = np.array(weights, dtype=np.float32).reshape(
            len(self.feature_names), 1
        )
        bias = np.array([coefficients["Intercept"]])
        self.hs_layer.w.assign(weights)
        self.hs_layer.b.assign(bias)
        print("Weights for 'DCALayer' loaded successfully from config.")


if __name__ == "__main__":

    # 1. Instantiate the specific layer for 'dca'
    dca_layer = DCALayer()
    # 2. Build the layer by calling it on dummy data. This creates the weight tensors.
    # The dummy input must be a dictionary with all the required feature keys.
    dummy_input = {name: tf.zeros((13, 1)) for name in dca_layer.feature_names}
    _ = dca_layer(dummy_input)

    # 3. Call your new method to load the weights
    dca_layer.load_weights_from_cfg(DCA_CONFIG)

    # 4. Verification Step: Check if the weights were loaded correctly
    loaded_weights = dca_layer.get_weights()
    print("\n--- Verification ---")
    print(f"Bias (Intercept) loaded: {loaded_weights[1][0]}")
    print(
        f"First weights weight (for '{dca_layer.feature_names[0]}') loaded: {loaded_weights[0]}"
    )

    # Now the dca_layer is ready for inference with pre-trained weights
    test_input_dict = {
        name: tf.constant([[2.0]], dtype=tf.float32)
        for name in DCALayer().feature_names
    }

    prediction = dca_layer(test_input_dict)
    tf.print("Prediction for test input:", prediction)
