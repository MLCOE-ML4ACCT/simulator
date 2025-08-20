import numpy as np
import tensorflow as tf

from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.configs.t12_oibd_config import OIBD_CONFIG


class OIBDLayer(tf.keras.layers.Layer):
    """A TensorFlow layer for the 'oibd' variable.

    This layer uses a single Heckman-style selection (HS) to model the 'oibd' variable.
    """

    def __init__(self, **kwargs):
        """Initializes the OIBDLayer.

        Args:
            **kwargs: Keyword arguments for the parent class.
        """
        self.feature_names = [
            "sumcaclt_1",
            "diffcaclt_1",
            "MAt_1",
            "I_MAt",
            "SMAt",
            "EDEPMAt",
            "EDEPMAt2",
            "BUt_1",
            "I_BUt",
            "EDEPBUt",
            "EDEPBUt2",
            "dcat",
            "dcat2",
            "ddmpat_1",
            "ddmpat_12",
            "ddmpat_13",
            "dcasht_1",
            "dcasht_12",
            "dclt",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        super().__init__(**kwargs)
        self.logistic_layer = LogisticLayer()

    def build(self, input_shape):
        """Builds the layer weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        num_features = len(self.feature_names)
        tensor_input_shape = tf.TensorShape((None, num_features))
        self.logistic_layer.build(tensor_input_shape)
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
        return self.logistic_layer(x_tensor)

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
        self.logistic_layer.w.assign(weights)
        self.logistic_layer.b.assign(bias)
        print("Weights for 'OIBDLayer' loaded successfully.")


if __name__ == "__main__":

    # 1. Instantiate the specific layer
    tflayer = OIBDLayer()
    # 2. Build the layer by calling it on dummy data. This creates the weight tensors.
    # The dummy input must be a dictionary with all the required feature keys.
    dummy_input = {name: tf.zeros((13, 1)) for name in tflayer.feature_names}
    _ = tflayer(dummy_input)

    # 3. Call your new method to load the weights
    tflayer.load_weights_from_cfg(OIBD_CONFIG)

    # 4. Verification Step: Check if the weights were loaded correctly
    loaded_weights = tflayer.get_weights()
    print("\n--- Verification ---")
    print(f"Bias (Intercept) loaded: {loaded_weights[1][0]}")
    print(
        f"First weights weight (for '{tflayer.feature_names[0]}') loaded: {loaded_weights[0]}"
    )

    # Now the tflayer is ready for inference with pre-trained weights
    test_input = {name: tf.zeros((3, 1)) for name in tflayer.feature_names}

    prediction = tflayer(test_input)
    tf.print("Prediction for test input:", prediction)
