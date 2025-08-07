import numpy as np
import tensorflow as tf

from estimators.base_layer.hs_layer import HSLayer
from estimators.configs.t7_dca_config import DCA_CONFIG


class DCALayer(tf.keras.layers.Layer):
    """A dedicated TensorFlow layer for estimating the 'dca' variable.

    This layer uses a single Heckman-style selection (HS) layer to model the 'dca'
    variable. It takes a dictionary of input tensors, assembles them into a
    single tensor, and passes them to the HS layer.

    Attributes:
        feature_names (list): A list of strings representing the names of the input
            features required by the layer.
        hs_layer (HSLayer): The underlying Heckman-style selection layer used for
            the estimation.
    """

    def __init__(self, **kwargs):
        """Initializes the DCALayer.

        Args:
            **kwargs: Additional keyword arguments for the parent `tf.keras.layers.Layer`.
        """
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
        """Creates the weights of the layer.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        num_features = len(self.feature_names)
        tensor_input_shape = tf.TensorShape((None, num_features))
        self.hs_layer.build(tensor_input_shape)
        super().build(input_shape)

    def _assemble_tensor(self, inputs):
        """Assembles the input dictionary into a single tensor.

        Args:
            inputs (dict): A dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, num_features).
        """
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.feature_names
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        """Defines the forward pass of the layer.

        Args:
            inputs (dict): A dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: The output tensor from the HS layer.
        """
        x_tensor = self._assemble_tensor(inputs)
        return self.hs_layer(x_tensor)

    def load_weights_from_cfg(self, cfg):
        """Loads the layer's weights from a configuration dictionary.

        Args:
            cfg (dict): A dictionary containing the model's configuration.
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
    test_input = {name: tf.zeros((3, 1)) for name in dca_layer.feature_names}

    prediction = dca_layer(test_input)
    tf.print("Prediction for test input:", prediction)
