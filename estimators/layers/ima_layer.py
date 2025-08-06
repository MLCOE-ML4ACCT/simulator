import numpy as np
import tensorflow as tf

from estimators.base_layer.tobit_layer import TobitLayer
from estimators.configs.t3_ima_config import IMA_CONFIG


class IMALayer(tf.keras.layers.Layer):
    """A TensorFlow layer for the 'ima' variable.

    This layer uses a Tobit model to estimate the 'ima' variable.
    """

    def __init__(self, **kwargs):
        """Initializes the IMALayer.

        Args:
            **kwargs: Keyword arguments for the parent class.
        """
        self.feature_names = [
            "sumcasht_1",
            "diffcasht_1",
            "smat",
            "I_BUt_1",
            "EDEPBUt_1",
            "EDEPBUt_12",
            "EDEPMAt",
            "TDEPMAt_1",
            "TDEPMAt_12",
            "ddmtdmt_1",
            "dcat_1",
            "ddmpat_1",
            "ddmpat_12",
            "dclt_1",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        super().__init__(**kwargs)
        self.level_layer = TobitLayer()

    def build(self):
        num_features = len(self.feature_names)
        tensor_input_shape = tf.TensorShape((None, num_features))
        self.level_layer.build(tensor_input_shape)
        super().build(tensor_input_shape)

    def _assemble_tensor(self, inputs):
        """Converts input dict to an ordered tensor."""
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.feature_names
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        """
        Args:
            inputs: Dictionary mapping feature names to tensors.
        Returns:
            tf.Tensor: Tobit model output, shape [batch, 1].
        """
        tensor_input = self._assemble_tensor(inputs)
        return self.level_layer.call(tensor_input)

    def load_weights_from_cfg(self, cfg):
        """
        Loads weights from the configuration dictionary.

        Args:
            cfg: Configuration dictionary containing model coefficients.
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
        bias = np.array([coefficients["Intercept"]], dtype=np.float32)
        scale = np.array([cfg["scale"]], dtype=np.float32)
        self.level_layer.w.assign(weights)
        self.level_layer.b.assign(bias)
        self.level_layer.scale.assign(scale)


if __name__ == "__main__":
    # Example usage
    ima_layer = IMALayer()

    dummy_input = {name: tf.zeros((1, 1)) for name in ima_layer.feature_names}
    _ = ima_layer(dummy_input)

    ima_layer.load_weights_from_cfg(IMA_CONFIG)

    loaded_weights = ima_layer.get_weights()
    print("Weights loaded successfully:", loaded_weights)

    test_input = {name: tf.zeros((3, 1)) for name in ima_layer.feature_names}
    output = ima_layer(test_input)
    print("Output shape:", output.shape)
    print("Output values:", output.numpy())
