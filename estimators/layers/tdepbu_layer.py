import numpy as np
import tensorflow as tf

from estimators.base_layer.tobit_layer import TobitLayer
from estimators.configs.t22_tdepbu_config import TDEPBU_CONFIG


class TDEPBULayer(tf.keras.layers.Layer):
    """Keras Layer for the 'tdepbu' variable."""

    def __init__(self, **kwargs):
        self.feature_names = [
            "sumcasht_1",
            "diffcasht_1",
            "EDEPMAt",
            "EDEPMAt2",
            "SMAt",
            "I_MAt",
            "BUt_1",
            "BUt_12",
            "dcat",
            "dcat2",
            "dclt",
            "ddmpat_1",
            "ddmpat_12",
            "ddmpat_13",
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
    tflayer = TDEPBULayer()

    dummy_input = {name: tf.zeros((1, 1)) for name in tflayer.feature_names}
    _ = tflayer(dummy_input)

    tflayer.load_weights_from_cfg(TDEPBU_CONFIG)

    loaded_weights = tflayer.get_weights()
    print("Weights loaded successfully:", loaded_weights)

    test_input = {name: tf.zeros((3, 1)) for name in tflayer.feature_names}
    output = tflayer(test_input)
    print("Output shape:", output.shape)
    print("Output values:", output.numpy())
