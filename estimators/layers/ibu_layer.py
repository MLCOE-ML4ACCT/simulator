import numpy as np
import tensorflow as tf

from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.configs.t5_ibu_config import IBU_CONFIG


class IBULayer(tf.keras.layers.Layer):
    """A TensorFlow layer for the 'ibu' variable.

    This layer models a two-step process with a probability and a level component.
    """

    def __init__(self, **kwargs):
        """Initializes the IBULayer.

        Args:
            **kwargs: Keyword arguments for the parent class.
        """
        self.prob_features = [
            "sumcasht_1",
            "diffcasht_1",
            "EDEPMAt",
            "EDEPMAt2",
            "SMAt",
            "IMAt",
            "EDEPBUt",
            "EDEPBUt2",
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
        self.level_features = [
            "sumcasht_1",
            "diffcasht_1",
            "sumcaclt_1",
            "diffcaclt_1",
            "EDEPMAt",
            "EDEPMAt2",
            "SMAt",
            "IMAt",
            "EDEPBUt",
            "EDEPBUt2",
            "ddmpat_1",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.feature_names = set(self.prob_features + self.level_features)
        super().__init__(**kwargs)
        self.prob_layer = LogisticLayer()
        self.level_layer = LogisticLayer()

    def build(self):
        num_prob_features = len(self.prob_features)
        num_level_features = len(self.level_features)

        # Build the probability layer
        prob_input_shape = tf.TensorShape((None, num_prob_features))
        self.prob_layer.build(prob_input_shape)

        # Build the level layer
        level_input_shape = tf.TensorShape((None, num_level_features))
        self.level_layer.build(level_input_shape)

        all_input_shape = tf.TensorShape((None, len(self.feature_names)))
        super().build(all_input_shape)

    def _assemble_prob_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.prob_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_level_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        # check input contains all required features
        for name in self.feature_names:
            if name not in inputs:
                raise ValueError(f"Missing input feature: {name}")

        prob_tensor = self._assemble_prob_tensor(inputs)
        level_tensor = self._assemble_level_tensor(inputs)

        prob_output = self.prob_layer(prob_tensor)
        level_output = self.level_layer(level_tensor)

        P_hat = 1.0 - tf.math.exp(-tf.math.exp(prob_output))
        num_firms = tf.shape(P_hat)[0]
        U = tf.random.uniform(shape=[num_firms, 1], minval=0, maxval=1)
        should_report_level = tf.cast(P_hat > U, dtype=tf.float32)

        return level_output * should_report_level

    def load_weights_from_cfg(self, cfg):
        # for prob layer
        prob_coefficients = cfg["steps"][0]["coefficients"]
        prob_weights = []
        for name in self.prob_features:
            if name in prob_coefficients:
                prob_weights.append(prob_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in prob features.")

        prob_weights = np.array(prob_weights, dtype=np.float32).reshape(
            len(prob_weights), 1
        )
        prob_bias = np.array([prob_coefficients["Intercept"]], dtype=np.float32)

        self.prob_layer.w.assign(prob_weights)
        self.prob_layer.b.assign(prob_bias)

        # for level layer
        level_coefficients = cfg["steps"][1]["coefficients"]
        level_weights = []
        for name in self.level_features:
            if name in level_coefficients:
                level_weights.append(level_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in level features.")

        level_weights = np.array(level_weights, dtype=np.float32).reshape(
            len(level_weights), 1
        )
        level_bias = np.array([level_coefficients["Intercept"]], dtype=np.float32)

        self.level_layer.w.assign(level_weights)
        self.level_layer.b.assign(level_bias)

        print("Weights for 'IBULayer' loaded successfully.")


if __name__ == "__main__":
    # 1. Instantiate the IBULayer
    ibulayer = IBULayer()
    dummy_input = {name: tf.zeros((1, 1)) for name in ibulayer.feature_names}
    _ = ibulayer(dummy_input)

    ibulayer.load_weights_from_cfg(IBU_CONFIG)

    loaded_weights = ibulayer.get_weights()
    print("Loaded Weights:", loaded_weights)
    print("IBULayer initialized and weights loaded successfully.")

    test_input = {name: tf.random.uniform((1, 1)) for name in ibulayer.feature_names}
    test_input = {name: tf.zeros((1, 1)) for name in ibulayer.feature_names}

    prediction = ibulayer(test_input)
    print("Prediction:", prediction)
