import numpy as np
import tensorflow as tf

from estimators.base_layer.hs_layer import HSLayer
from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.configs.t20_tl_config import TL_CONFIG


class TLLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.prob_features = [
            "OIBDt",
            "OIBDt2",
            "FIt",
            "FIt2",
            "FEt",
            "FEt2",
            "TDEPMAt",
            "TDEPMAt2",
            "EDEPBUt",
            "EDEPBUt2",
            "dourt",
            "dourt2",
            "ZPFt",
            "PALLOt_1",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.level_features = [
            "OIBDt",
            "OIBDt2",
            "FIt",
            "FIt2",
            "FEt",
            "FEt2",
            "TDEPMAt",
            "TDEPMAt2",
            "EDEPBUt",
            "EDEPBUt2",
            "dourt",
            "dourt2",
            "ZPFt",
            "PALLOt_1",
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
        self.level_layer = HSLayer()

    def build(self):
        all_features = len(self.feature_names)
        all_features_tensor_shape = tf.TensorShape((None, all_features))

        num_prob_features = len(self.prob_features)
        num_level_features = len(self.level_features)

        prob_input_shape = tf.TensorShape((None, num_prob_features))
        level_input_shape = tf.TensorShape((None, num_level_features))

        self.prob_layer.build(prob_input_shape)
        self.level_layer.build(level_input_shape)

        super().build(all_features_tensor_shape)

    def _assemble_prob_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[feature], [-1, 1]) for feature in self.prob_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_level_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[feature], [-1, 1]) for feature in self.level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        for name in self.feature_names:
            if name not in inputs:
                raise ValueError(f"Missing input required feature: {name}")

        level_tensor = self._assemble_level_tensor(inputs)
        prob_tensor = self._assemble_prob_tensor(inputs)

        prob_output = self.prob_layer(prob_tensor)
        level_output = self.level_layer(level_tensor)

        P_hat = tf.math.sigmoid(prob_output)
        U = tf.random.uniform(shape=[tf.shape(P_hat)[0], 1], minval=0, maxval=1)

        should_report_level = tf.cast(P_hat > U, dtype=tf.float32)

        return level_output * should_report_level

    def load_weights_from_cfg(self, cfg):
        # Load weights for the probability layer
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

        print("TLLayer weights loaded from configuration.")


if __name__ == "__main__":
    dll_layer = TLLayer()

    dummy_input = {name: tf.zeros((3, 1)) for name in dll_layer.feature_names}
    _ = dll_layer(dummy_input)

    dll_layer.load_weights_from_cfg(TL_CONFIG)

    loaded_weights = dll_layer.get_weights()
    print("Loaded Weights:", loaded_weights)
    print("TLLayer initialized and weights loaded successfully.")

    test_input = {name: tf.zeros((3, 1)) for name in dll_layer.feature_names}
    prediction = dll_layer(test_input)
    print("Prediction:", prediction)
    print("Output shape:", prediction.shape)
