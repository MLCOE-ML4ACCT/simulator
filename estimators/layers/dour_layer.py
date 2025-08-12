import numpy as np
import tensorflow as tf

from estimators.base_layer.hs_layer import HSLayer
from estimators.base_layer.multinomial_layer import MultinomialLayer
from estimators.configs.t17_dour_config import DOUR_CONFIG


class DOURLayer(tf.keras.layers.Layer):
    """A TensorFlow layer for the 'dour' variable.

    This layer models a three-step process with a probability and positive/negative
    level components.
    """

    def __init__(self, **kwargs):
        """Initializes the DOURLayer.

        Args:
            **kwargs: Keyword arguments for the parent class.
        """
        self.prob_features = [
            "sumcasht_1",
            "diffcasht_1",
            "ddmpat_1",
            "ddmpat_12",
            "ddmpat_13",
            "DTDEPMA",
            "DZPF",
            "realr",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]

        self.pos_level_features = [
            "sumcasht_1",
            "diffcasht_1",
            "ddmpat_1",
            "ddmpat_12",
            "DTDEPMA",
            "DZPF",
            "realr",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.neg_level_features = [
            "sumcasht_1",
            "diffcasht_1",
            "ddmpat_1",
            "DTDEPMA",
            "DZPF",
            "realr",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.feature_names = set(
            self.prob_features + self.pos_level_features + self.neg_level_features
        )
        super().__init__(**kwargs)
        self.prob_layer = MultinomialLayer()
        self.pos_level_layer = HSLayer()
        self.neg_level_layer = HSLayer()

    def build(self):
        num_prob_features = len(self.prob_features)
        num_pos_level_features = len(self.pos_level_features)
        num_neg_level_features = len(self.neg_level_features)

        tensor_input_shape = tf.TensorShape((None, num_prob_features))
        self.prob_layer.build(tensor_input_shape)

        pos_level_input_shape = tf.TensorShape((None, num_pos_level_features))
        self.pos_level_layer.build(pos_level_input_shape)

        neg_level_input_shape = tf.TensorShape((None, num_neg_level_features))
        self.neg_level_layer.build(neg_level_input_shape)

        super().build((None, len(self.feature_names)))

    def _assemble_prob_tensor(self, inputs):
        """Converts input dict to an ordered tensor for probability features."""
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.prob_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_pos_level_tensor(self, inputs):
        """Converts input dict to an ordered tensor for positive level features."""
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.pos_level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_neg_level_tensor(self, inputs):
        """Converts input dict to an ordered tensor for negative level features."""
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.neg_level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        """Runs the layer on input data."""
        prob_tensor = self._assemble_prob_tensor(inputs)
        pos_level_tensor = self._assemble_pos_level_tensor(inputs)
        neg_level_tensor = self._assemble_neg_level_tensor(inputs)

        pos_level = self.pos_level_layer(pos_level_tensor)
        neg_level = self.neg_level_layer(neg_level_tensor)

        prob = self.prob_layer(prob_tensor)

        eta1 = prob[:, 0:1]
        eta2 = prob[:, 1:2]

        pos_level = tf.maximum(pos_level, 0.0)
        neg_level = tf.minimum(neg_level, 0.0)

        eta2 = tf.maximum(eta1, eta2)

        P_hat1 = 1 - tf.exp(-tf.exp(eta1))
        P_hat2 = 1 - tf.exp(-tf.exp(eta2))

        num_firms = tf.shape(P_hat1)[0]
        U = tf.random.uniform(shape=[num_firms, 1], minval=0, maxval=1)

        is_negative = tf.cast(U < P_hat1, dtype=tf.float32)
        is_zero = tf.cast((U >= P_hat1) & (U < P_hat2), dtype=tf.float32)
        is_positive = tf.cast(U >= P_hat2, dtype=tf.float32)

        final_value = (
            (is_positive * pos_level) + (is_negative * neg_level) + (is_zero * 0.0)
        )

        return final_value

    def load_weights_from_cfg(self, cfg):
        """Loads weights from a configuration dictionary."""
        prob_coefficients = cfg["steps"][0]["coefficients"]
        prob_weights = []
        for name in self.prob_features:
            if name in prob_coefficients:
                prob_weights.append(prob_coefficients[name])
            else:
                raise ValueError(f"Coefficient for {name} not found in config.")

        prob_weights = np.array(prob_weights, dtype=np.float32).reshape(
            len(self.prob_features), 1
        )
        prob_bias = np.array(prob_coefficients["Intercept"], dtype=np.float32)

        self.prob_layer.w.assign(prob_weights)
        self.prob_layer.b.assign(prob_bias)

        pos_level_coefficients = cfg["steps"][1]["coefficients"]
        pos_level_weights = []
        for name in self.pos_level_features:
            if name in pos_level_coefficients:
                pos_level_weights.append(pos_level_coefficients[name])
            else:
                raise ValueError(f"Coefficient for {name} not found in config.")

        pos_level_weights = np.array(pos_level_weights, dtype=np.float32).reshape(
            len(self.pos_level_features), 1
        )
        pos_level_bias = np.array(
            [pos_level_coefficients["Intercept"]], dtype=np.float32
        )

        self.pos_level_layer.w.assign(pos_level_weights)
        self.pos_level_layer.b.assign(pos_level_bias)

        neg_level_coefficients = cfg["steps"][2]["coefficients"]
        neg_level_weights = []
        for name in self.neg_level_features:
            if name in neg_level_coefficients:
                neg_level_weights.append(neg_level_coefficients[name])
            else:
                raise ValueError(f"Coefficient for {name} not found in config.")

        neg_level_weights = np.array(neg_level_weights, dtype=np.float32).reshape(
            len(self.neg_level_features), 1
        )
        neg_level_bias = np.array(
            [neg_level_coefficients["Intercept"]], dtype=np.float32
        )

        self.neg_level_layer.w.assign(neg_level_weights)
        self.neg_level_layer.b.assign(neg_level_bias)


if __name__ == "__main__":
    # 1. Instantiate the DOURLayer
    tf_layer = DOURLayer()
    dummy_input = {name: tf.zeros((5, 1)) for name in tf_layer.feature_names}
    _ = tf_layer(dummy_input)

    # 2. Load weights from configuration
    tf_layer.load_weights_from_cfg(DOUR_CONFIG)

    print("DOURLayer initialized and weights loaded successfully.")

    # 3. Test the layer with dummy input
    prediction = tf_layer(dummy_input)
    print("Prediction:", prediction)

    # 4. Print the weights to verify
    print("Weights:", tf_layer.get_weights())

    # 5. Print the feature names to verify
    print("Feature Names:", tf_layer.feature_names)
