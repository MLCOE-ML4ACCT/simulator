import numpy as np
import tensorflow as tf

from estimators.base_layer.hs_layer import HSLayer
from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.configs.t10_dsc_config import DSC_CONFIG


class DSCLayer(tf.keras.layers.Layer):
    """A TensorFlow layer for the 'dsc' variable.

    This layer models a four-step process with positive and negative probability
    and level components.
    """

    def __init__(self, **kwargs):
        """Initializes the DSCLayer.

        Args:
            **kwargs: Keyword arguments for the parent class.
        """
        self.pos_prob_features = [
            "sumcasht_1",
            "diffcasht_1",
            "ddmpat_1",
            "ddmpat_12",
            "ddmpat_13",
            "DIMA",
            "DIBU",
            "Ddofa",
            "Ddll",
            "realr",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.neg_prob_features = [
            "sumcasht_1",
            "diffcasht_1",
            "ddmpat_1",
            "ddmpat_12",
            "ddmpat_13",
            "DIMA",
            "DIBU",
            "Ddofa",
            "Ddll",
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
            "DIMA",
            "DIBU",
            "Ddofa",
            "Ddll",
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
            "DIMA",
            "DIBU",
            "Ddofa",
            "Ddll",
            "realr",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        self.feature_names = set(
            self.pos_prob_features
            + self.neg_prob_features
            + self.pos_level_features
            + self.neg_level_features
        )

        super().__init__(**kwargs)
        self.pos_prob_layer = LogisticLayer()
        self.neg_prob_layer = LogisticLayer()
        self.pos_level_layer = HSLayer()
        self.neg_level_layer = HSLayer()

    def build(self):
        num_pos_prob_features = len(self.pos_prob_features)
        num_neg_prob_features = len(self.neg_prob_features)
        num_pos_level_features = len(self.pos_level_features)
        num_neg_level_features = len(self.neg_level_features)

        # Build the probability layer
        pos_prob_input_shape = tf.TensorShape((None, num_pos_prob_features))
        self.pos_prob_layer.build(pos_prob_input_shape)
        neg_prob_input_shape = tf.TensorShape((None, num_neg_prob_features))
        self.neg_prob_layer.build(neg_prob_input_shape)

        # Build the level layer
        pos_level_input_shape = tf.TensorShape((None, num_pos_level_features))
        self.pos_level_layer.build(pos_level_input_shape)
        neg_level_input_shape = tf.TensorShape((None, num_neg_level_features))
        self.neg_level_layer.build(neg_level_input_shape)

        all_input_shape = tf.TensorShape((None, len(self.feature_names)))
        super().build(all_input_shape)

    def _assemble_pos_prob_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.pos_prob_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_neg_prob_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.neg_prob_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_pos_level_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.pos_level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def _assemble_neg_level_tensor(self, inputs):
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.neg_level_features
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        # check input contains all required features
        for name in self.feature_names:
            if name not in inputs:
                raise ValueError(f"Missing input feature: {name}")

        pos_prob_tensor = self._assemble_pos_prob_tensor(inputs)
        neg_prob_tensor = self._assemble_neg_prob_tensor(inputs)
        pos_level_tensor = self._assemble_pos_level_tensor(inputs)
        neg_level_tensor = self._assemble_neg_level_tensor(inputs)

        eta_pos = self.pos_prob_layer(pos_prob_tensor)
        eta_neg = self.neg_prob_layer(neg_prob_tensor)
        pos_levels = self.pos_level_layer(pos_level_tensor)
        neg_levels = self.neg_level_layer(neg_level_tensor)

        P_hat1 = tf.sigmoid(eta_pos)
        P_hat2 = tf.sigmoid(eta_neg)
        P_hat1 = tf.minimum(P_hat1, P_hat2)

        num_firms = tf.shape(P_hat1)[0]
        U = tf.random.uniform(shape=(num_firms, 1), minval=0, maxval=1)

        is_positive = tf.cast(U < P_hat1, dtype=tf.float32)
        is_zero = tf.cast((U >= P_hat1) & (U < P_hat2), dtype=tf.float32)
        is_negative = tf.cast(U >= P_hat2, dtype=tf.float32)

        final_value = (
            (is_positive * pos_levels) + (is_negative * neg_levels) + (is_zero * 0.0)
        )

        return final_value

    def load_weights_from_cfg(self, cfg):
        # for prob layer
        pos_prob_coefficients = cfg["steps"][0]["coefficients"]
        pos_prob_weights = []
        for name in self.pos_prob_features:
            if name in pos_prob_coefficients:
                pos_prob_weights.append(pos_prob_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in prob features.")

        pos_prob_weights = np.array(pos_prob_weights, dtype=np.float32).reshape(
            len(pos_prob_weights), 1
        )
        pos_prob_bias = np.array([pos_prob_coefficients["Intercept"]], dtype=np.float32)

        self.pos_prob_layer.w.assign(pos_prob_weights)
        self.pos_prob_layer.b.assign(pos_prob_bias)

        neg_prob_coefficients = cfg["steps"][1]["coefficients"]
        neg_prob_weights = []
        for name in self.neg_prob_features:
            if name in neg_prob_coefficients:
                neg_prob_weights.append(neg_prob_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in prob features.")

        neg_prob_weights = np.array(neg_prob_weights, dtype=np.float32).reshape(
            len(neg_prob_weights), 1
        )
        neg_prob_bias = np.array([neg_prob_coefficients["Intercept"]], dtype=np.float32)

        self.neg_prob_layer.w.assign(neg_prob_weights)
        self.neg_prob_layer.b.assign(neg_prob_bias)

        pos_level_coefficients = cfg["steps"][2]["coefficients"]
        pos_level_weights = []
        for name in self.pos_level_features:
            if name in pos_level_coefficients:
                pos_level_weights.append(pos_level_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in level features.")

        pos_level_weights = np.array(pos_level_weights, dtype=np.float32).reshape(
            len(pos_level_weights), 1
        )
        pos_level_bias = np.array(
            [pos_level_coefficients["Intercept"]], dtype=np.float32
        )

        self.pos_level_layer.w.assign(pos_level_weights)
        self.pos_level_layer.b.assign(pos_level_bias)

        neg_level_coefficients = cfg["steps"][3]["coefficients"]
        neg_level_weights = []
        for name in self.neg_level_features:
            if name in neg_level_coefficients:
                neg_level_weights.append(neg_level_coefficients[name])
            else:
                raise ValueError(f"Missing coefficient for {name} in level features.")

        neg_level_weights = np.array(neg_level_weights, dtype=np.float32).reshape(
            len(neg_level_weights), 1
        )
        neg_level_bias = np.array(
            [neg_level_coefficients["Intercept"]], dtype=np.float32
        )

        self.neg_level_layer.w.assign(neg_level_weights)
        self.neg_level_layer.b.assign(neg_level_bias)

        print("Weights for 'DSCLayer' loaded successfully.")


if __name__ == "__main__":
    # 1. Instantiate the DSCLayer
    tflayer = DSCLayer()
    dummy_input = {name: tf.zeros((3, 1)) for name in tflayer.feature_names}
    _ = tflayer(dummy_input)

    tflayer.load_weights_from_cfg(DSC_CONFIG)

    loaded_weights = tflayer.get_weights()
    print("Loaded Weights:", loaded_weights)
    print("DSCLayer initialized and weights loaded successfully.")

    test_input = {name: tf.zeros((3, 1)) for name in tflayer.feature_names}

    prediction = tflayer(test_input)
    print("Prediction:", prediction)
