import numpy as np
import tensorflow as tf


class LogisticLayer(tf.keras.layers.Layer):
    """
    Keras Layer for logistic regression (logit computation).
    Accepts a tensor input of shape [batch, num_features].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_features = input_shape[-1]
        self.w = self.add_weight(
            shape=(num_features, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="weights",
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: tf.Tensor of shape [batch, num_features]
        Returns:
            tf.Tensor: logits, shape [batch, 1]
        """
        return tf.matmul(inputs, self.w) + self.b

    def get_logits(self, inputs):
        return self.call(inputs)

    def get_probabilities(self, inputs):
        logits = self.call(inputs)
        return tf.sigmoid(logits)

    def loss(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )
