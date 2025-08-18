import numpy as np
import tensorflow as tf


class TobitLayer(tf.keras.layers.Layer):
    """
    Keras Layer for Tobit regression.
    Accepts a tensor input of shape [batch, num_features].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        num_features = input_shape[-1]
        self.w = self.add_weight(
            shape=(num_features, 1),
            initializer="zeros",
            trainable=True,
            name="weights",
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        self.scale = self.add_weight(
            shape=(1,),
            initializer="ones",
            trainable=True,
            name="scale",
        )
        super().build(input_shape)

    def _generate_logistic_error_term(self, shape: tf.TensorShape) -> tf.Tensor:
        """
        Generates a random error term from a logistic distribution with a
        given scale. This is the method specified by your colleague.
        """
        epsilon = 1e-9
        # Draw from a uniform distribution
        u = tf.random.uniform(shape=shape, minval=epsilon, maxval=1.0 - epsilon)
        # Inverse transform sampling for the logistic distribution
        error = self.scale * tf.math.log(u / (1.0 - u))
        return error

    def call(self, inputs):
        """
        Args:
            inputs: tf.Tensor of shape [batch, num_features]
        Returns:
            tf.Tensor: logits, shape [batch, 1]
        """
        deterministic_part = tf.matmul(inputs, self.w) + self.b
        deterministic_part = tf.reshape(deterministic_part, [-1, 1])

        num_firms = tf.shape(deterministic_part)[0]
        error_term = self._generate_logistic_error_term(shape=[num_firms, 1])

        # Latent variable
        latent_variable = deterministic_part + error_term

        return tf.maximum(0.0, latent_variable)
