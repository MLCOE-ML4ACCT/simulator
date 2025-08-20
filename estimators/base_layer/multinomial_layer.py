import numpy as np
import tensorflow as tf


class MultinomialLayer(tf.keras.layers.Layer):
    """
    A Keras Layer for multinomial logistic regression that accepts a
    single tensor as input.
    """

    def __init__(self, **kwargs):
        """The layer no longer needs to know about feature names."""
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Creates weights based on the input tensor's shape."""
        # The number of features is inferred directly from the input tensor.
        num_features = input_shape[-1]

        self.w = self.add_weight(
            shape=(num_features, 1),
            initializer="zeros",
            trainable=True,
            name="multinomial_weights",
        )

        self.b = self.add_weight(
            shape=(2,),
            initializer="zeros",
            trainable=True,
            name="multinomial_intercepts",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        """
        Performs the forward pass with a pre-assembled tensor.

        Args:
            inputs (tf.Tensor): A tensor of shape [batch, num_features].

        Returns:
            tf.Tensor: A tensor of shape [batch, 2] containing the two logits.
        """
        # The call method is now much simpler.
        # It works directly with the input tensor 'inputs'.
        base_logit = tf.matmul(inputs, self.w)

        logit1 = base_logit + self.b[0]
        logit2 = base_logit + self.b[1]

        return tf.concat([logit1, logit2], axis=1)
