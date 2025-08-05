import tensorflow as tf


class HSLayer(tf.keras.layers.Layer):
    """
    A Keras Layer for  linear regression using the HS loss.

    This layer functions like a Dense layer but is designed to be trained
    with its own loss function, making it less sensitive to outliers.
    """

    def __init__(self, **kwargs):
        """
        Initializes the layer.
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Creates the trainable weights of the layer (the coefficients).
        This method is automatically called by Keras.
        """
        tf.print(f"Building HSLayer with input shape: {input_shape}")
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="weights",  # Regression coefficients
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True,
            name="bias",  # Intercept term
        )

    def call(self, inputs):
        """
        Performs the forward pass
        """
        return tf.matmul(inputs, self.w) + self.b

    def loss(self, y_true, y_pred):
        # !TODO: Implement the HS loss function
        # This is a normal squared loss
        residuals = tf.abs(y_true - y_pred)

        squared_loss = 0.5 * tf.square(residuals)

        return tf.reduce_mean(squared_loss)
