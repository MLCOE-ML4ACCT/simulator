from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class HSEstimator(AbstractEstimator):
    """Estimator for Huber-Schweppes robust regression.

    Uses a linear regression model for prediction.
    """

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Calculates predictions using a linear model.

        Args:
            packet (Dict[str, tf.Tensor]): Dictionary of input tensors.

        Returns:
            tf.Tensor: Predicted values as a column vector [num_firms, 1].
        """
        # Dynamically build the input matrix X
        input_tensors = [packet[key] for key in self.config["input_variables"]]
        X = tf.concat(input_tensors, axis=1)

        # Load coefficients from the config
        coeffs = self.config["coefficients"]

        weights = tf.constant(
            [coeffs[key] for key in self.config["input_variables"]], dtype=tf.float32
        )
        intercept = tf.constant(coeffs["Intercept"], dtype=tf.float32)

        # Perform the linear regression calculation
        predictions = tf.linalg.matvec(X, weights) + intercept

        # Reshape the output to be a column vector (e.g., [num_firms, 1]) for consistency.
        return tf.reshape(predictions, [-1, 1])
