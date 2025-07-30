from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class LogisticEstimator(AbstractEstimator):
    """Estimator for performing logistic regression.

    This model predicts logit values for each firm.
    """

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Calculates the logit values using the logistic function.

        Args:
            packet (Dict[str, tf.Tensor]): Dictionary of input tensors.

        Returns:
            tf.Tensor: Logit values with shape [num_firms, 1].
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

        # Calculate the linear component (logit)
        logit = tf.linalg.matvec(X, weights) + intercept

        # The LLG estimator uses the raw logit for its stochastic decision.
        # We reshape for consistency, ensuring the output is a column vector.
        return tf.reshape(logit, [-1, 1])
