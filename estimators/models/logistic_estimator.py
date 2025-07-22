from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class LogisticEstimator(AbstractEstimator):
    """
    A concrete estimator for performing a logistic regression.
    This model predicts a probability.
    """

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Calculates the probability using the logistic function (sigmoid).
        The complementary log-log function from the paper is approximated
        by the standard logistic function for this prototype.
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
