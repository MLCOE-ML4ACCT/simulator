"""
This module contains the concrete implementation for a Multinomial Logistic
Regression estimator.
"""

from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class MultinomialEstimator(AbstractEstimator):
    """Estimator for multinomial logistic regression.

    Predicts logits for multiple outcome categories.
    """

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Calculates logits for the multinomial model.

        Args:
            packet (Dict[str, tf.Tensor]): Dictionary of input tensors.

        Returns:
            tf.Tensor: Logits for each category boundary, shape [num_firms, 2].
        """
        # Dynamically build the input matrix X from the provided data packet.
        input_tensors = [packet[key] for key in self.config["input_variables"]]
        X = tf.concat(input_tensors, axis=1)

        # Load coefficients from the configuration blueprint.
        coeffs = self.config["coefficients"]
        weights = tf.constant(
            [coeffs[key] for key in self.config["input_variables"]], dtype=tf.float32
        )
        # The multinomial model has J-1 intercepts.
        intercepts = tf.constant(coeffs["Intercept"], dtype=tf.float32)

        # --- Calculate the linear components (logits) ---
        # This calculates the base logit shared by all categories.
        base_logit = tf.linalg.matvec(X, weights)

        # We then create the specific logits for each category boundary by adding
        # the corresponding intercept. For a 3-category model (neg, zero, pos),
        # we calculate 2 logits.
        # logit1 is for P(state <= negative)
        # logit2 is for P(state <= zero)
        logit1 = tf.reshape(base_logit + intercepts[0], [-1, 1])
        logit2 = tf.reshape(base_logit + intercepts[1], [-1, 1])

        # The output is a tensor of shape (num_firms, 2), containing the two
        # calculated logits for each firm.
        return tf.concat([logit1, logit2], axis=1)
