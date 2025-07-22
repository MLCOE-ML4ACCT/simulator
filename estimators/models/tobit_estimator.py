"""
This module contains the concrete implementation for a Tobit estimator,
designed for censored dependent variables.
"""

from typing import Dict, List

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class TobitEstimator(AbstractEstimator):
    """
    A concrete estimator for the 'Tobit' method.

    This estimator is designed for models where the dependent variable is
    censored (e.g., cannot be negative). It simulates the outcome by:
    1. Generating a random error term from a logistic distribution.
    2. Computing a 'latent' (unobserved) variable by adding the error to the
        linear prediction.
    3. Censoring the latent variable at zero to obtain the final prediction.
    """

    def __init__(
        self,
        config: Dict,
        input_signature: List[Dict[str, tf.TensorSpec]],
        num_firms: int = None,
    ):
        """Initializes the Tobit estimator."""
        super().__init__(config, input_signature, num_firms)
        # The scale parameter is a crucial part of the Tobit calculation.
        self.scale = tf.constant(self.config["scale"], dtype=tf.float32)
        # The Tobit model has a single underlying regression model.
        self.model_config = self.config["steps"][0]

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

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Executes the stochastic Tobit prediction logic based on latent variables.
        """
        # --- Step 1: Calculate the deterministic part of the latent variable ---
        # This is the result of the underlying linear regression (X'B).
        input_tensors = [packet[key] for key in self.model_config["input_variables"]]
        X = tf.concat(input_tensors, axis=1)

        coeffs = self.model_config["coefficients"]
        weights = tf.constant(
            [coeffs[key] for key in self.model_config["input_variables"]],
            dtype=tf.float32,
        )
        intercept = tf.constant(coeffs["Intercept"], dtype=tf.float32)

        deterministic_part = tf.linalg.matvec(X, weights) + intercept
        deterministic_part = tf.reshape(deterministic_part, [-1, 1])

        # --- Step 2: Generate error and compute the full latent variable ---
        num_firms = tf.shape(deterministic_part)[0]
        error_term = self._generate_logistic_error_term(shape=(num_firms, 1))

        # The latent variable y* = X'B + error
        latent_variable = deterministic_part + error_term

        # --- Step 3: Censor the variable to get the final prediction ---
        # The final observed variable y = max(0, y*)
        prediction = tf.maximum(0.0, latent_variable)

        return prediction
