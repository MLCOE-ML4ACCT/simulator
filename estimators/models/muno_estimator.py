"""
This module contains the composite estimator for the 'MUNO' method, which handles
a three-state (negative, zero, positive) prediction.
"""

from typing import Dict, List

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator
from estimators.models.hs_estimator import HSEstimator
from estimators.models.multinomial_estimator import MultinomialEstimator
from estimators.utils import create_input_signature, filter_packet


class MUNOEstimator(AbstractEstimator):
    """Estimator for the 'MUNO' multinomial outcome model."""

    def __init__(
        self,
        config: Dict,
        input_signature: List[Dict[str, tf.TensorSpec]],
        num_firms: int = None,
    ):
        """Initializes the MUNO estimator and its sub-models."""
        super().__init__(config, input_signature, num_firms)

        # --- Instantiate Sub-Models ---
        # Get the specific "blueprints" for each of the three sub-tasks.
        prob_config = self.config["steps"][0]
        pos_level_config = self.config["steps"][1]
        neg_level_config = self.config["steps"][2]

        # Create the specific input signatures for each sub-model.
        prob_signature = create_input_signature(
            prob_config["input_variables"], num_firms
        )
        pos_level_signature = create_input_signature(
            pos_level_config["input_variables"], num_firms
        )
        neg_level_signature = create_input_signature(
            neg_level_config["input_variables"], num_firms
        )

        # Instantiate the "worker" models for each step.
        self.probability_model = MultinomialEstimator(prob_config, prob_signature)
        self.positive_level_model = HSEstimator(pos_level_config, pos_level_signature)
        self.negative_level_model = HSEstimator(neg_level_config, neg_level_signature)

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Calculates predictions for multinomial outcomes.

        Args:
            packet: Dictionary of input tensors.

        Returns:
            tf.Tensor: Predicted values for each outcome category. [num_firms, 1].
        """
        filtered_packet_pos = filter_packet(packet, self.positive_level_model.config)
        filtered_packet_neg = filter_packet(packet, self.negative_level_model.config)
        filtered_packet = filter_packet(packet, self.probability_model.config)
        # --- Step 1: Predict Potential Levels ---
        # Get the potential positive and negative values from the level models.
        pos_levels = self.positive_level_model.predict(filtered_packet_pos)
        neg_levels = self.negative_level_model.predict(filtered_packet_neg)

        # --- Step 2: Perform Stochastic State Choice via Thresholding ---
        # Get the logits from the multinomial probability model.
        logits = self.probability_model.predict(filtered_packet)

        eta1 = logits[:, 0:1]  # Logit for P(state <= negative)
        eta2 = logits[:, 1:2]  # Logit for P(state <= zero)

        # Boundary for pos/neg levels
        pos_levels = tf.maximum(pos_levels, 0.0)
        neg_levels = tf.minimum(neg_levels, 0.0)
        tf.debugging.assert_greater_equal(
            pos_levels, 0.0, message="Positive levels must be non-negative."
        )
        tf.debugging.assert_less_equal(
            neg_levels, 0.0, message="Negative levels must be non-positive."
        )

        # Enforce the ordering constraint: eta2 must be >= eta1.
        # This makes the model robust to configuration errors.

        eta2 = tf.maximum(eta1, eta2)

        # Convert logits to cumulative probabilities using the cloglog inverse function.
        # P_hat1 is the probability of choosing the 'negative' state.
        # P_hat2 is the probability of choosing 'negative' OR 'zero'.
        P_hat1 = 1.0 - tf.math.exp(-tf.math.exp(eta1))
        P_hat2 = 1.0 - tf.math.exp(-tf.math.exp(eta2))

        tf.debugging.assert_less_equal(
            P_hat1, P_hat2, message="P_hat1 > P_hat2, probability ordering violated."
        )

        # Generate a single uniform random number for each firm to act as a threshold.
        num_firms = tf.shape(P_hat1)[0]
        U = tf.random.uniform(shape=[num_firms, 1], minval=0, maxval=1)

        # Determine the state based on the random draw.
        is_negative = tf.cast(U < P_hat1, dtype=tf.float32)
        is_zero = tf.cast((U >= P_hat1) & (U < P_hat2), dtype=tf.float32)
        is_positive = tf.cast(U >= P_hat2, dtype=tf.float32)

        # --- Step 3: Combine Results ---
        # The final value is the selected state's level.
        # Note: is_zero results in a value of 0, as expected.
        final_value = (
            (is_positive * pos_levels) + (is_negative * neg_levels) + (is_zero * 0.0)
        )

        return final_value
