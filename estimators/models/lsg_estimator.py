from typing import Dict, List

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator
from estimators.models.hs_estimator import HSEstimator
from estimators.models.logistic_estimator import LogisticEstimator
from estimators.utils import create_input_signature, filter_packet


class LSGEstimator(AbstractEstimator):
    """Composite estimator for the 'LSG' two-step stochastic estimation method."""

    def __init__(
        self,
        config: Dict,
        input_signature: List[Dict[str, tf.TensorSpec]],
        num_firms: int = None,
    ):
        super().__init__(config, input_signature, num_firms)

        self.num_firms = num_firms

        pos_prob_config = self.config["steps"][0]
        neg_prob_config = self.config["steps"][1]
        pos_level_config = self.config["steps"][2]
        neg_level_config = self.config["steps"][3]

        pos_prob_signature = create_input_signature(
            pos_prob_config["input_variables"], num_firms
        )
        neg_prob_signature = create_input_signature(
            neg_prob_config["input_variables"], num_firms
        )
        pos_level_signature = create_input_signature(
            pos_level_config["input_variables"], num_firms
        )
        neg_level_signature = create_input_signature(
            neg_level_config["input_variables"], num_firms
        )

        self.pos_probability_model = LogisticEstimator(
            pos_prob_config, pos_prob_signature
        )
        self.neg_probability_model = LogisticEstimator(
            neg_prob_config, neg_prob_signature
        )
        self.positive_level_model = HSEstimator(pos_level_config, pos_level_signature)
        self.negative_level_model = HSEstimator(neg_level_config, neg_level_signature)

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Executes the two-step stochastic prediction logic.

        Args:
            packet: Dictionary of input tensors.

        Returns:
            tf.Tensor: Final predicted values after stochastic selection with shape [num_firms, 1].
        """
        filtered_packet_pos = filter_packet(packet, self.pos_probability_model.config)
        filtered_packet_neg = filter_packet(packet, self.neg_probability_model.config)
        filtered_packet_pos_level = filter_packet(
            packet, self.positive_level_model.config
        )
        filtered_packet_neg_level = filter_packet(
            packet, self.negative_level_model.config
        )

        # Step 1: Predict cumulative probabilities for positive and negative states.
        eta_pos = self.pos_probability_model.predict(filtered_packet_pos)
        eta_neg = self.neg_probability_model.predict(filtered_packet_neg)

        # Step 2: Predict the potential levels for positive and negative states.
        pos_levels = self.positive_level_model.predict(filtered_packet_pos_level)
        neg_levels = self.negative_level_model.predict(filtered_packet_neg_level)

        pos_levels = tf.maximum(pos_levels, 0.0)
        neg_levels = tf.minimum(neg_levels, 0.0)

        tf.debugging.assert_greater_equal(
            pos_levels, 0.0, message="Positive levels must be non-negative."
        )
        tf.debugging.assert_less_equal(
            neg_levels, 0.0, message="Negative levels must be non-positive."
        )

        P_hat1 = 1.0 - tf.math.exp(-tf.math.exp(eta_pos))
        P_hat2 = 1.0 - tf.math.exp(-tf.math.exp(eta_neg))
        # Ensure P_hat1 <= P_hat2 to avoid instability in the stochastic selection.
        P_hat1 = tf.minimum(P_hat1, P_hat2)
        # Add a debugging assertion to check for the invalid state
        tf.debugging.assert_less_equal(
            P_hat1, P_hat2, message="LSGEstimator instability: P_hat1 > P_hat2 detected"
        )

        num_firms = tf.shape(eta_pos)[0]
        U = tf.random.uniform(shape=(num_firms, 1), minval=0.0, maxval=1.0)
        # tf.print("*********************************************")
        # tf.print("P_hat1:", P_hat1)
        # tf.print("P_hat2:", P_hat2)
        # tf.print("U:", U)
        # tf.print("*********************************************")

        # Create binary masks using robust boundary conditions (>=) to cover all cases.
        is_positive = tf.cast(U < P_hat1, dtype=tf.float32)
        is_zero = tf.cast((U >= P_hat1) & (U < P_hat2), dtype=tf.float32)
        is_negative = tf.cast(U >= P_hat2, dtype=tf.float32)

        # --- Step 4: Combine Results ---
        final_value = (
            (is_positive * pos_levels) + (is_negative * neg_levels) + (is_zero * 0.0)
        )

        return final_value
