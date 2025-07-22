from typing import Dict, List

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator
from estimators.models.hs_estimator import HSEstimator
from estimators.models.logistic_estimator import LogisticEstimator
from estimators.utils import create_input_signature, filter_packet


class LLGEstimator(AbstractEstimator):
    """
    A composite estimator for the 'LLG' method. It orchestrates the
    two-step stochastic estimation process.
    """

    def __init__(
        self,
        config: Dict,
        input_signature: List[Dict[str, tf.TensorSpec]],
        num_firms: int = None,
    ):
        super().__init__(config, input_signature, num_firms)

        self.num_firms = num_firms

        # Get the specific "blueprints" for our sub-tasks from the main config.
        prob_config = self.config["steps"][0]
        prob_signature = create_input_signature(
            prob_config["input_variables"], num_firms=num_firms
        )

        level_config = self.config["steps"][1]
        level_signature = create_input_signature(
            level_config["input_variables"], num_firms=num_firms
        )

        # Instantiate the full "worker" models, giving each its own specific config.
        # This correctly handles the case where each step has different inputs.
        self.probability_model = LogisticEstimator(prob_config, prob_signature)
        self.level_model = HSEstimator(level_config, level_signature)

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Executes the two-step stochastic prediction logic. This is the method
        that gets compiled into a high-performance graph by the factory.
        """
        filtered_packet_prob = filter_packet(packet, self.probability_model.config)
        filtered_packet_lv = filter_packet(packet, self.level_model.config)

        # Step 1: Get the potential value from the level model (HSEstimator).
        levels = self.level_model.predict(filtered_packet_lv)

        # Step 2: Perform the stochastic decision using the probability model.
        logit = self.probability_model.predict(filtered_packet_prob)

        P_hat = 1.0 - tf.math.exp(-tf.math.exp(logit))
        num_firms = tf.shape(P_hat)[0]
        U = tf.random.uniform(shape=[num_firms, 1], minval=0, maxval=1)

        should_report_level = tf.cast(P_hat > U, dtype=tf.float32)

        # Step 3: Combine results.
        return levels * should_report_level
