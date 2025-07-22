from typing import Dict, List

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator
from estimators.models.hs_estimator import HSEstimator
from estimators.models.logistic_estimator import LogisticEstimator
from estimators.utils import create_input_signature, filter_packet


class LLNEstimator(AbstractEstimator):
    """
    A composite estimator for the 'LLN' method. It orchestrates the
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

        probs_config = self.config["steps"][0]
        levels_config = self.config["steps"][1]

        probs_signature = create_input_signature(
            probs_config["input_variables"], num_firms
        )
        levels_signature = create_input_signature(
            levels_config["input_variables"], num_firms
        )
        self.probability_model = LogisticEstimator(probs_config, probs_signature)
        self.level_model = HSEstimator(levels_config, levels_signature)

    def _predict_logic(self, packet):
        """
        Executes the two-step stochastic prediction logic. This entire
        method is vectorized to handle multiple firms in a single call.
        """
        filtered_packet_levels = filter_packet(packet, self.level_model.config)
        filtered_packet_probs = filter_packet(packet, self.probability_model.config)

        levels = self.level_model.predict(filtered_packet_levels)

        eta = self.probability_model.predict(filtered_packet_probs)

        P_hat = tf.math.sigmoid(eta)

        num_firms = tf.shape(P_hat)[0]
        U = tf.random.uniform(shape=[num_firms, 1], minval=0, maxval=1)

        should_report_level = tf.cast(P_hat > U, tf.float32)
        return should_report_level * levels
