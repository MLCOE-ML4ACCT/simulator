from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class DummyZeroEstimator(AbstractEstimator):
    """Estimator that always predicts zero for all inputs."""

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Returns a tensor of zeros for each firm.

        Args:
            packet: Dictionary of input tensors.

        Returns:
            tf.Tensor: Tensor of zeros with shape [num_firms, 1].
        """
        # Testing
        num_firms = tf.shape(next(iter(packet.values())))[0]
        return tf.zeros([num_firms, 1], dtype=tf.float32)
