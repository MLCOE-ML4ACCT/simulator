from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class DummyOnesEstimator(AbstractEstimator):
    """A dummy estimator for testing that always predicts ones."""

    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Predicts a tensor of ones based on the input batch size.

        Args:
            packet: A dictionary of tensors, where the first dimension of each
                tensor is the number of firms.

        Returns:
            A tensor of ones with shape (num_firms,).
        """

        num_firms = tf.shape(next(iter(packet.values())))[0]
        return tf.ones([num_firms, 1], dtype=tf.float32)
