from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


class DummyOnesEstimator(AbstractEstimator):
    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Testing
        num_firms = tf.shape(next(iter(packet.values())))[0]
        return tf.ones(num_firms, dtype=tf.float32)
