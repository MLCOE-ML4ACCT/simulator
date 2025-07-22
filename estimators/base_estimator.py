from abc import ABC, abstractmethod
from typing import Dict, List

import tensorflow as tf


class AbstractEstimator(ABC):
    """
    The Abstract Base Class (ABC) defines the 'contract' for all estimators.

    It guarantees that every estimator will have a .predict() method that is
    pre-compiled into a high-performance TensorFlow graph using a specific
    input_signature. This is the standard for robust, high-performance tools
    at the Machine Learning Centre of Excellence.
    """

    def __init__(
        self,
        config: Dict,
        input_signature: List[Dict[str, tf.TensorSpec]],
        num_firms: int = None,
    ):
        """
        Initializes the estimator with its specific configuration and a
        dynamically generated input signature for graph compilation.

        Args:
            config: The configuration dictionary for this specific estimator.
            input_signature: The TensorFlow TensorSpec defining the exact
                structure of the input packet, used to prevent retracing.
        """
        self.config = config

        # We compile the predict method into a static graph upon initialization.
        # This is the key to achieving high performance and avoiding retracing.
        self.predict = tf.function(self._predict_logic, input_signature=input_signature)

    @abstractmethod
    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        The core prediction logic for the estimator. This method will be
        wrapped by tf.function in the constructor.

        Args:
            packet: A dictionary where keys are variable names (e.g., 'MA_t-1')
                    and values are tensors of shape (num_firms, 1).

        Returns:
            A tensor of shape (num_firms, 1) with the predicted values.
        """
        pass
