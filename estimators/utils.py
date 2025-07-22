"""
This module contains utility functions shared across the estimator framework.
"""

from typing import Dict, List

import tensorflow as tf


def create_input_signature(
    required_inputs: List[str], num_firms: int = None
) -> List[Dict[str, tf.TensorSpec]]:
    """
    Dynamically builds the specific tf.TensorSpec for an estimator.

    This is the core of the performance optimization, preventing retracing by
    creating a precise definition of the expected input tensors.

    Args:
        required_inputs: A list of the names of the input variables.
        num_firms: The number of firms (batch size). If None, the dimension
                   is left dynamic.

    Returns:
        A list containing a single dictionary that maps variable names to
        their corresponding TensorSpec, suitable for `tf.function`.
    """
    packet_spec = {
        key: tf.TensorSpec(shape=(num_firms, 1), dtype=tf.float32, name=key)
        for key in required_inputs
    }
    return [packet_spec]


def filter_packet(packet: Dict[str, tf.Tensor], config: Dict):
    """
    Filters a data packet to include only the required variables specified in a config.

    Args:
        packet: The source dictionary of tensors.
        config: The configuration dictionary for a model, which contains
                the "input_variables" list.

    Returns:
        A new dictionary containing only the required key-value pairs.
    """
    required_vars = config["input_variables"]
    return {key: packet[key] for key in required_vars if key in packet}
