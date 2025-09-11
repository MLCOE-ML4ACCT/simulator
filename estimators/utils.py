"""
This module contains utility functions shared across the estimator framework.
"""

from typing import Dict, List, Union, Optional
import sys
import os

import tensorflow as tf

# Import noise generation functions
try:
    from noises.generate_noise_distributions import (
        create_distribution_for_variable,
        load_noise_parameters
    )
except ImportError:
    # Handle relative import
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from noises.generate_noise_distributions import (
        create_distribution_for_variable,
        load_noise_parameters
    )


def create_input_signature(
    required_inputs: List[str], num_firms: int = None
) -> List[Dict[str, tf.TensorSpec]]:
    """
    Dynamically builds the specific tf.TensorSpec for an estimator.

    This is the core of the performance optimization, preventing retracing by
    creating a precise definition of the expected input tensors.

    Args:
        required_inputs: A list of the names of the input variables.
        num_firms: The number of firms (batch size).
            If None, the dimension is left dynamic.

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


def generate_noise_for_variable(
    var_name: str, 
    batch_size: int, 
    noise_type: str = "pos",
    parameters_file: str = "noises/combined_noise_parameters.json"
) -> Optional[tf.Tensor]:
    """
    Generate noise for a specific variable.
    
    This function creates the distribution for a variable and samples noise
    with shape [batch_size, 1].
    
    Args:
        var_name (str): Variable name (e.g., "OA", "SMA").
        batch_size (int): Number of firms (batch size).
        noise_type (str): Type of noise for multi-stage models ("pos" or "neg").
                         Ignored for single-stage models.
        parameters_file (str): Path to noise parameters file.
        
    Returns:
        tf.Tensor or None: Noise tensor with shape [batch_size, 1].
                          Returns None for Tobit method variables.
    """
    # Load noise parameters
    try:
        parameters = load_noise_parameters(parameters_file)
    except Exception as e:
        raise ValueError(f"Failed to load noise parameters: {e}")
    
    # Check if variable exists in parameters
    if var_name not in parameters:
        raise ValueError(f"Variable '{var_name}' not found in noise parameters.")
    
    var_params = parameters[var_name]
    
    # Create distribution for the variable
    distribution = create_distribution_for_variable(var_name, var_params)
    
    # Handle Tobit method (no noise needed)
    if distribution is None:
        return None
    
    # Handle single distribution methods (HS, LLG, LLN)
    if not isinstance(distribution, dict):
        # Sample noise with correct shape
        noise_sample = distribution.sample(batch_size)
        return tf.expand_dims(noise_sample, axis=-1)
    
    # Handle dual distribution methods (LSG, MUNO)
    if isinstance(distribution, dict):
        if noise_type not in distribution:
            raise ValueError(
                f"Noise type '{noise_type}' not available for variable '{var_name}'. "
                f"Available types: {list(distribution.keys())}"
            )
        
        # Sample from the specified distribution type
        target_distribution = distribution[noise_type]
        noise_sample = target_distribution.sample(batch_size)
        return tf.expand_dims(noise_sample, axis=-1)
    
    raise ValueError(f"Unexpected distribution type for variable '{var_name}'.")
