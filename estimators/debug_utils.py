import re
from typing import Dict

import tensorflow as tf

from estimators.base_estimator import AbstractEstimator


def debug_tf_input_signature(e, simulation_packet):
    """
    Debugs a TypeError from a tf.function call to identify mismatches in input keys.

    Args:
        e (TypeError): The TypeError exception caught from the tf.function call.
        simulation_packet (dict): The dictionary that was passed as input to the tf.function.
    """
    print(
        "A TypeError occurred. This is often due to a mismatch in the input signature of the tf.function."
    )
    print("Let's inspect the keys...")

    # Extracting the expected keys from the error message
    match = re.search(r"TraceType Dict\[\[(.*)\]\]", str(e))
    if match:
        expected_keys_str = match.group(1)
        # This regex is to extract the keys from the TensorSpec string
        expected_keys = re.findall(r"'([^']*)'", expected_keys_str)

        input_keys = set(simulation_packet.keys())
        expected_keys = set(expected_keys)

        missing_in_input = expected_keys - input_keys
        extra_in_input = input_keys - expected_keys

        if missing_in_input:
            print("\nKeys expected by the function but missing from the input packet:")
            for key in sorted(list(missing_in_input)):
                print(f"- {key}")

        if extra_in_input:
            print(
                "\nKeys present in the input packet but not expected by the function:"
            )
            for key in sorted(list(extra_in_input)):
                print(f"- {key}")
    else:
        print("Could not parse the expected keys from the error message.")
        print("Full error message:", e)
