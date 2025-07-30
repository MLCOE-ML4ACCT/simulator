from abc import ABC, abstractmethod
from typing import Dict, List

import tensorflow as tf


class AbstractEstimator(ABC):
    """Base class for all estimators.

    It guarantees that every estimator will have a .predict() method that is
    pre-compiled into a high-performance TensorFlow graph using a specific
    input_signature.
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
            num_firms: Optional number of firms, used to create input signatures.
        """
        self.config = config

        # If no predict method is assigned (for sub-estimators), use _predict_logic
        # Factory will override this for main estimators

    def __getattribute__(self, name):
        """Auto-assign predict method if it's requested but doesn't exist."""
        if name == "predict":
            # If predict method doesn't exist, create it from _predict_logic
            if "predict" not in self.__dict__:
                self.predict = self._predict_logic
        return super().__getattribute__(name)

    def get_input_var_set(self) -> set:
        """
        Returns the set of input variable names required by this estimator.
        This is used to filter input data packets before prediction.
        """
        if "input_variables" in self.config:
            return set(self.config["input_variables"])
        if "steps" in self.config:
            required_inputs = set()
            for step in self.config["steps"]:
                required_inputs.update(step.get("input_variables", []))
            return required_inputs

    @abstractmethod
    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Core prediction logic to be implemented by subclasses.

        Args:
            packet: A dictionary where keys are variable names (e.g., 'MA_t-1')
                    and values are tensors of shape (num_firms, 1).

        Returns:
            A tensor of shape (num_firms, 1) with the predicted values.
        """
        pass
