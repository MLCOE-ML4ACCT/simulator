import importlib.util
from pathlib import Path
from typing import Dict, List, Type

import tensorflow as tf

# Import the abstract base class
from estimators.base_estimator import AbstractEstimator
from estimators.models.hs_estimator import HSEstimator
from estimators.models.llg_estimator import LLGEstimator
from estimators.models.lln_estimator import LLNEstimator
from estimators.models.lsg_estimator import LSGEstimator
from estimators.models.muno_estimator import MUNOEstimator
from estimators.models.tobit_estimator import TobitEstimator
from estimators.utils import create_input_signature


class DummyEstimator(AbstractEstimator):
    def _predict_logic(self, packet: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Testing
        num_firms = tf.shape(next(iter(packet.values())))[0]
        return tf.zeros(num_firms, dtype=tf.float32)


class EstimatorFactory:
    """
    A factory for creating and managing estimator objects.

    This class decouples the main simulation engine from the complex details
    of building and configuring individual estimators. It reads a configuration,
    dynamically builds the required input signature, and returns a ready-to-use,
    high-performance estimator object.
    """

    def __init__(
        self, num_firms: int | None = None, config_dir: str = "estimators/configs"
    ):
        self.configs = self._load_configs(config_dir)
        self.num_firms = num_firms

        self._estimator_classes: Dict[str, Type[AbstractEstimator]] = {
            "LLG": LLGEstimator,
            "MUNO": MUNOEstimator,
            "TOBIT": TobitEstimator,
            "HS": HSEstimator,
            "LSG": LSGEstimator,
            "LLN": LLNEstimator,
        }

    def _load_configs(self, config_dir: str) -> Dict[str, Dict]:
        """
        Dynamically loads all estimator configurations from a specified directory.
        This allows for a modular and maintainable configuration system.
        """
        configs = {}
        config_path = Path(config_dir)
        if not config_path.is_dir():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

        for py_file in config_path.glob("*_config.py"):
            if py_file.name.startswith("__"):
                continue

            # Variable name is derived from the filename, e.g., "edepma_config.py" -> "EDEPMA"
            variable_name = py_file.stem.replace("_config", "").upper()

            # Dynamically import the module from the file path
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Convention: The config dictionary inside the file must be named in uppercase,
            # e.g., EDEPMA_CONFIG.
            config_variable_name = f"{variable_name}_CONFIG"
            if not hasattr(module, config_variable_name):
                raise AttributeError(
                    f"Could not find '{config_variable_name}' dict in {py_file}"
                )

            configs[variable_name] = getattr(module, config_variable_name)

        return configs

    def _get_required_inputs_from_config(self, config: Dict) -> List[str]:
        """Parses a config to find all unique input variables required."""
        required_inputs = set()
        if "steps" in config:
            for step in config["steps"]:
                required_inputs.update(step.get("input_variables", []))
        else:
            required_inputs.update(config.get("input_variables", []))

        # Add universal inputs if any (e.g., firm ID, characteristics)
        # For now, we assume all are in the config list.
        return sorted(list(required_inputs))

    def get_estimator(self, variable_name: str) -> AbstractEstimator:
        """
        Retrieves a configured and compiled estimator.
        Uses a cache to ensure each estimator is built only once.

        Args:
            variable_name: The name of the flow variable to be estimated (e.g., "EDEPMA").

        Returns:
            An initialized instance of an AbstractEstimator subclass.
        """

        if variable_name not in self.configs:
            raise ValueError(f"No configuration found for variable: '{variable_name}'")

        config = self.configs[variable_name]
        method_shorthand = config.get("method")
        if not method_shorthand:
            raise ValueError(
                f"Config for '{variable_name}' is missing the 'method' key."
            )

        if method_shorthand not in self._estimator_classes:
            raise NotImplementedError(
                f"Estimator method '{method_shorthand}' is not registered in the factory."
            )

        EstimatorClass = self._estimator_classes[method_shorthand]

        # 1. Determine the exact inputs this estimator needs from its config.
        required_inputs = self._get_required_inputs_from_config(config)

        # 2. Dynamically create the specific input signature for this estimator.
        input_signature = create_input_signature(required_inputs, self.num_firms)

        # 3. For single-step models, extract the step config to match expected format
        estimator_config = config
        if (
            "steps" in config
            and len(config["steps"]) == 1
            and method_shorthand in ["HS"]
        ):
            estimator_config = config["steps"][0]

        # 4. Create and compile the estimator instance.
        estimator = EstimatorClass(estimator_config, input_signature)

        return estimator
