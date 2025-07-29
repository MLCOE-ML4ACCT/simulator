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


class EstimatorFactory:
    """Factory for creating and managing estimator objects.

    Decouples the simulation engine from estimator configuration and instantiation.
    Reads configuration files and builds estimator input signatures.
    """

    def __init__(
        self, num_firms: int | None = None, config_dir: str = "estimators/configs"
    ):
        """Initializes the factory.

        Args:
            num_firms (int | None): Optional number of firms.
            config_dir (str): Directory where estimator configuration files are located.
        """
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
        """Loads estimator configurations from a directory.

        Args:
            config_dir (str): Path to the configuration directory.

        Returns:
            Dict[str, Dict]: Dictionary of loaded configurations.
        """
        configs = {}
        config_path = Path(config_dir)
        if not config_path.is_dir():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

        for py_file in config_path.glob("*_config.py"):
            if py_file.name.startswith("__"):
                continue

            # Variable name is derived from the filename
            # Handle both old format: "edepma_config.py" -> "EDEPMA"
            # and new format: "t12_oibd_config.py" -> "OIBD"
            filename_stem = py_file.stem.replace("_config", "")
            # Remove the "t{number}_" prefix if present
            if filename_stem.startswith("t") and "_" in filename_stem:
                # Split on underscore and take everything after the first part if it's a number prefix
                parts = filename_stem.split("_", 1)
                if (
                    len(parts) > 1 and parts[0][1:].isdigit()
                ):  # Check if t{number} format
                    variable_name = parts[1].upper()
                else:
                    variable_name = filename_stem.upper()
            else:
                variable_name = filename_stem.upper()

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
        """Finds all unique input variables required by an estimator config.

        Args:
            config (Dict): Estimator configuration dictionary.

        Returns:
            List[str]: List of required input variable names.
        """
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
        """Retrieves a configured and compiled estimator instance.

        Args:
            variable_name (str): Name of the flow variable to be estimated (e.g., "EDEPMA").

        Returns:
            AbstractEstimator: Initialized estimator instance.
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
        estimator = EstimatorClass(estimator_config, input_signature, self.num_firms)

        # Assign predict method based on estimator type
        # Apply concrete function approach to all estimators for consistency and performance
        def create_concrete_predict_function(
            estimator_instance, variable_name, method_type
        ):
            """Create a concrete, non-retracing prediction function for any estimator."""

            # All estimators now use consistent _predict_logic method name
            predict_method = estimator_instance._predict_logic

            # Create tf.function from the estimator's predict logic
            tf_function = tf.function(predict_method, input_signature=input_signature)

            # Get concrete function - this will never retrace
            sample_input = {
                k: tf.zeros(spec.shape, dtype=spec.dtype)
                for k, spec in input_signature[0].items()
            }
            concrete_function = tf_function.get_concrete_function(sample_input)

            return concrete_function

        # Apply concrete functions to all estimator types
        if method_shorthand == "LLG":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "LLG"
            )

        elif method_shorthand == "HS":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "HS"
            )

        elif method_shorthand == "MUNO":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "MUNO"
            )

        elif method_shorthand == "TOBIT":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "TOBIT"
            )

        elif method_shorthand == "LSG":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "LSG"
            )

        elif method_shorthand == "LLN":
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "LLN"
            )

        else:
            # Fallback for any other estimator types
            estimator.predict = create_concrete_predict_function(
                estimator, variable_name, "OTHER"
            )

        return estimator
