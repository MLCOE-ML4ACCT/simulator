# Example Usage of the EstimatorFactory

This directory contains examples of how to use the `EstimatorFactory` to instantiate and use estimators within the simulation model.

## Getting Started

Start with `config_creation_example.py` and `basic_usage_example.py` to understand the core concepts, then explore the other examples as needed.

## Available Examples


### 1. `config_creation_example.py`
Shows how to create and use a mock configuration with the EstimatorFactory:
- Create a custom estimator configuration
- Load it into the factory
- Demonstrates input order independence

**Run with:**
```bash
python -m examples.estimators.config_creation_example
```

### 2. `basic_usage_example.py`
Demonstrates the fundamental workflow of using the EstimatorFactory:
- Initialize the factory
- Get an estimator for a specific variable (e.g., "DCA")
- Create dummy input data matching the estimator's requirements
- Run a prediction

**Run with:**
```bash
python -m examples.estimators.basic_usage_example
```


### 3. `debugging_examples.py`
Demonstrates debugging utilities for common input errors:
- Key mismatch debugging (missing or extra input variables)
- Shape mismatch debugging (incorrect tensor shapes)
- Using `debug_tf_input_signature` for error diagnosis

**Run with:**
```bash
python -m examples.estimators.debugging_examples
```

## How the EstimatorFactory Works

1.  **Initialization**: Creates an instance of `EstimatorFactory`. The factory automatically discovers and loads all estimator configurations from the `estimators/configs/` directory.

2.  **Estimator Retrieval**: Get a specific, compiled estimator by calling `factory.get_estimator("ESTIMATOR_NAME")`. The factory handles:
    *   Reading the estimator's configuration
    *   Determining the required input variables
    *   Building a TensorFlow input signature
    *   Instantiating the correct `AbstractEstimator` subclass
    *   Compiling the estimator's `predict` method into a high-performance `tf.function`

3.  **Prediction**: Use the estimator for prediction by passing a dictionary of input tensors. You can inspect the estimator's `input_signature` to understand what data it expects.


