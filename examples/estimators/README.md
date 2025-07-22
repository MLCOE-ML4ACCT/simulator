# Example Usage of the EstimatorFactory

This directory contains examples of how to use the `EstimatorFactory` to instantiate and use estimators within the simulation model.

## `factory_usage_example.py`

This script provides a hands-on demonstration of the factory pattern.

### How it Works

1.  **Initialization**: It starts by creating an instance of `EstimatorFactory`. The factory automatically discovers and loads all estimator configurations from the `estimators/configs/` directory.

2.  **Estimator Retrieval**: It then showcases how to get a specific, compiled estimator by calling `factory.get_estimator("ESTIMATOR_NAME")`, for example, `"DCA"` or `"LLG"`. The factory handles the complex process of:
    *   Reading the estimator's configuration.
    *   Determining the required input variables.
    *   Building a TensorFlow input signature.
    *   Instantiating the correct `AbstractEstimator` subclass.
    *   Compiling the estimator's `predict` method into a high-performance `tf.function`.

3.  **Prediction**: Once an estimator is retrieved, the example shows how to use it for prediction by passing a dictionary of input tensors. It also demonstrates how to inspect the estimator's `input_signature` to understand what data it expects.

### How to Run the Example

You can run the script directly from the root of the repository:

```bash
python -m examples.factory_usage_example
```

This will execute the example function and print out the steps, showing the factory's behavior and the shape of the predictions from the estimators it creates.
