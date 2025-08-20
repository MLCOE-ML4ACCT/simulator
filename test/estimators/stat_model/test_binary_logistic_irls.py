import unittest

import numpy as np
import tensorflow as tf

from estimators.stat_model.binary_logistic_irls import BinaryLogisticIRLS


class TestBinaryLogisticIRLS(unittest.TestCase):

    def test_convergence_to_known_coefficients(self):
        """Test if the BinaryLogisticIRLS model can recover known coefficients."""
        # 1. Generate synthetic data with a known logistic relationship
        np.random.seed(42)  # for reproducibility
        num_samples = 5000  # Use a large sample for stable convergence
        num_features = 2

        # True parameters to recover
        true_intercept = np.array([0.5])
        true_weights = np.array([[-1.5], [2.0]])  # Shape (num_features, 1)

        X = np.random.rand(num_samples, num_features)

        # Calculate probabilities using the logistic (sigmoid) link function
        eta = X @ true_weights + true_intercept
        probabilities = 1 / (1 + np.exp(-eta))

        # Generate binary outcomes based on these probabilities
        y = np.random.binomial(1, probabilities, size=(num_samples, 1)).astype(
            np.float32
        )

        # 2. Instantiate and fit the model
        model = BinaryLogisticIRLS(max_iterations=100, tolerance=1e-6, patience=10)

        # Use a small validation set for early stopping
        split_point = int(num_samples * 0.9)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)

        # 3. Get the estimated coefficients
        estimated_intercept = model.logistic_layer.b.numpy()
        estimated_weights = model.logistic_layer.w.numpy()

        # 4. Assert that the estimated parameters are close to the true parameters
        # A reasonable tolerance is needed due to the probabilistic nature of the data.
        np.testing.assert_allclose(
            estimated_intercept, true_intercept, rtol=0.1, atol=0.1
        )
        np.testing.assert_allclose(estimated_weights, true_weights, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    unittest.main()
