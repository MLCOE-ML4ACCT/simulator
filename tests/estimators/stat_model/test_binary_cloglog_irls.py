import unittest

import numpy as np
import tensorflow as tf

from estimators.stat_model.binary_cloglog_irls import BinaryCLogLogIRLS


class TestBinaryCLogLogIRLS(unittest.TestCase):

    def test_convergence_to_known_coefficients(self):
        """Test if the BinaryCLogLogIRLS model can recover known coefficients."""
        # 1. Generate synthetic data with a known cloglog relationship
        np.random.seed(42)  # for reproducibility
        # A larger sample size is needed to ensure the distribution of the sampled
        # binary outcomes accurately reflects the underlying probabilities.
        num_samples = 5000
        num_features = 2

        # True parameters to recover
        true_intercept = np.array([0.5])
        true_weights = np.array([[-1.5], [2.0]])  # Shape (num_features, 1)

        X = np.random.rand(num_samples, num_features)

        # Calculate probabilities using the cloglog link function
        eta = X @ true_weights + true_intercept
        # p = 1 - exp(-exp(eta))
        probabilities = 1 - np.exp(-np.exp(eta))

        # Generate binary outcomes based on these probabilities
        y = np.random.binomial(1, probabilities, size=(num_samples, 1)).astype(
            np.float32
        )

        # 2. Instantiate and fit the model
        model = BinaryCLogLogIRLS(
            max_iterations=100,  # Increased iterations for larger dataset
            tolerance=1e-6,
            patience=10,
        )

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
            estimated_intercept, true_intercept, rtol=0.2, atol=0.2
        )
        np.testing.assert_allclose(estimated_weights, true_weights, rtol=0.2, atol=0.2)


if __name__ == "__main__":
    unittest.main()
