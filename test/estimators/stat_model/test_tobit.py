import unittest

import numpy as np
import tensorflow as tf

from estimators.stat_model.tobit import TobitIRLS


class TestTobitIRLS(unittest.TestCase):

    def test_convergence_to_known_coefficients(self):
        """Test if the TobitIRLS model can recover known coefficients from censored data."""
        # 1. Generate synthetic data for a Tobit model
        np.random.seed(42)
        num_samples = 5000  # Tobit models often need more data for stable estimation
        num_features = 2

        # True parameters to recover
        true_intercept = np.array([1.0])
        true_weights = np.array([[1.5], [-2.0]])
        true_sigma = 1.5  # True standard deviation of the error term

        X = np.random.rand(num_samples, num_features)

        # 2. Create the latent and observed variables
        # Latent variable y_star = X*beta + intercept + error
        noise = np.random.normal(0, true_sigma, (num_samples, 1))
        y_star = X @ true_weights + true_intercept + noise

        # Observed variable y is censored at 0
        y_observed = np.maximum(0, y_star)

        # Check the degree of censoring
        censoring_fraction = np.mean(y_observed == 0)
        # This test is more meaningful if there's a reasonable amount of censoring
        # print(f"Fraction of censored observations: {censoring_fraction:.2f}")
        self.assertTrue(
            0.1 < censoring_fraction < 0.9,
            "Test requires a moderate level of censoring",
        )

        # 3. Instantiate and fit the model
        model = TobitIRLS(
            max_iterations=100,
            tolerance=1e-5,  # EM algorithms can be sensitive
            patience=10,
        )

        # Use a validation set for early stopping
        split_point = int(num_samples * 0.9)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y_observed[:split_point], y_observed[split_point:]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)

        # 4. Get the estimated coefficients
        estimated_intercept, estimated_weights, estimated_sigma = (
            model.get_coefficients()
        )

        # 5. Assert that the estimated parameters are close to the true parameters
        np.testing.assert_allclose(
            estimated_intercept, true_intercept, rtol=0.1, atol=0.1
        )
        np.testing.assert_allclose(
            estimated_weights, true_weights.flatten(), rtol=0.1, atol=0.1
        )
        np.testing.assert_allclose(estimated_sigma, true_sigma, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    unittest.main()
