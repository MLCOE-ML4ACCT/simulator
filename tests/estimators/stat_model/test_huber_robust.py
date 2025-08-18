import unittest

import numpy as np
import tensorflow as tf

from estimators.stat_model.huber_robust import HuberSchweppeIRLS


class TestHuberSchweppeIRLS(unittest.TestCase):

    def test_convergence_to_known_coefficients(self):
        """Test if the HuberSchweppeIRLS model can recover known coefficients."""
        # 1. Generate synthetic data with a known linear relationship
        np.random.seed(42)  # for reproducibility
        num_samples = 200
        num_features = 3

        # True parameters to recover
        true_intercept = np.array([1.5])
        true_weights = np.array([[-2.0], [3.5], [-1.0]])  # Shape (num_features, 1)

        X = np.random.rand(num_samples, num_features)

        # y = X * true_weights + true_intercept + noise
        y_true = X @ true_weights + true_intercept
        noise = np.random.normal(0, 0.5, (num_samples, 1))
        y_with_noise = y_true + noise

        # 2. Introduce some significant outliers
        num_outliers = 10
        outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
        y_with_outliers = np.copy(y_with_noise)
        y_with_outliers[outlier_indices] += np.random.normal(0, 10, (num_outliers, 1))

        # 3. Instantiate and fit the model
        model = HuberSchweppeIRLS(
            max_iterations=100,
            tolerance=1e-6,
            patience=20,  # Using patience for validation-based early stopping
        )

        # Use a small validation set for early stopping
        split_point = int(num_samples * 0.9)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y_with_outliers[:split_point], y_with_outliers[split_point:]

        # We expect the model to be robust to the outliers in y_train
        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)

        # 4. Get the estimated coefficients
        estimated_intercept = model.logistic_layer.b.numpy()
        estimated_weights = model.logistic_layer.w.numpy()
        # 5. Assert that the estimated parameters are close to the true parameters
        # A reasonable tolerance (rtol) is needed due to noise and the iterative fit.
        np.testing.assert_allclose(
            estimated_intercept, true_intercept, rtol=0.1, atol=0.1
        )
        np.testing.assert_allclose(estimated_weights, true_weights, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    unittest.main()
