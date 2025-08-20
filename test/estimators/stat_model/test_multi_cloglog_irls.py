import unittest

import numpy as np
import tensorflow as tf

from estimators.stat_model.multi_cloglog_irls import MultinomialOrdinalIRLS


class TestMultinomialOrdinalIRLS(unittest.TestCase):

    def test_convergence_to_known_coefficients(self):
        """Test if the MultinomialOrdinalIRLS model can recover known coefficients."""
        # 1. Generate synthetic data for an ordinal model
        np.random.seed(42)
        num_samples = 5000
        num_features = 2
        num_classes = 3

        # True parameters to recover
        true_weights = np.array([[-1.0], [2.5]])  # Shape (num_features, 1)
        # Intercepts must be ordered for an ordinal model
        true_intercepts = np.array([-1.0, 1.0])  # Two intercepts for 3 classes

        X = np.random.rand(num_samples, num_features)
        eta = X @ true_weights

        # Calculate cumulative probabilities using the cloglog link
        # P(y <= k) = 1 - exp(-exp(intercept_k - eta))
        cum_probs = np.zeros((num_samples, num_classes))
        for k in range(num_classes - 1):
            linear_predictor = true_intercepts[k] - eta
            cum_probs[:, k] = 1 - np.exp(-np.exp(linear_predictor.flatten()))
        cum_probs[:, -1] = 1.0  # P(y <= K) is always 1

        # Calculate individual class probabilities
        # P(y=k) = P(y<=k) - P(y<=k-1)
        class_probs = np.zeros_like(cum_probs)
        class_probs[:, 0] = cum_probs[:, 0]
        for k in range(1, num_classes):
            class_probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]

        # Normalize to handle potential floating point inaccuracies
        class_probs = class_probs / class_probs.sum(axis=1, keepdims=True)

        # Generate categorical outcomes based on these probabilities
        y = np.array([np.random.choice(num_classes, p=p) for p in class_probs])

        # 2. Instantiate and fit the model
        model = MultinomialOrdinalIRLS(
            max_iterations=100,
            tolerance=1e-5,  # Relaxed tolerance a bit for this complex model
            patience=10,
        )

        # Use a validation set for early stopping
        split_point = int(num_samples * 0.9)
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)

        # 3. Get the estimated coefficients
        estimated_intercepts = model.multinomial_layer.b.numpy()
        estimated_weights = model.multinomial_layer.w.numpy()
        # Sort the estimated intercepts as their order might not be strictly enforced during fitting
        estimated_intercepts.sort()

        # 4. Assert that the estimated parameters are close to the true parameters
        np.testing.assert_allclose(
            estimated_intercepts, true_intercepts, rtol=0.1, atol=0.1
        )
        np.testing.assert_allclose(estimated_weights, true_weights, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    unittest.main()
