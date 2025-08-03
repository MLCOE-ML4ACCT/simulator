import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path for importing noises module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noises.generate_noise_distributions import (
    load_noise_parameters,
    create_johnson_su_distribution,
    create_gaussian_mixture_distribution,
    create_distribution_for_variable,
    generate_all_distributions,
    sample_from_distributions,
)


class TestGenerateNoiseDistributions(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        tf.config.run_functions_eagerly(True)
        tf.random.set_seed(42)

    def test_load_noise_parameters(self):
        """Test loading noise parameters from JSON file"""
        # Create a temporary test JSON file
        test_params = {
            "var1": {
                "method": "HS",
                "johnsonSU_has_result": True,
                "gamma": 0.1,
                "delta": 0.8,
                "xi": 0.0,
                "lambda": 1.0,
            },
            "var2": {
                "method": "LLG",
                "johnsonSU_has_result": False,
                "weights": [0.5, 0.5],
                "means": [0.0, 2.0],
                "stds": [1.0, 1.0],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_params, f)
            temp_filename = f.name

        try:
            # Test loading parameters
            loaded_params = load_noise_parameters(temp_filename)

            # Check that parameters are loaded correctly
            assert loaded_params == test_params
            assert "var1" in loaded_params
            assert "var2" in loaded_params
            assert loaded_params["var1"]["method"] == "HS"
            assert loaded_params["var2"]["method"] == "LLG"
        finally:
            # Clean up temporary file
            os.unlink(temp_filename)

    def test_create_johnson_su_distribution(self):
        """Test creating Johnson SU distribution"""
        gamma = 0.1
        delta = 0.8
        xi = 0.0
        lambda_param = 1.0

        dist = create_johnson_su_distribution(gamma, delta, xi, lambda_param)

        # Check that it's a JohnsonSU distribution
        assert isinstance(dist, tfp.distributions.JohnsonSU)

        # Check parameters
        assert dist.skewness == gamma
        assert dist.tailweight == delta
        assert dist.loc == xi
        assert dist.scale == lambda_param

        # Test sampling
        samples = dist.sample(100)
        assert samples.shape == (100,)

    def test_create_gaussian_mixture_distribution_basic(self):
        """Test creating basic Gaussian mixture distribution without transformation"""
        weights = [0.3, 0.7]
        means = [0.0, 2.0]
        stds = [1.0, 1.5]

        dist = create_gaussian_mixture_distribution(weights, means, stds)

        # Check that it's a MixtureSameFamily distribution
        assert isinstance(dist, tfp.distributions.MixtureSameFamily)

        # Test sampling
        samples = dist.sample(100)
        assert samples.shape == (100,)

        # Check that weights sum to 1
        assert tf.reduce_sum(dist.mixture_distribution.probs) == 1.0

    def test_create_gaussian_mixture_distribution_with_transformation(self):
        """Test creating Gaussian mixture distribution with linear transformation"""
        weights = [0.5, 0.5]
        means = [0.0, 2.0]
        stds = [1.0, 1.0]
        target_mean = 5.0
        target_variance = 4.0

        dist = create_gaussian_mixture_distribution(
            weights, means, stds, target_mean, target_variance
        )

        # Check that it's a TransformedDistribution
        assert isinstance(dist, tfp.distributions.TransformedDistribution)

        # Test sampling
        samples = dist.sample(1000)
        assert samples.shape == (1000,)

        # Check that empirical mean and variance are close to targets
        emp_mean = tf.reduce_mean(samples)
        emp_var = tf.reduce_mean(tf.square(samples - emp_mean))

        assert abs(emp_mean - target_mean) < 0.5
        assert abs(emp_var - target_variance) < 1.0

    def test_create_distribution_for_variable_tobit(self):
        """Test creating distribution for Tobit method"""
        var_name = "test_var"
        var_params = {"method": "Tobit"}

        result = create_distribution_for_variable(var_name, var_params)

        # Tobit method should return None
        assert result is None

    def test_create_distribution_for_variable_hs_johnson(self):
        """Test creating distribution for HS method with Johnson SU"""
        var_name = "test_var"
        var_params = {
            "method": "HS",
            "johnsonSU_has_result": True,
            "gamma": 0.1,
            "delta": 0.8,
            "xi": 0.0,
            "lambda": 1.0,
        }

        result = create_distribution_for_variable(var_name, var_params)

        # Should return a JohnsonSU distribution
        assert isinstance(result, tfp.distributions.JohnsonSU)

    def test_create_distribution_for_variable_hs_gmm(self):
        """Test creating distribution for HS method with GMM"""
        var_name = "test_var"
        var_params = {
            "method": "HS",
            "johnsonSU_has_result": False,
            "weights": [0.5, 0.5],
            "means": [0.0, 2.0],
            "stds": [1.0, 1.0],
        }

        # Mock Flow_info to provide target statistics
        with patch(
            "noises.generate_noise_distributions.Flow_info",
            {"test_var": {"mean": 1.0, "variance": 2.0}},
        ):
            result = create_distribution_for_variable(var_name, var_params)

        # Should return a TransformedDistribution (GMM with linear transformation)
        assert isinstance(result, tfp.distributions.TransformedDistribution)

    def test_create_distribution_for_variable_lsg(self):
        """Test creating distribution for LSG method"""
        var_name = "test_var"
        var_params = {
            "method": "LSG",
            "pos": {
                "johnsonSU_has_result": True,
                "gamma": 0.1,
                "delta": 0.8,
                "xi": 0.0,
                "lambda": 1.0,
            },
            "neg": {
                "johnsonSU_has_result": False,
                "weights": [0.5, 0.5],
                "means": [0.0, 2.0],
                "stds": [1.0, 1.0],
            },
        }

        # Mock Flow_info to provide target statistics
        with patch(
            "noises.generate_noise_distributions.Flow_info",
            {
                "test_var": {
                    "mean_pos": 1.0,
                    "variance_pos": 2.0,
                    "mean_neg": -1.0,
                    "variance_neg": 2.0,
                }
            },
        ):
            result = create_distribution_for_variable(var_name, var_params)

        # Should return a dictionary with pos and neg distributions
        assert isinstance(result, dict)
        assert "pos" in result
        assert "neg" in result
        assert isinstance(result["pos"], tfp.distributions.JohnsonSU)
        assert isinstance(result["neg"], tfp.distributions.TransformedDistribution)

    def test_create_distribution_for_variable_unknown_method(self):
        """Test creating distribution for unknown method"""
        var_name = "test_var"
        var_params = {"method": "UNKNOWN"}

        # Should raise ValueError for unknown method
        with self.assertRaises(ValueError):
            create_distribution_for_variable(var_name, var_params)

    @patch("noises.generate_noise_distributions.load_noise_parameters")
    def test_generate_all_distributions(self, mock_load_params):
        """Test generating all distributions"""
        # Mock parameters
        mock_params = {
            "var1": {
                "method": "HS",
                "johnsonSU_has_result": True,
                "gamma": 0.1,
                "delta": 0.8,
                "xi": 0.0,
                "lambda": 1.0,
            },
            "var2": {"method": "Tobit"},
            "var3": {
                "method": "LSG",
                "pos": {
                    "johnsonSU_has_result": True,
                    "gamma": 0.1,
                    "delta": 0.8,
                    "xi": 0.0,
                    "lambda": 1.0,
                },
                "neg": {
                    "johnsonSU_has_result": False,
                    "weights": [0.5, 0.5],
                    "means": [0.0, 2.0],
                    "stds": [1.0, 1.0],
                },
            },
        }
        mock_load_params.return_value = mock_params

        # Mock Flow_info
        with patch(
            "noises.generate_noise_distributions.Flow_info",
            {
                "var3": {
                    "mean_pos": 1.0,
                    "variance_pos": 2.0,
                    "mean_neg": -1.0,
                    "variance_neg": 2.0,
                }
            },
        ):
            distributions = generate_all_distributions()

        # Check that distributions are created for all variables
        assert "var1" in distributions
        assert "var2" in distributions
        assert "var3" in distributions

        # Check distribution types
        assert isinstance(distributions["var1"], tfp.distributions.JohnsonSU)
        assert distributions["var2"] is None  # Tobit method
        assert isinstance(distributions["var3"], dict)
        assert "pos" in distributions["var3"]
        assert "neg" in distributions["var3"]

    def test_sample_from_distributions_single(self):
        """Test sampling from single distributions"""
        # Create test distributions
        johnson_dist = create_johnson_su_distribution(0.1, 0.8, 0.0, 1.0)
        gmm_dist = create_gaussian_mixture_distribution(
            [0.5, 0.5], [0.0, 2.0], [1.0, 1.0]
        )

        distributions = {
            "var1": johnson_dist,
            "var2": gmm_dist,
            "var3": None,  # Tobit method
        }

        samples = sample_from_distributions(distributions, num_samples=100)

        # Check that samples are generated for non-None distributions
        assert "var1" in samples
        assert "var2" in samples
        assert "var3" not in samples  # Tobit method should be skipped

        # Check sample shapes
        assert samples["var1"].shape == (100, 1)
        assert samples["var2"].shape == (100, 1)

    def test_sample_from_distributions_dual(self):
        """Test sampling from dual distributions (LSG/MUNO)"""
        # Create test dual distributions
        pos_dist = create_johnson_su_distribution(0.1, 0.8, 0.0, 1.0)
        neg_dist = create_gaussian_mixture_distribution(
            [0.5, 0.5], [0.0, 2.0], [1.0, 1.0]
        )

        distributions = {"var1": {"pos": pos_dist, "neg": neg_dist}}

        samples = sample_from_distributions(distributions, num_samples=100)

        # Check that samples are generated for both pos and neg
        assert "var1" in samples
        assert "pos" in samples["var1"]
        assert "neg" in samples["var1"]

        # Check sample shapes
        assert samples["var1"]["pos"].shape == (100, 1)
        assert samples["var1"]["neg"].shape == (100, 1)

    def test_numerical_stability(self):
        """Test numerical stability of distribution creation and sampling"""
        # Test with extreme parameters
        gamma = 0.0
        delta = 0.5
        xi = 1e6
        lambda_param = 1e-6

        dist = create_johnson_su_distribution(gamma, delta, xi, lambda_param)
        samples = dist.sample(1000)

        # Check that samples are finite
        assert tf.reduce_all(tf.math.is_finite(samples))

        # Test GMM with extreme parameters
        weights = [0.1, 0.9]
        means = [1e6, -1e6]
        stds = [1e-6, 1e6]

        gmm_dist = create_gaussian_mixture_distribution(weights, means, stds)
        gmm_samples = gmm_dist.sample(1000)

        # Check that samples are finite
        assert tf.reduce_all(tf.math.is_finite(gmm_samples))

    def test_linear_transformation_accuracy(self):
        """Test accuracy of linear transformation in GMM"""
        weights = [0.3, 0.7]
        means = [0.0, 2.0]
        stds = [1.0, 1.5]
        target_mean = 10.0
        target_variance = 25.0

        dist = create_gaussian_mixture_distribution(
            weights, means, stds, target_mean, target_variance
        )

        # Generate many samples for accurate statistics
        samples = dist.sample(10000)

        # Calculate empirical statistics
        emp_mean = tf.reduce_mean(samples)
        emp_var = tf.reduce_mean(tf.square(samples - emp_mean))

        # Check that empirical statistics are close to targets
        assert abs(emp_mean - target_mean) < 0.1
        assert abs(emp_var - target_variance) < 0.5

    def test_distribution_properties(self):
        """Test basic properties of created distributions"""
        # Test Johnson SU distribution properties
        gamma, delta, xi, lambda_param = 0.1, 0.8, 0.0, 1.0
        johnson_dist = create_johnson_su_distribution(gamma, delta, xi, lambda_param)

        # Check that distribution has expected methods
        assert hasattr(johnson_dist, "sample")
        assert hasattr(johnson_dist, "mean")
        assert hasattr(johnson_dist, "variance")

        # Test GMM distribution properties
        gmm_dist = create_gaussian_mixture_distribution(
            [0.5, 0.5], [0.0, 2.0], [1.0, 1.0]
        )

        # Check that distribution has expected methods
        assert hasattr(gmm_dist, "sample")
        assert hasattr(gmm_dist, "mean")
        assert hasattr(gmm_dist, "variance")

        # Test that both distributions can compute mean and variance
        johnson_mean = johnson_dist.mean()
        johnson_var = johnson_dist.variance()
        gmm_mean = gmm_dist.mean()
        gmm_var = gmm_dist.variance()

        # Check that means and variances are finite
        assert tf.math.is_finite(johnson_mean)
        assert tf.math.is_finite(johnson_var)
        assert tf.math.is_finite(gmm_mean)
        assert tf.math.is_finite(gmm_var)


if __name__ == "__main__":
    unittest.main()
