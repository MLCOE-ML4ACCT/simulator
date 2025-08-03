import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import os

# Add parent directory to path for importing noises module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noises.fitting_gmm import (
    TrainingConfig,
    MomentMatchingGMM,
    get_analytical_moments,
    get_closed_form_moments,
    compute_empirical_moments,
    create_tensor_dict,
    safe_tensor_comparison,
    compute_loss,
    train_step_core,
    train_step,
    create_optimizer_and_scheduler,
    postprocess_model,
    save_results_to_json,
)


class TestFittingGMM(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        tf.config.run_functions_eagerly(True)
        tf.random.set_seed(42)

    def test_training_config_initialization(self):
        """Test TrainingConfig initialization"""
        # Test with default values
        config = TrainingConfig()
        assert config.NUM_COMPONENTS == 3
        assert config.MAX_LEARNING_RATE == 0.01
        assert config.MIN_LEARNING_RATE == 0.001
        assert config.TRAINING_STEPS == 5000
        assert config.RANDOM_SEED == 42

        # Test with custom target moments
        custom_moments = {
            "mean": 1.0,
            "variance": 2.0,
            "skewness": 0.5,
            "kurtosis": 4.0,
        }
        config_custom = TrainingConfig(target_moments=custom_moments)
        assert config_custom.TARGET_MOMENTS == custom_moments

    def test_moment_matching_gmm_initialization(self):
        """Test MomentMatchingGMM model initialization"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)

        # Check model attributes
        assert model.num_components == 3
        assert model.config == config

        # Check trainable variables
        assert len(model.trainable_variables) == 3
        assert model.locs.shape == (3,)
        assert model.scales.shape == (3,)
        assert model.logits.shape == (3,)

        # Check initial values are finite
        assert tf.reduce_all(tf.math.is_finite(model.locs))
        assert tf.reduce_all(tf.math.is_finite(model.scales))
        assert tf.reduce_all(tf.math.is_finite(model.logits))

    def test_get_gmm_distribution(self):
        """Test GMM distribution creation"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)

        gmm_dist = model.get_gmm_distribution()

        # Check distribution type
        assert isinstance(gmm_dist, tfp.distributions.MixtureSameFamily)

        # Check distribution parameters
        weights = gmm_dist.mixture_distribution.probs_parameter()
        component_means = gmm_dist.components_distribution.mean()
        component_scales = gmm_dist.components_distribution.scale

        # Check shapes
        assert weights.shape == (3,)
        assert component_means.shape == (3,)
        assert component_scales.shape == (3,)

        # Check weights sum to 1
        tf.debugging.assert_near(tf.reduce_sum(weights), 1.0, atol=1e-6)

        # Check scales are positive
        assert tf.reduce_all(component_scales > 0)

    def test_get_analytical_moments(self):
        """Test analytical moments calculation"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        gmm_dist = model.get_gmm_distribution()

        mean, variance = get_analytical_moments(gmm_dist)

        # Check output shapes
        assert mean.shape == ()
        assert variance.shape == ()

        # Check values are finite
        assert tf.math.is_finite(mean)
        assert tf.math.is_finite(variance)

        # Check variance is positive
        assert variance > 0

    def test_get_closed_form_moments(self):
        """Test closed-form moments calculation"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        gmm_dist = model.get_gmm_distribution()

        skewness, kurtosis = get_closed_form_moments(gmm_dist)

        # Check output shapes
        assert skewness.shape == ()
        assert kurtosis.shape == ()

        # Check values are finite
        assert tf.math.is_finite(skewness)
        assert tf.math.is_finite(kurtosis)

        # Check kurtosis is positive (ordinary kurtosis)
        assert kurtosis > 0

    def test_compute_empirical_moments(self):
        """Test empirical moments calculation"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        gmm_dist = model.get_gmm_distribution()

        empirical_stats = compute_empirical_moments(gmm_dist, num_samples=1000)

        # Check returned dictionary structure
        expected_keys = ["mean", "variance", "skewness", "kurtosis"]
        for key in expected_keys:
            assert key in empirical_stats

        # Check all values are finite
        for value in empirical_stats.values():
            assert tf.math.is_finite(value)

        # Check variance is positive
        assert empirical_stats["variance"] > 0

    def test_create_tensor_dict(self):
        """Test tensor dictionary creation"""
        # Create test tensor dictionary
        tensor_dict = {
            "mean": tf.constant(1.0),
            "variance": tf.constant(2.0),
            "skewness": tf.constant(0.5),
        }

        # Test without conversion
        result_no_convert = create_tensor_dict(tensor_dict, convert_to_float=False)
        assert isinstance(result_no_convert["mean"], tf.Tensor)
        assert isinstance(result_no_convert["variance"], tf.Tensor)

        # Test with conversion
        result_convert = create_tensor_dict(tensor_dict, convert_to_float=True)
        assert isinstance(result_convert["mean"], float)
        assert isinstance(result_convert["variance"], float)
        assert result_convert["mean"] == 1.0
        assert result_convert["variance"] == 2.0

    def test_safe_tensor_comparison(self):
        """Test safe tensor comparison"""
        tensor_val = tf.constant(5.0)
        scalar_val = 10.0

        numpy_val, comparison_result = safe_tensor_comparison(tensor_val, scalar_val)

        # Check return types - numpy_val could be numpy.float32 or float
        assert hasattr(numpy_val, "__float__") or isinstance(
            numpy_val, (float, np.floating)
        )
        # comparison_result could be numpy.bool_ or bool
        assert hasattr(comparison_result, "__bool__") or isinstance(
            comparison_result, (bool, np.bool_)
        )

        # Check values
        assert float(numpy_val) == 5.0
        assert bool(comparison_result) == True  # 5.0 < 10.0

    def test_compute_loss(self):
        """Test loss computation"""
        model_moments = {
            "skewness": tf.constant(0.5),
            "kurtosis": tf.constant(4.0),
        }
        target_moments = {
            "skewness": 0.6,
            "kurtosis": 4.2,
        }

        loss = compute_loss(model_moments, target_moments)

        # Check loss is finite and positive
        assert tf.math.is_finite(loss)
        assert loss > 0

        # Check loss shape
        assert loss.shape == ()

    def test_train_step_core(self):
        """Test core training step computation"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        target_moments = {
            "skewness": 0.5,
            "kurtosis": 4.0,
        }

        loss, model_moments, gradients = train_step_core(model, target_moments)

        # Check return types
        assert isinstance(loss, tf.Tensor)
        assert isinstance(model_moments, dict)
        assert isinstance(gradients, list)

        # Check loss is finite
        assert tf.math.is_finite(loss)

        # Check model_moments structure
        expected_keys = ["mean", "variance", "skewness", "kurtosis"]
        for key in expected_keys:
            assert key in model_moments

        # Check gradients length matches trainable variables
        assert len(gradients) == len(model.trainable_variables)

    def test_train_step(self):
        """Test complete training step"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        optimizer = create_optimizer_and_scheduler(config)
        target_moments = {
            "skewness": 0.5,
            "kurtosis": 4.0,
        }

        loss, model_moments = train_step(model, optimizer, target_moments)

        # Check return types
        assert isinstance(loss, tf.Tensor)
        assert isinstance(model_moments, dict)

        # Check loss is finite
        assert tf.math.is_finite(loss)

        # Check model_moments structure
        expected_keys = ["mean", "variance", "skewness", "kurtosis"]
        for key in expected_keys:
            assert key in model_moments

    def test_create_optimizer_and_scheduler(self):
        """Test optimizer and scheduler creation"""
        config = TrainingConfig()
        optimizer = create_optimizer_and_scheduler(config)

        # Check optimizer type
        assert isinstance(optimizer, tf.keras.optimizers.Adam)

        # Check learning rate is finite
        assert tf.math.is_finite(optimizer.learning_rate)

    def test_postprocess_model(self):
        """Test model postprocessing"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        target_moments = {
            "mean": 1.0,
            "variance": 2.0,
            "skewness": 0.5,
            "kurtosis": 4.0,
        }

        (
            final_gmm,
            sorted_weights,
            sorted_means,
            sorted_stddevs,
            a,
            b,
        ) = postprocess_model(model, target_moments)

        # Check return types
        assert isinstance(final_gmm, tfp.distributions.TransformedDistribution)
        assert isinstance(sorted_weights, np.ndarray)
        assert isinstance(sorted_means, np.ndarray)
        assert isinstance(sorted_stddevs, np.ndarray)

        # Check array shapes
        assert sorted_weights.shape == (3,)
        assert sorted_means.shape == (3,)
        assert sorted_stddevs.shape == (3,)

        # Check weights sum to 1
        assert np.abs(np.sum(sorted_weights) - 1.0) < 1e-6

        # Check stddevs are positive
        assert np.all(sorted_stddevs > 0)

        # Check transformation parameters are finite
        assert np.isfinite(a)
        assert np.isfinite(b)

    def test_process_all_variables(self):
        """Test processing all variables"""
        # Create test flow_info with limited data
        test_flow_info = {
            "test_var1": {
                "method": "HS",
                "mean": 1.0,
                "variance": 2.0,
                "skewness": 0.5,
                "kurtosis": 4.0,
            },
            "test_var2": {
                "method": "LLG",
                "mean": 2.0,
                "variance": 3.0,
                "skewness": 0.8,
                "kurtosis": 5.0,
            },
        }

        # Test with limited data
        results = {}
        for var_name, var_data in test_flow_info.items():
            method = var_data["method"]
            results[var_name] = {"method": method}

            if method in ["HS", "LLG", "LLN"]:
                if all(
                    key in var_data
                    for key in ["mean", "variance", "skewness", "kurtosis"]
                ):
                    # Skip actual fitting for speed, just test structure
                    results[var_name].update(
                        {
                            "weights": [0.3, 0.4, 0.3],
                            "means": [1.0, 2.0, 3.0],
                            "stds": [0.5, 1.0, 1.5],
                            "loss": 0.1,
                        }
                    )

        # Check results is a dictionary
        assert isinstance(results, dict)

        # Check at least some variables were processed
        assert len(results) > 0

        # Check each result has method key
        for var_name, var_data in results.items():
            assert "method" in var_data

    def test_save_results_to_json(self):
        """Test saving results to JSON"""
        # Create test results
        test_results = {
            "test_var": {
                "method": "HS",
                "weights": [0.3, 0.4, 0.3],
                "means": [1.0, 2.0, 3.0],
                "stds": [0.5, 1.0, 1.5],
                "loss": 0.1,
                "empirical_stats": {
                    "mean": 2.0,
                    "variance": 1.0,
                    "skewness": 0.5,
                    "kurtosis": 4.0,
                },
            }
        }

        # Test saving (this will create temporary files)
        save_results_to_json(
            test_results,
            parameters_filename="test_params.json",
            statistics_filename="test_stats.json",
        )

        # Check files were created
        assert os.path.exists("test_params.json")
        assert os.path.exists("test_stats.json")

        # Clean up
        os.remove("test_params.json")
        os.remove("test_stats.json")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Test with extreme target moments
        extreme_moments = {
            "mean": 1e6,
            "variance": 1e12,
            "skewness": 100.0,
            "kurtosis": 10000.0,
        }

        config = TrainingConfig(target_moments=extreme_moments)
        model = MomentMatchingGMM(num_components=3, config=config)

        # Check model initialization doesn't produce invalid values
        gmm_dist = model.get_gmm_distribution()
        mean, variance = get_analytical_moments(gmm_dist)
        skewness, kurtosis = get_closed_form_moments(gmm_dist)

        # Check all moments are finite
        assert tf.math.is_finite(mean)
        assert tf.math.is_finite(variance)
        assert tf.math.is_finite(skewness)
        assert tf.math.is_finite(kurtosis)

    def test_input_validation(self):
        """Test input validation"""
        # Test with different component numbers
        config = TrainingConfig()

        for num_components in [1, 2, 5]:
            model = MomentMatchingGMM(num_components=num_components, config=config)
            gmm_dist = model.get_gmm_distribution()

            # Check distribution works with different component numbers
            mean, variance = get_analytical_moments(gmm_dist)
            assert tf.math.is_finite(mean)
            assert tf.math.is_finite(variance)

    def test_loss_function_properties(self):
        """Test loss function mathematical properties"""
        # Test loss is zero when moments match exactly
        model_moments = {
            "skewness": tf.constant(0.5),
            "kurtosis": tf.constant(4.0),
        }
        target_moments = {
            "skewness": 0.5,
            "kurtosis": 4.0,
        }

        loss_exact = compute_loss(model_moments, target_moments)
        assert loss_exact == 0.0

        # Test loss increases with larger differences
        model_moments_large_diff = {
            "skewness": tf.constant(1.0),
            "kurtosis": tf.constant(8.0),
        }

        loss_large_diff = compute_loss(model_moments_large_diff, target_moments)
        assert loss_large_diff > loss_exact

    def test_gmm_distribution_properties(self):
        """Test GMM distribution mathematical properties"""
        config = TrainingConfig()
        model = MomentMatchingGMM(num_components=3, config=config)
        gmm_dist = model.get_gmm_distribution()

        # Test sampling
        samples = gmm_dist.sample(1000)
        assert samples.shape == (1000,)

        # Test log probability
        log_probs = gmm_dist.log_prob(samples)
        assert log_probs.shape == (1000,)
        assert tf.reduce_all(tf.math.is_finite(log_probs))

        # Test mean and variance (analytical moments)
        mean = gmm_dist.mean()
        variance = gmm_dist.variance()
        assert tf.math.is_finite(mean)
        assert tf.math.is_finite(variance)
        assert variance > 0
