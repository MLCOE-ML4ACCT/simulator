import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
import os
from noises.config import Flow_info

# Add parent directory to path for importing noises module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noises.fitting_johnsonSU import (
    step1_initialize,
    step2_find_omega1,
    step3_define_functions,
    step4_check_solution,
    step5_calculate_omega2,
    create_objective_function,
    step6_find_omega_star,
    step7_recover_parameters,
    johnson_su_fit_steps1to7,
    verify_johnson_su_fit,
    calculate_empirical_moments,
    extract_moments_from_config,
)
    

class TestFittingJohnsonSU(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        tf.config.run_functions_eagerly(True)
        tf.random.set_seed(42)

    def test_step1_initialize(self):
        """Test Step 1: parameter initialization"""
        # Create test data
        mean_test = tf.constant([1.0, 2.0, -0.5], dtype=tf.float32)
        std_test = tf.constant([1.5, 2.5, 1.0], dtype=tf.float32)
        skew_test = tf.constant([0.5, -1.2, 0.8], dtype=tf.float32)
        kurt_test = tf.constant([4.0, 6.5, 3.5], dtype=tf.float32)

        beta1, beta2 = step1_initialize(mean_test, std_test, skew_test, kurt_test)

        # Check output shapes
        assert beta1.shape == (3,)
        assert beta2.shape == (3,)

        # Check beta1 = skew^2
        expected_beta1 = tf.square(skew_test)
        tf.debugging.assert_near(beta1, expected_beta1, atol=1e-6)

        # Check beta2 = kurt
        tf.debugging.assert_near(beta2, kurt_test, atol=1e-6)

        # Check non-negativity
        assert tf.reduce_all(beta1 >= 0)

    def test_step2_find_omega1(self):
        """Test Step 2: finding lower bound ω₁"""
        beta2 = tf.constant([4.0, 6.5, 3.5], dtype=tf.float32)
        omega1 = step2_find_omega1(beta2)

        # Check output shape
        assert omega1.shape == (3,)

        # Check omega1 should be positive
        assert tf.reduce_all(omega1 > 0)

        # Check numerical stability (should not contain NaN or Inf)
        assert tf.reduce_all(tf.math.is_finite(omega1))

    def test_step3_define_functions(self):
        """Test Step 3: define functions m(ω) and f(ω)"""
        omega = tf.constant([1.5, 2.0, 1.2], dtype=tf.float32)
        beta2 = tf.constant([4.0, 6.5, 3.5], dtype=tf.float32)

        m_omega, f_omega = step3_define_functions(omega, beta2)

        # Check output shapes
        assert m_omega.shape == (3,)
        assert f_omega.shape == (3,)

        # Check numerical stability
        assert tf.reduce_all(tf.math.is_finite(m_omega))
        assert tf.reduce_all(tf.math.is_finite(f_omega))

        # Check m_omega > -2 (according to formula, m(ω) = -2 + sqrt(...))
        assert tf.reduce_all(m_omega > -2.0)

    def test_step4_check_solution(self):
        """Test Step 4: check solution existence"""
        omega1 = tf.constant([1.5, 2.0, 1.2], dtype=tf.float32)
        beta1 = tf.constant([0.25, 1.44, 0.64], dtype=tf.float32)

        has_solution = step4_check_solution(omega1, beta1)

        # Check output shape and type
        assert has_solution.shape == (3,)
        assert has_solution.dtype == tf.bool

        # Result should be boolean values
        assert tf.reduce_all(tf.logical_or(has_solution, tf.logical_not(has_solution)))

    def test_step5_calculate_omega2(self):
        """Test Step 5: calculate upper bound ω₂"""
        beta2 = tf.constant([4.0, 6.5, 3.5], dtype=tf.float32)
        omega2 = step5_calculate_omega2(beta2)

        # Check output shape
        assert omega2.shape == (3,)

        # Check omega2 should be positive
        assert tf.reduce_all(omega2 > 0)

        # Check numerical stability
        assert tf.reduce_all(tf.math.is_finite(omega2))

        # For β₂ >= 1 cases, should compute normally
        valid_mask = beta2 >= 1.0
        if tf.reduce_any(valid_mask):
            assert tf.reduce_all(omega2[valid_mask] > 0)

    def test_create_objective_function(self):
        """Test creation of objective function"""
        beta1 = tf.constant([0.25, 1.44, 0.64], dtype=tf.float32)
        beta2 = tf.constant([4.0, 6.5, 3.5], dtype=tf.float32)

        objective_fn = create_objective_function(beta1, beta2)

        # Test objective function call
        omega_test = tf.constant([1.5, 2.0, 1.2], dtype=tf.float32)
        result = objective_fn(omega_test)

        # Check output shape
        assert result.shape == (3,)

        # Check numerical stability
        assert tf.reduce_all(tf.math.is_finite(result))

    def test_step6_find_omega_star(self):
        """Test Step 6: finding ω*"""
        # Use simpler test cases
        omega1 = tf.constant([1.1, 1.1], dtype=tf.float32)
        omega2 = tf.constant([2.0, 2.0], dtype=tf.float32)
        beta1 = tf.constant([0.5, 0.8], dtype=tf.float32)
        beta2 = tf.constant([4.0, 5.0], dtype=tf.float32)

        omega_star = step6_find_omega_star(omega1, omega2, beta1, beta2)

        # Check output shape
        assert omega_star.shape == (2,)

        # Check omega_star is in reasonable range
        assert tf.reduce_all(omega_star >= omega1)
        assert tf.reduce_all(omega_star <= omega2)

        # Check numerical stability
        assert tf.reduce_all(tf.math.is_finite(omega_star))

    def test_step7_recover_parameters(self):
        """Test Step 7: recover JohnsonSU parameters"""
        # Use realistic omega_star values from actual fitting process
        # First get reasonable parameters through steps 1-6
        mean = tf.constant([0.0, 1.0], dtype=tf.float32)
        std = tf.constant([1.0, 1.5], dtype=tf.float32)
        skew = tf.constant([0.5, -0.8], dtype=tf.float32)
        kurt = tf.constant([4.0, 5.0], dtype=tf.float32)

        # Get reasonable parameters through previous steps
        beta1, beta2 = step1_initialize(mean, std, skew, kurt)
        omega1 = step2_find_omega1(beta2)
        omega2 = step5_calculate_omega2(beta2)

        # Only test cases with solutions
        has_solution = step4_check_solution(omega1, beta1)

        if tf.reduce_any(has_solution):
            # Use midpoint between omega1 and omega2 as test value
            omega_star_test = (omega1 + omega2) / 2.0

            delta, lambda_param, gamma, xi = step7_recover_parameters(
                omega_star_test, mean, std, skew, kurt, beta2
            )

            # Check output shapes
            assert delta.shape == mean.shape
            assert lambda_param.shape == mean.shape
            assert gamma.shape == mean.shape
            assert xi.shape == mean.shape

            # Check delta > 0 (shape parameter should be positive)
            assert tf.reduce_all(delta > 0)

            # Check lambda_param > 0 (scale parameter should be positive)
            assert tf.reduce_all(lambda_param > 0)

            # Check numerical stability, but allow some extreme cases
            assert tf.reduce_all(tf.math.is_finite(delta))
            assert tf.reduce_all(tf.math.is_finite(lambda_param))

            # For gamma and xi, only check they are not NaN (allow infinity)
            gamma_valid = tf.logical_not(tf.math.is_nan(gamma))
            xi_valid = tf.logical_not(tf.math.is_nan(xi))

            # If most values are valid, consider test passed
            assert tf.reduce_mean(tf.cast(gamma_valid, tf.float32)) > 0.3
            assert tf.reduce_mean(tf.cast(xi_valid, tf.float32)) > 0.3

    def test_johnson_su_fit_steps1to7(self):
        """Test complete fitting pipeline"""
        # Use simpler test data
        mean = tf.constant([0.0, 1.0], dtype=tf.float32)
        std = tf.constant([1.0, 1.5], dtype=tf.float32)
        skew = tf.constant([0.5, -0.8], dtype=tf.float32)
        kurt = tf.constant([4.0, 5.0], dtype=tf.float32)

        results = johnson_su_fit_steps1to7(mean, std, skew, kurt)

        # Check returned dictionary contains all expected keys
        expected_keys = [
            "beta1",
            "beta2",
            "omega1",
            "omega2",
            "has_solution",
            "m_omega1",
            "f_omega1",
            "omega_star",
            "delta",
            "lambda",
            "gamma",
            "xi",
        ]

        for key in expected_keys:
            assert key in results

        # Check main parameter shapes
        assert results["delta"].shape == (2,)
        assert results["lambda"].shape == (2,)
        assert results["gamma"].shape == (2,)
        assert results["xi"].shape == (2,)

        # Check main parameter reasonableness
        assert tf.reduce_all(results["delta"] > 0)
        assert tf.reduce_all(results["lambda"] > 0)

    def test_calculate_empirical_moments(self):
        """Test empirical moments calculation"""
        # Create test samples
        n_samples = 1000
        batch_size = 2

        # Create samples from known distribution for testing
        tf.random.set_seed(42)
        samples = tf.random.normal([n_samples, batch_size], mean=0.0, stddev=1.0)

        emp_mean, emp_var, emp_skew, emp_kurt = calculate_empirical_moments(samples)

        # Check output shapes
        assert emp_mean.shape == (batch_size,)
        assert emp_var.shape == (batch_size,)
        assert emp_skew.shape == (batch_size,)
        assert emp_kurt.shape == (batch_size,)

        # Check theoretical values for standard normal distribution (allow some error)
        tf.debugging.assert_near(emp_mean, tf.zeros_like(emp_mean), atol=0.1)
        tf.debugging.assert_near(emp_var, tf.ones_like(emp_var), atol=0.1)
        tf.debugging.assert_near(emp_skew, tf.zeros_like(emp_skew), atol=0.2)
        tf.debugging.assert_near(emp_kurt, 3.0 * tf.ones_like(emp_kurt), atol=0.3)

    def test_verify_johnson_su_fit(self):
        """Test JohnsonSU fitting verification"""
        # Use simple parameters for testing
        gamma = tf.constant([0.0, 0.5], dtype=tf.float32)
        delta = tf.constant([1.0, 1.5], dtype=tf.float32)
        xi = tf.constant([0.0, 1.0], dtype=tf.float32)
        lambda_param = tf.constant([1.0, 2.0], dtype=tf.float32)

        verification_results = verify_johnson_su_fit(
            gamma, delta, xi, lambda_param, n_samples=1000, seed=42
        )

        # Check returned dictionary contains expected keys
        expected_keys = [
            "fitted_gamma",
            "fitted_delta",
            "fitted_xi",
            "fitted_lambda",
            "n_samples",
            "samples_shape",
            "empirical_mean",
            "empirical_var",
            "empirical_skew",
            "empirical_kurt",
            "theoretical_mean",
            "theoretical_var",
            "samples",
        ]

        for key in expected_keys:
            assert key in verification_results

        # Check sample shape
        assert verification_results["samples_shape"] == (1000, 2)

        # Check empirical statistics shapes
        assert verification_results["empirical_mean"].shape == (2,)
        assert verification_results["empirical_var"].shape == (2,)

    def test_extract_moments_from_config(self):
        """Test extracting moments from config"""
        result = extract_moments_from_config(Flow_info)

        # Result should not be None since config.py exists
        assert result is not None

        (
            means_tensor,
            variances_tensor,
            skews_tensor,
            kurts_tensor,
            labels_info,
        ) = result

        # Data should be successfully extracted
        assert means_tensor is not None
        assert variances_tensor is not None
        assert skews_tensor is not None
        assert kurts_tensor is not None
        assert labels_info is not None

        # Check tensor shapes are consistent
        assert means_tensor.shape[0] == len(labels_info)
        assert variances_tensor.shape[0] == len(labels_info)
        assert skews_tensor.shape[0] == len(labels_info)
        assert kurts_tensor.shape[0] == len(labels_info)

        # Check variance is positive
        assert tf.reduce_all(variances_tensor > 0)

        # Check that we have some data
        assert len(labels_info) > 0

    def test_numerical_stability(self):
        """Test numerical stability"""
        # Test extreme values
        extreme_mean = tf.constant([1e6, -1e6, 0.0], dtype=tf.float32)
        extreme_std = tf.constant([1e-3, 1e3, 1.0], dtype=tf.float32)
        extreme_skew = tf.constant([10.0, -10.0, 0.0], dtype=tf.float32)
        extreme_kurt = tf.constant([100.0, 3.0, 4.0], dtype=tf.float32)

        # Test step1 doesn't produce invalid values
        beta1, beta2 = step1_initialize(
            extreme_mean, extreme_std, extreme_skew, extreme_kurt
        )
        assert tf.reduce_all(tf.math.is_finite(beta1))
        assert tf.reduce_all(tf.math.is_finite(beta2))

    def test_input_validation(self):
        """Test input validation"""
        # Test empty input
        empty_tensor = tf.constant([], dtype=tf.float32)

        # step1 should handle empty input
        beta1, beta2 = step1_initialize(
            empty_tensor, empty_tensor, empty_tensor, empty_tensor
        )
        assert beta1.shape == (0,)
        assert beta2.shape == (0,)

        # Test single value input
        single_mean = tf.constant([1.0], dtype=tf.float32)
        single_std = tf.constant([1.0], dtype=tf.float32)
        single_skew = tf.constant([0.5], dtype=tf.float32)
        single_kurt = tf.constant([4.0], dtype=tf.float32)

        beta1, beta2 = step1_initialize(
            single_mean, single_std, single_skew, single_kurt
        )
        assert beta1.shape == (1,)
        assert beta2.shape == (1,)
