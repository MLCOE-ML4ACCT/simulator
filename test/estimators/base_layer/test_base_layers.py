import unittest
import tensorflow as tf
import numpy as np
from unittest.mock import patch

from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.base_layer.multinomial_layer import MultinomialLayer
from estimators.base_layer.tobit_layer import TobitLayer

class TestBaseLayers(unittest.TestCase):

    def test_logistic_layer_shape(self):
        """Test the output shape of the LogisticLayer."""
        layer = LogisticLayer()
        # Input shape: (batch_size, num_features) -> (3, 4)
        test_input = tf.constant(np.random.rand(3, 4), dtype=tf.float32)
        output = layer(test_input)
        # Expected output shape: (batch_size, 1) -> (3, 1)
        self.assertEqual(output.shape, (3, 1))

    def test_logistic_layer_calculation(self):
        """Test the output value of the LogisticLayer with known weights."""
        layer = LogisticLayer()
        test_input = tf.constant([[1.0, 2.0]], dtype=tf.float32) # Shape (1, 2)
        
        # Build layer to create weights
        layer.build(test_input.shape)
        
        # Manually set weights for predictable output
        # w = [[0.5], [1.5]], b = [0.1]
        layer.set_weights([np.array([[0.5], [1.5]]), np.array([0.1])])
        
        output = layer(test_input)
        
        # Expected output: (1.0 * 0.5) + (2.0 * 1.5) + 0.1 = 0.5 + 3.0 + 0.1 = 3.6
        expected_output = tf.constant([[3.6]], dtype=tf.float32)
        
        tf.debugging.assert_near(output, expected_output, rtol=1e-6)

    def test_multinomial_layer_shape(self):
        """Test the output shape of the MultinomialLayer."""
        layer = MultinomialLayer()
        # Input shape: (batch_size, num_features) -> (5, 3)
        test_input = tf.constant(np.random.rand(5, 3), dtype=tf.float32)
        output = layer(test_input)
        # Expected output shape: (batch_size, 2) -> (5, 2)
        self.assertEqual(output.shape, (5, 2))

    def test_multinomial_layer_calculation(self):
        """Test the output value of the MultinomialLayer with known weights."""
        layer = MultinomialLayer()
        test_input = tf.constant([[1.0, 2.0]], dtype=tf.float32) # Shape (1, 2)
        
        layer.build(test_input.shape)
        
        # Manually set weights
        # w = [[0.5], [1.0]], b = [0.1, 0.2]
        layer.set_weights([np.array([[0.5], [1.0]]), np.array([0.1, 0.2])])
        
        output = layer(test_input)
        
        # base_logit = (1.0 * 0.5) + (2.0 * 1.0) = 2.5
        # logit1 = 2.5 + 0.1 = 2.6
        # logit2 = 2.5 + 0.2 = 2.7
        expected_output = tf.constant([[2.6, 2.7]], dtype=tf.float32)
        
        tf.debugging.assert_near(output, expected_output, rtol=1e-6)

    def test_tobit_layer_shape(self):
        """Test the output shape of the TobitLayer."""
        layer = TobitLayer()
        # Input shape: (batch_size, num_features) -> (4, 2)
        test_input = tf.constant(np.random.rand(4, 2), dtype=tf.float32)
        output = layer(test_input)
        # Expected output shape: (batch_size, 1) -> (4, 1)
        self.assertEqual(output.shape, (4, 1))

    @patch('tensorflow.random.uniform')
    def test_tobit_layer_calculation(self, mock_tf_random_uniform):
        """Test the TobitLayer calculation with randomness mocked."""
        # Mock tf.random.uniform to return a constant value for deterministic testing
        # The inverse transform for logistic is scale * log(u / (1-u))
        # If u = 0.5, error term is 0.
        # If u = 0.73105858, log(u/(1-u)) is approx 1.0
        mock_u = 0.73105858
        mock_tf_random_uniform.return_value = tf.constant([[mock_u]], dtype=tf.float32)

        layer = TobitLayer()
        test_input = tf.constant([[1.0, 2.0]], dtype=tf.float32) # Shape (1, 2)
        
        layer.build(test_input.shape)
        
        # Manually set weights
        # w = [[0.5], [1.0]], b = [0.1], scale = [0.5]
        layer.set_weights([
            np.array([[0.5], [1.0]]), 
            np.array([0.1]), 
            np.array([0.5])
        ])
        
        output = layer(test_input)
        
        # deterministic_part = (1.0 * 0.5) + (2.0 * 1.0) + 0.1 = 2.6
        # error_term = scale * log(u / (1-u)) = 0.5 * log(0.731... / 0.268...) = 0.5 * 1.0 = 0.5
        # latent_variable = 2.6 + 0.5 = 3.1
        # output = max(0.0, 3.1) = 3.1
        expected_output = tf.constant([[3.1]], dtype=tf.float32)
        
        tf.debugging.assert_near(output, expected_output, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()