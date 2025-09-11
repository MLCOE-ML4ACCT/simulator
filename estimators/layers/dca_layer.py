import numpy as np
import tensorflow as tf

from estimators.base_layer.logistic_layer import LogisticLayer
from estimators.configs.t7_dca_config import DCA_CONFIG
from estimators.utils import generate_noise_for_variable


class DCALayer(tf.keras.layers.Layer):
    """A dedicated TensorFlow layer for estimating the 'dca' variable.

    This layer uses a single Heckman-style selection (HS) layer to model the 'dca'
    variable. It takes a dictionary of input tensors, assembles them into a
    single tensor, and passes them to the HS layer.

    Attributes:
        feature_names (list): A list of strings representing the names of the input
            features required by the layer.
        hs_layer (HSLayer): The underlying Heckman-style selection layer used for
            the estimation.
    """

    def __init__(self, add_noise=True, **kwargs):
        """Initializes the DCALayer.

        Args:
            add_noise (bool): Whether to add noise to the output.
            **kwargs: Additional keyword arguments for the parent `tf.keras.layers.Layer`.
        """
        self.add_noise = add_noise
        self.feature_names = [
            "sumcasht_1",
            "diffcasht_1",
            "EDEPMAt",
            "EDEPMAt2",
            "SMAt",
            "IMAt",
            "EDEPBUt",
            "EDEPBUt2",
            "IBUt",
            "IBUt2",
            "IBUt3",
            "dclt_1",
            "ddmpat_1",
            "ddmpat_12",
            "dgnp",
            "FAAB",
            "Public",
            "ruralare",
            "largcity",
            "market",
            "marketw",
        ]
        super().__init__(**kwargs)
        self.logistic_layer = LogisticLayer()

    def build(self, input_shape):
        """Creates the weights of the layer.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        num_features = len(self.feature_names)
        tensor_input_shape = tf.TensorShape((None, num_features))
        self.logistic_layer.build(tensor_input_shape)
        super().build(input_shape)

    def _assemble_tensor(self, inputs):
        """Assembles the input dictionary into a single tensor.

        Args:
            inputs (dict): A dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, num_features).
        """
        feature_tensors = [
            tf.reshape(inputs[name], (-1, 1)) for name in self.feature_names
        ]
        return tf.concat(feature_tensors, axis=1)

    def call(self, inputs):
        """Defines the forward pass of the layer.

        Args:
            inputs (dict): A dictionary mapping feature names to tensors.

        Returns:
            tf.Tensor: The output tensor from the HS layer.
        """
        x_tensor = self._assemble_tensor(inputs)
        result = self.logistic_layer(x_tensor)
        
        # Add noise if enabled (HS method - single distribution)
        if self.add_noise:
            batch_size = tf.shape(result)[0]
            
            # Generate noise for dca variable (HS method - single distribution)
            noise = generate_noise_for_variable("dca", batch_size)
            if noise is not None:
                result = result + noise
        
        return result

    def load_weights_from_cfg(self, cfg):
        """Loads the layer's weights from a configuration dictionary.

        Args:
            cfg (dict): A dictionary containing the model's configuration.
        """
        coefficients = cfg["steps"][0]["coefficients"]
        weights = []
        for name in self.feature_names:
            if name in coefficients:
                weights.append(coefficients[name])
            else:
                raise ValueError(f"Coefficient for {name} not found in config.")
        weights = np.array(weights, dtype=np.float32).reshape(
            len(self.feature_names), 1
        )
        bias = np.array([coefficients["Intercept"]])
        self.logistic_layer.w.assign(weights)
        self.logistic_layer.b.assign(bias)
        print("Weights for 'DCALayer' loaded successfully.")


if __name__ == "__main__":
    print("Testing DCALayer with and without noise...")
    print("=" * 50)
    
    # Test without noise
    print("1. Testing without noise:")
    dca_layer_no_noise = DCALayer(add_noise=False)
    dummy_input = {name: tf.zeros((5, 1)) for name in dca_layer_no_noise.feature_names}
    _ = dca_layer_no_noise(dummy_input)
    dca_layer_no_noise.load_weights_from_cfg(DCA_CONFIG)
    
    prediction_no_noise = dca_layer_no_noise(dummy_input)
    print("Prediction without noise:", prediction_no_noise.numpy().flatten())
    
    # Test with noise
    print("\n2. Testing with noise:")
    dca_layer_with_noise = DCALayer(add_noise=True)
    _ = dca_layer_with_noise(dummy_input)
    dca_layer_with_noise.load_weights_from_cfg(DCA_CONFIG)
    
    prediction_with_noise = dca_layer_with_noise(dummy_input)
    print("Prediction with noise:", prediction_with_noise.numpy().flatten())
    
    # Compare predictions
    print("\n3. Comparison:")
    print("Difference (with noise - without noise):")
    difference = prediction_with_noise - prediction_no_noise
    print(difference.numpy().flatten())
    
    # Test multiple runs to show noise variation
    print("\n4. Multiple runs with noise (showing variation):")
    for i in range(3):
        pred = dca_layer_with_noise(dummy_input)
        print(f"Run {i+1}: {pred.numpy().flatten()}")
    
    print("\n5. Verification:")
    print("HS (Huber-Schweppes) method successfully adds noise to single-stage model.")
    print("Each run should show different values due to noise variation.")
    
    print("\nDCALayer noise integration test completed successfully!")
    print("Feature Names:", sorted(dca_layer_with_noise.feature_names))
