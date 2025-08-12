import numpy as np
import tensorflow as tf


class BinaryCLogLogIRLSGPU(tf.keras.Model):
    """
    TensorFlow model for Binary Classification with Complementary Log-Log link using IRLS

    This version is optimized for GPU execution with better memory management
    to avoid OOM errors while maintaining mathematical correctness.
    """

    def __init__(
        self,
        n_features,
        max_iterations=25,
        tolerance=1e-6,
        patience=5,
        regularization=1e-6,
        **kwargs,
    ):
        """
        Initialize the Binary CLogLog IRLS model

        Args:
            n_features: Number of input features
            max_iterations: Maximum IRLS iterations
            tolerance: Convergence tolerance for training weights
            patience: Early stopping patience
            regularization: Ridge regularization for numerical stability
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.regularization = regularization

        # Model weights (including bias term)
        self.W = None

        # Track training and validation metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy_tracker = tf.keras.metrics.BinaryAccuracy(
            name="train_accuracy"
        )
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

    def build(self, input_shape):
        """Build the layer with the given input shape"""
        # Initialize weights: bias + features
        self.W = self.add_weight(
            name="weights",
            shape=(self.n_features + 1, 1),  # +1 for bias term
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    @tf.function
    def _cloglog_loss(self, y_true, y_pred_eta, epsilon=1e-8):
        """Helper function to calculate the cloglog negative log-likelihood."""
        y_pred_eta = tf.clip_by_value(y_pred_eta, -30.0, 30.0)
        p = 1.0 - tf.exp(-tf.exp(y_pred_eta))
        log_likelihood = y_true * tf.math.log(p + epsilon) + (1.0 - y_true) * (
            -tf.exp(y_pred_eta)
        )
        return -tf.reduce_mean(log_likelihood)

    @tf.function
    def _add_bias_term(self, X):
        """Add bias term (column of ones) to feature matrix"""
        return tf.concat([tf.ones((tf.shape(X)[0], 1), dtype=X.dtype), X], axis=1)

    @tf.function
    def call(self, inputs, training=None):
        """
        Forward pass of the model

        Args:
            inputs: Input tensor (batch_size, n_features)
            training: Whether in training mode

        Returns:
            probabilities: Predicted probabilities (batch_size, 1)
        """
        # Add bias term
        X_with_bias = self._add_bias_term(inputs)

        # Compute linear predictor
        eta = tf.matmul(X_with_bias, self.W)
        eta = tf.clip_by_value(eta, -10.0, 10.0)

        # Compute probabilities using cloglog link
        p = 1.0 - tf.exp(-tf.exp(eta))

        return p

    @tf.function
    def _compute_loss(self, X, y):
        """
        Compute negative log-likelihood loss

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size, 1)

        Returns:
            loss: Negative log-likelihood
        """
        X_with_bias = self._add_bias_term(X)
        eta = tf.matmul(X_with_bias, self.W)
        return self._cloglog_loss(y, eta)

    @tf.function
    def _irls_step_memory_efficient(self, X, y):
        """
        Memory-efficient IRLS iteration step for GPU execution

        This version avoids creating large diagonal matrices and uses
        efficient vectorized operations instead.

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size, 1)

        Returns:
            loss: Current loss value
            weight_change: Change in weights
        """
        # Store old weights for change calculation
        W_old = tf.identity(self.W)

        # Add bias term to features
        X_with_bias = self._add_bias_term(X)

        # Calculate components
        eta = tf.matmul(X_with_bias, self.W)
        eta = tf.clip_by_value(eta, -10.0, 10.0)

        mu = 1.0 - tf.exp(-tf.exp(eta))
        d_mu_d_eta = tf.exp(eta) * tf.exp(-tf.exp(eta))
        var_y = mu * (1.0 - mu)

        epsilon = 1e-8
        s_diag = (d_mu_d_eta**2) / (var_y + epsilon)
        s_diag = tf.squeeze(s_diag)

        # Memory-efficient weighted operations
        # Instead of creating full diagonal matrix S, use element-wise operations
        sqrt_s = tf.sqrt(s_diag)
        X_weighted = X_with_bias * tf.expand_dims(sqrt_s, axis=1)

        # Equivalent to X^T @ S @ X but more memory efficient
        XT_S_X = tf.matmul(tf.transpose(X_weighted), X_weighted)

        # Calculate z vector
        z = eta + (y - mu) / (d_mu_d_eta + epsilon)
        z = tf.squeeze(z)

        # Weighted z: equivalent to X^T @ S @ z
        s_times_z = s_diag * z
        XT_S_z = tf.reduce_sum(X_with_bias * tf.expand_dims(s_times_z, axis=1), axis=0)
        XT_S_z = tf.expand_dims(XT_S_z, axis=1)

        # Add regularization for numerical stability
        n_features = tf.shape(XT_S_X)[0]
        identity_matrix = tf.eye(n_features, dtype=tf.float32)
        XT_S_X_reg = XT_S_X + self.regularization * identity_matrix

        # Solve the system with better numerical stability
        try:
            # Use Cholesky decomposition for better numerical stability
            L = tf.linalg.cholesky(XT_S_X_reg)
            y_solve = tf.linalg.triangular_solve(L, XT_S_z, lower=True)
            new_W = tf.linalg.triangular_solve(tf.transpose(L), y_solve, lower=False)
        except tf.errors.InvalidArgumentError:
            # Fallback to regular inverse if Cholesky fails
            try:
                XT_S_X_inv = tf.linalg.inv(XT_S_X_reg)
                new_W = tf.matmul(XT_S_X_inv, XT_S_z)
            except tf.errors.InvalidArgumentError:
                # Final fallback to pseudo-inverse
                new_W = tf.linalg.pinv(XT_S_X_reg) @ XT_S_z

        # Update weights
        self.W.assign(new_W)

        # Calculate loss and weight change
        loss = self._compute_loss(X, y)
        weight_change = tf.reduce_sum(tf.abs(self.W - W_old))

        return loss, weight_change

    def fit(self, X, y, validation_data=None, verbose=1):
        """
        Fit the model using memory-efficient IRLS algorithm with early stopping

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples, 1) or (n_samples,)
            validation_data: Optional validation data tuple (X_val, y_val)
            verbose: Verbosity level

        Returns:
            self: Fitted model
        """
        # Convert to tensors and ensure proper shapes
        if not isinstance(X, tf.Tensor):
            X = tf.constant(X, dtype=tf.float32)
        if not isinstance(y, tf.Tensor):
            y = tf.constant(y, dtype=tf.float32)

        # Ensure y is column vector
        if len(y.shape) == 1:
            y = tf.expand_dims(y, axis=1)

        # Build the model if not already built
        if not self.built:
            self.build(X.shape)

        if verbose:
            print("Starting CLogLog MLE Estimation via Memory-Efficient IRLS...")
            print("-" * 60)

        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_data is not None:
            X_val, y_val = validation_data
            if not isinstance(X_val, tf.Tensor):
                X_val = tf.constant(X_val, dtype=tf.float32)
            if not isinstance(y_val, tf.Tensor):
                y_val = tf.constant(y_val, dtype=tf.float32)
            if len(y_val.shape) == 1:
                y_val = tf.expand_dims(y_val, axis=1)

        # Variables for early stopping
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_weights = tf.Variable(tf.zeros_like(self.W))

        # Training loop
        for iteration in range(self.max_iterations):
            loss, weight_change = self._irls_step_memory_efficient(X, y)

            # Update training metrics
            self.train_loss_tracker.update_state(loss)
            predictions = self.call(X)
            self.train_accuracy_tracker.update_state(y, predictions)

            # Validation evaluation and early stopping
            if validation_data is not None:
                val_loss = self._compute_loss(X_val, y_val)
                val_predictions = self.call(X_val)

                self.val_loss_tracker.update_state(val_loss)
                self.val_accuracy_tracker.update_state(y_val, val_predictions)

                val_loss_val = val_loss.numpy()

                # Early stopping logic
                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    best_weights.assign(self.W)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(
                            f"\nEarly stopping: Val loss did not improve for {self.patience} iterations."
                        )
                    # Restore best weights
                    self.W.assign(best_weights)
                    break

                # Print progress with validation metrics
                if verbose:
                    print(
                        f"Iter {iteration + 1:2d}: "
                        f"Change in weights = {weight_change:.8f} | "
                        f"Train Loss = {loss:.6f} | "
                        f"Train Acc = {self.train_accuracy_tracker.result():.4f} | "
                        f"Val Loss = {val_loss_val:.6f} | "
                        f"Val Acc = {self.val_accuracy_tracker.result():.4f}"
                    )
            else:
                # Print progress without validation metrics
                if verbose:
                    print(
                        f"Iter {iteration + 1:2d}: "
                        f"Change in weights = {weight_change:.8f} | "
                        f"Train Loss = {loss:.6f} | "
                        f"Train Acc = {self.train_accuracy_tracker.result():.4f}"
                    )

            # Training convergence check
            if weight_change < self.tolerance:
                if verbose:
                    print(
                        f"\nConvergence on training weights reached after {iteration + 1} iterations."
                    )
                break
        else:
            if verbose:
                print(f"\nMax iterations ({self.max_iterations}) reached.")

        # Print final summary
        if verbose:
            if validation_data is not None:
                print(f"Best validation loss achieved: {best_val_loss:.6f}")
            print(f"\nFinal Training Metrics:")
            print(f"  Loss: {self.train_loss_tracker.result().numpy():.6f}")
            print(f"  Accuracy: {self.train_accuracy_tracker.result().numpy():.4f}")

            if validation_data is not None:
                print(f"Final Validation Metrics:")
                print(f"  Loss: {self.val_loss_tracker.result().numpy():.6f}")
                print(f"  Accuracy: {self.val_accuracy_tracker.result().numpy():.4f}")

            print("-" * 60)

        return self

    def predict(self, X):
        """
        Predict class labels (0 or 1)

        Args:
            X: Input features

        Returns:
            predictions: Predicted class labels
        """
        if not isinstance(X, tf.Tensor):
            X = tf.constant(X, dtype=tf.float32)

        probabilities = self.call(X)
        predictions = tf.cast(probabilities > 0.5, tf.int32)
        return predictions.numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Input features

        Returns:
            probabilities: Class probabilities
        """
        if not isinstance(X, tf.Tensor):
            X = tf.constant(X, dtype=tf.float32)

        return self.call(X).numpy()

    @property
    def metrics(self):
        """Return list of the model's metrics"""
        return [
            self.train_loss_tracker,
            self.train_accuracy_tracker,
            self.val_loss_tracker,
            self.val_accuracy_tracker,
        ]

    def get_coefficients(self):
        """
        Get the fitted coefficients

        Returns:
            dict: Dictionary with 'weights' (including bias)
        """
        return {"weights": self.W.numpy()}

    def get_metrics(self):
        """
        Get current metric values

        Returns:
            dict: Dictionary with current metric values
        """
        metrics_dict = {
            "train_loss": self.train_loss_tracker.result().numpy(),
            "train_accuracy": self.train_accuracy_tracker.result().numpy(),
            "val_loss": self.val_loss_tracker.result().numpy(),
            "val_accuracy": self.val_accuracy_tracker.result().numpy(),
        }
        return metrics_dict

    def get_weights_and_bias(self):
        """
        Get the weights and bias separately

        Returns:
            tuple: (bias, feature_weights)
        """
        if self.W is not None:
            bias = self.W[0, 0].numpy()
            weights = self.W[1:, 0].numpy()
            return bias, weights
        else:
            return None, None
