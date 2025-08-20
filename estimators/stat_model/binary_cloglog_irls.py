import numpy as np
import tensorflow as tf

from estimators.base_layer.logistic_layer import LogisticLayer


class BinaryCLogLogIRLS(tf.keras.Model):
    """
    TensorFlow model for Binary Classification with Complementary Log-Log link using IRLS

    This version closely follows the original cloglog_irls_estimator function approach
    with CPU-only execution to ensure consistency with original results.
    """

    def __init__(
        self,
        max_iterations=25,
        tolerance=1e-6,
        patience=5,
        regularization=1e-6,
        **kwargs,
    ):
        """
        Initialize the Binary CLogLog IRLS model

        Args:
            max_iterations: Maximum IRLS iterations
            tolerance: Convergence tolerance for training weights
            patience: Early stopping patience
            regularization: Ridge regularization for numerical stability
        """
        super().__init__(**kwargs)

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.regularization = regularization
        self.cov_matrix_ = None

        # Use LogisticLayer for weights
        self.logistic_layer = LogisticLayer()

        # Track training and validation metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy_tracker = tf.keras.metrics.BinaryAccuracy(
            name="train_accuracy"
        )
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="val_accuracy")

    def build(self, input_shape):
        """Build the layer with the given input shape"""
        if not self.logistic_layer.built:
            self.logistic_layer.build(input_shape)
        super().build(input_shape)

    def _cloglog_loss(self, y_true, y_pred_eta, epsilon=1e-8):
        """Helper function to calculate the cloglog negative log-likelihood."""
        y_pred_eta = tf.clip_by_value(y_pred_eta, -30.0, 30.0)
        p = 1.0 - tf.exp(-tf.exp(y_pred_eta))
        log_likelihood = y_true * tf.math.log(p + epsilon) + (1.0 - y_true) * (
            -tf.exp(y_pred_eta)
        )
        return -tf.reduce_mean(log_likelihood)

    def _add_bias_term(self, X):
        """Add bias term (column of ones) to feature matrix"""
        return tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)

    def call(self, inputs, training=None):
        """
        Forward pass of the model

        Args:
            inputs: Input tensor (batch_size, n_features)
            training: Whether in training mode

        Returns:
            probabilities: Predicted probabilities (batch_size, 1)
        """
        # Force CPU execution like original function
        with tf.device("/CPU:0"):
            # Compute linear predictor using logistic layer
            eta = self.logistic_layer(inputs)
            eta = tf.clip_by_value(eta, -10.0, 10.0)

            # Compute probabilities using cloglog link
            p = 1.0 - tf.exp(-tf.exp(eta))

            return p

    def _compute_loss(self, X, y):
        """
        Compute negative log-likelihood loss

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size, 1)

        Returns:
            loss: Negative log-likelihood
        """
        with tf.device("/CPU:0"):
            eta = self.logistic_layer(X)
            return self._cloglog_loss(y, eta)

    def _irls_step(self, X, y):
        """
        Single IRLS iteration step - exactly following original algorithm

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size, 1)

        Returns:
            loss: Current loss value
            weight_change: Change in weights
        """
        # Force CPU execution exactly like original
        with tf.device("/CPU:0"):
            # Get combined weights from logistic layer
            W = tf.concat(
                [tf.reshape(self.logistic_layer.b, (1, 1)), self.logistic_layer.w],
                axis=0,
            )
            W_old = tf.identity(W)

            # Add bias term to features
            X_with_bias = self._add_bias_term(X)

            # Calculate components - eta must be calculated with X_with_bias for IRLS
            eta = tf.matmul(X_with_bias, W)
            eta = tf.clip_by_value(eta, -10.0, 10.0)

            mu = 1.0 - tf.exp(-tf.exp(eta))
            d_mu_d_eta = tf.exp(eta) * tf.exp(-tf.exp(eta))
            var_y = mu * (1.0 - mu)

            epsilon = 1e-8
            s_diag = (d_mu_d_eta**2) / (var_y + epsilon)
            S = tf.linalg.diag(tf.squeeze(s_diag))  # Full diagonal matrix like original
            z = eta + (y - mu) / (d_mu_d_eta + epsilon)

            # Solve the Weighted Least Squares Equation - exactly like original
            XT_S = tf.matmul(tf.transpose(X_with_bias), S)
            XT_S_X = tf.matmul(XT_S, X_with_bias)

            # Add regularization for numerical stability
            identity_matrix = tf.eye(tf.shape(XT_S_X)[0], dtype=tf.float32)
            XT_S_X_reg = XT_S_X + self.regularization * identity_matrix

            XT_S_z = tf.matmul(XT_S, z)
            XT_S_X_inv = tf.linalg.inv(XT_S_X_reg)
            self.cov_matrix_ = XT_S_X_inv
            new_W = tf.matmul(XT_S_X_inv, XT_S_z)

            # Update weights in the logistic layer
            self.logistic_layer.b.assign(new_W[0])
            self.logistic_layer.w.assign(new_W[1:])

            # Calculate loss and weight change
            loss = self._compute_loss(X, y)
            weight_change = tf.reduce_sum(tf.abs(new_W - W_old))

            return loss, weight_change

    def fit(self, X, y, validation_data=None, verbose=1):
        """
        Fit the model using IRLS algorithm with early stopping
        Following the exact same logic as the original function

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples, 1) or (n_samples,)
            validation_data: Optional validation data tuple (X_val, y_val)
            verbose: Verbosity level

        Returns:
            self: Fitted model
        """
        # Force CPU execution like original
        with tf.device("/CPU:0"):
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
                print("Starting CLogLog MLE Estimation via IRLS with Early Stopping...")
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
            best_weights = None

            # Training loop - exactly like original
            for iteration in range(self.max_iterations):
                loss, weight_change = self._irls_step(X, y)

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
                        best_weights = self.logistic_layer.get_weights()
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= self.patience:
                        if verbose:
                            print(
                                f"\nEarly stopping: Val loss did not improve for {self.patience} iterations."
                            )
                        # Restore best weights
                        if best_weights is not None:
                            self.logistic_layer.set_weights(best_weights)
                        break

                    # Print progress - exactly like original format
                    if verbose:
                        print(
                            f"Iter {iteration + 1:2d}: Change in weights = {weight_change:.8f} | Val Loss = {val_loss_val:.6f}"
                        )
                else:
                    # Print progress without validation metrics
                    if verbose:
                        print(
                            f"Iter {iteration + 1:2d}: Change in weights = {weight_change:.8f}"
                        )

                # Training convergence check - exactly like original
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
                print("-" * 60)

            return self

    def summary(self, X, y, feature_names=None):
        """
        Computes and returns a summary of the regression results.

        Args:
            X (tf.Tensor or np.ndarray): The input features used for fitting.
            y (tf.Tensor or np.ndarray): The target variable used for fitting.
            feature_names (list of str, optional): Names of the features.

        Returns:
            pd.DataFrame: A DataFrame containing the coefficient summary.
            dict: A dictionary containing the model-level statistics.
        """
        if self.cov_matrix_ is None:
            raise RuntimeError("Model has not been fitted yet, or covariance matrix is not available.")

        import pandas as pd
        from scipy.stats import chi2

        # --- 1. Coefficient Statistics (Wald Test) --- 
        params = tf.concat([self.logistic_layer.b, tf.squeeze(self.logistic_layer.w)], axis=0)
        std_errors = tf.sqrt(tf.linalg.diag_part(self.cov_matrix_))
        
        wald_chi2 = (params / std_errors)**2
        p_values_chi2 = 1 - chi2.cdf(wald_chi2.numpy(), df=1)

        # --- 2. Likelihood-Ratio Test --- 
        log_likelihood_full = -self._compute_loss(X, y).numpy()

        # Log-likelihood of the null model (intercept only)
        y_mean = tf.reduce_mean(y)
        # Avoid log(0) for y_mean = 1
        if y_mean >= 1.0:
            y_mean = 0.999999
        eta_null = tf.math.log(-tf.math.log(1 - y_mean))
        log_likelihood_null = -self._cloglog_loss(y, tf.fill(y.shape, eta_null)).numpy()
        
        lr_statistic = 2 * (log_likelihood_full - log_likelihood_null)
        if feature_names:
            lr_dof = len(feature_names)
        else:
            lr_dof = X.shape[1]
            
        lr_p_value = 1 - chi2.cdf(lr_statistic, df=lr_dof)

        # --- 3. Format Output --- 
        if feature_names is None:
            index_names = ['Intercept'] + [f'feature_{i+1}' for i in range(X.shape[1])]
        else:
            index_names = ['Intercept'] + feature_names
        
        summary_df = pd.DataFrame({
            'Coefficient': params.numpy(),
            'Std. Error': std_errors.numpy(),
            'Chi-square': wald_chi2.numpy(),
            'Pr(>ChiSq)': p_values_chi2
        }, index=index_names)

        model_stats = {
            'Log-Likelihood': float(log_likelihood_full),
            'LL-Null': float(log_likelihood_null),
            'LR Chi-square': float(lr_statistic),
            'LR df': int(lr_dof),
            'Pr(>ChiSq)': float(lr_p_value)
        }

        return summary_df, model_stats

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
