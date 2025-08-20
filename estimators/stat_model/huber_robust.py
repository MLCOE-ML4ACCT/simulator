import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from estimators.base_layer.logistic_layer import LogisticLayer


class HuberSchweppeIRLS(tf.keras.Model):
    """
    TensorFlow implementation of a Huber-Schweppe Robust Regression Model
    using Iteratively Reweighted Least Squares (IRLS), structured to align
    with Keras best practices.

    This model is designed to be robust to outliers in both the dependent
    variable (y) and the independent variables (X) by using leverage-adjusted
    weights in the IRLS procedure.
    """

    def __init__(
        self,
        max_iterations=100,
        tolerance=1e-6,
        patience=5,
        k=1.345,
        regularization=1e-8,
        **kwargs,
    ):
        """
        Initialize the Huber-Schweppe Robust Regression model.

        Args:
            max_iterations (int): Maximum IRLS iterations.
            tolerance (float): Convergence tolerance for parameter changes.
            patience (int): Early stopping patience on validation loss.
            k (float): Tuning constant for Huber psi function (1.345 for 95% efficiency).
            regularization (float): Ridge regularization for numerical stability.
        """
        super().__init__(**kwargs)

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.k = k
        self.regularization = regularization

        # Use LogisticLayer for weights
        self.logistic_layer = LogisticLayer()

        # Keras-style metric trackers for training and validation
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="train_mae")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="val_mae")

    def build(self, input_shape):
        """
        Create the model's weights (parameters) using Keras's build method.
        This is called automatically the first time the model is used.
        """
        if not self.logistic_layer.built:
            self.logistic_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Defines the forward pass of the model.

        Args:
            inputs: Input tensor (batch_size, n_features).
            training: Boolean indicating if the model is in training mode.

        Returns:
            Predicted values (batch_size, 1).
        """
        return self.logistic_layer(inputs)

    def _huber_loss(self, y_true, y_pred, scale):
        """Helper function to calculate the Huber loss."""
        residuals = y_true - y_pred
        abs_scaled_residuals = tf.abs(residuals / scale)

        # Quadratic loss for small residuals, linear for large ones
        quadratic_loss = 0.5 * (residuals**2)
        linear_loss = scale * self.k * (tf.abs(residuals) - 0.5 * scale * self.k)

        loss = tf.where(abs_scaled_residuals <= self.k, quadratic_loss, linear_loss)
        return tf.reduce_mean(loss)

    def _compute_loss(self, X, y):
        """
        Computes the Huber loss for the model.
        """
        y_pred = self.call(X)
        residuals = y - y_pred

        # Calculate robust scale using MAD
        median_residuals = tfp.stats.percentile(residuals, 50.0)
        mad = tfp.stats.percentile(tf.abs(residuals - median_residuals), 50.0)
        scale = tf.maximum(mad * 1.4826, 1e-6)  # Avoid zero scale

        return self._huber_loss(y, y_pred, scale)

    def _compute_leverage(self, X):
        """
        Compute leverage (diagonal of hat matrix) for each observation.
        """
        # Add intercept column
        X_with_intercept = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)

        try:
            X_T = tf.transpose(X_with_intercept)
            XTX = tf.matmul(X_T, X_with_intercept)
            # Calculate (X'X)^-1 * X'
            XTX_inv_XT = tf.linalg.solve(XTX, X_T)
            # Calculate diagonal of H = X * (X'X)^-1 * X'
            hat_matrix_diag = tf.einsum("ij,ji->i", X_with_intercept, XTX_inv_XT)
        except tf.errors.InvalidArgumentError:
            # Add regularization if matrix is singular
            X_T = tf.transpose(X_with_intercept)
            identity = (
                tf.eye(tf.shape(X_with_intercept)[1], dtype=tf.float32)
                * self.regularization
            )
            XTX_reg = tf.matmul(X_T, X_with_intercept) + identity
            XTX_inv_XT = tf.linalg.solve(XTX_reg, X_T)
            hat_matrix_diag = tf.einsum("ij,ji->i", X_with_intercept, XTX_inv_XT)

        return tf.reshape(hat_matrix_diag, (-1, 1))

    def _irls_step(self, X, y):
        """
        Performs a single Iteratively Reweighted Least Squares (IRLS) step
        to update all model parameters.
        """
        # Store old parameters for convergence check
        old_params = tf.concat(
            [self.logistic_layer.b, tf.squeeze(self.logistic_layer.w)], axis=0
        )

        # Calculate current predictions and residuals
        y_pred = self.call(X)
        residuals = y - y_pred

        # Calculate robust scale using MAD
        median_residuals = tfp.stats.percentile(residuals, 50.0)
        mad = tfp.stats.percentile(tf.abs(residuals - median_residuals), 50.0)

        # Avoid division by zero - use fallback scale if MAD is too small
        if mad < 1e-6:
            scale = tf.maximum(tf.math.reduce_std(residuals), 1e-6)
        else:
            scale = mad * 1.4826

        # Calculate leverage
        h_ii = self._compute_leverage(X)

        # Calculate Schweppe weights
        scaled_residuals = residuals / (scale * tf.sqrt(1.0 - h_ii + 1e-8))
        abs_scaled_residuals = tf.abs(scaled_residuals)
        weights = tf.where(
            abs_scaled_residuals <= self.k, 1.0, self.k / (abs_scaled_residuals + 1e-8)
        )

        # Weighted least squares update
        X_with_intercept = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)
        W_diag = tf.squeeze(weights)

        X_T = tf.transpose(X_with_intercept)
        X_T_W = X_T * W_diag
        X_T_W_X = tf.matmul(X_T_W, X_with_intercept)
        X_T_W_y = tf.matmul(X_T_W, y)

        try:
            # Add regularization for numerical stability
            identity = (
                tf.eye(tf.shape(X_T_W_X)[0], dtype=tf.float32) * self.regularization
            )
            params = tf.linalg.solve(X_T_W_X + identity, X_T_W_y)
        except tf.errors.InvalidArgumentError:
            # Fallback with stronger regularization
            identity = tf.eye(tf.shape(X_T_W_X)[0], dtype=tf.float32) * 1e-3
            params = tf.linalg.solve(X_T_W_X + identity, X_T_W_y)

        # Update parameters in the logistic layer
        self.logistic_layer.b.assign([params[0, 0]])
        self.logistic_layer.w.assign(params[1:])

        # Calculate parameter change and loss
        new_params = tf.concat(
            [self.logistic_layer.b, tf.squeeze(self.logistic_layer.w)], axis=0
        )
        param_change = tf.reduce_max(tf.abs(new_params - old_params))
        loss = self._compute_loss(X, y)

        return loss, param_change

    def fit(self, X, y, validation_data=None, verbose=1):
        """
        Fits the model using the custom IRLS training loop.
        """
        # --- Data Preparation ---
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if len(y.shape) > 1:
            y = tf.squeeze(y)
        y = tf.expand_dims(y, axis=1)  # Ensure y is column vector

        # Build model if not already built
        if not self.built:
            self.build(X.shape)

        # Prepare validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            if len(y_val.shape) > 1:
                y_val = tf.squeeze(y_val)
            y_val = tf.expand_dims(y_val, axis=1)

            best_val_loss = np.inf
            epochs_no_improve = 0
            # Store best weights
            best_weights = None

        if verbose:
            print("Starting Huber-Schweppe Robust Regression via IRLS...")
            print("-" * 60)

        # --- IRLS Training Loop ---
        for iteration in range(self.max_iterations):
            loss, param_change = self._irls_step(X, y)

            # Update training metrics
            self.train_loss_tracker.update_state(loss)
            self.train_mae_tracker.update_state(y, self.call(X))

            log_line = f"Iter {iteration + 1:2d}: Change = {param_change:.4e} | Train Loss = {self.train_loss_tracker.result():.4f} | Train MAE = {self.train_mae_tracker.result():.4f}"

            # Validation and Early Stopping
            if validation_data:
                val_loss = self._compute_loss(X_val, y_val)
                self.val_loss_tracker.update_state(val_loss)
                self.val_mae_tracker.update_state(y_val, self.call(X_val))
                log_line += f" | Val Loss = {self.val_loss_tracker.result():.4f} | Val MAE = {self.val_mae_tracker.result():.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = self.logistic_layer.get_weights()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    if best_weights is not None:
                        self.logistic_layer.set_weights(best_weights)
                    if verbose:
                        print(log_line)
                        print(
                            f"\nEarly stopping: Val loss did not improve for {self.patience} iterations."
                        )
                    break

            if verbose:
                print(log_line)

            if param_change < self.tolerance:
                if verbose:
                    print(f"\nConvergence reached after {iteration + 1} iterations.")
                break
        else:  # This else belongs to the for loop, runs if loop finishes without break
            if verbose:
                print(f"\nMax iterations ({self.max_iterations}) reached.")

        if verbose:
            print("-" * 60)
        return self

    def summary(self, X, y, feature_names=None):
        """
        Computes and returns a summary of the regression results, including
        coefficients, standard errors, t-statistics, and p-values.

        Args:
            X (tf.Tensor or np.ndarray): The input features used for fitting.
            y (tf.Tensor or np.ndarray): The target variable used for fitting.
            feature_names (list of str, optional): Names of the features.

        Returns:
            pd.DataFrame: A DataFrame containing the regression summary.
        """
        if not self.built:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        import pandas as pd
        from scipy.stats import norm

        # Ensure data is in the correct format
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        if len(y.shape) > 1:
            y = tf.squeeze(y)
        y = tf.expand_dims(y, axis=1)

        # --- 1. Get Final Model State ---
        y_pred = self.call(X)
        residuals = y - y_pred
        X_with_intercept = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)
        
        # --- 2. Calculate Final Weights ---
        # Re-calculate final weights as they are not stored
        median_residuals = tfp.stats.percentile(residuals, 50.0)
        mad = tfp.stats.percentile(tf.abs(residuals - median_residuals), 50.0)
        scale = mad * 1.4826
        h_ii = self._compute_leverage(X)
        scaled_residuals = residuals / (scale * tf.sqrt(1.0 - h_ii + 1e-8))
        abs_scaled_residuals = tf.abs(scaled_residuals)
        weights = tf.where(
            abs_scaled_residuals <= self.k, 1.0, self.k / (abs_scaled_residuals + 1e-8)
        )
        W_diag = tf.squeeze(weights)

        # --- 3. Calculate Covariance Matrix ---
        # Classic asymptotic covariance matrix for WLS
        X_T_W = tf.transpose(X_with_intercept) * W_diag
        XTWX = tf.matmul(X_T_W, X_with_intercept)
        
        # Add regularization to prevent singularity
        identity = tf.eye(tf.shape(XTWX)[0], dtype=tf.float32) * self.regularization
        XTWX_inv = tf.linalg.inv(XTWX + identity)

        # Estimate of error variance (sigma^2)
        # Using sum of squared weighted residuals
        df = tf.cast(tf.shape(X_with_intercept)[0] - tf.shape(X_with_intercept)[1], dtype=tf.float32)
        sigma_sq_hat = tf.reduce_sum(weights * (residuals**2)) / df
        
        cov_matrix = sigma_sq_hat * XTWX_inv

        # --- 4. Extract Results ---
        std_errors = tf.sqrt(tf.linalg.diag_part(cov_matrix))
        
        params = tf.concat([self.logistic_layer.b, tf.squeeze(self.logistic_layer.w)], axis=0)
        
        t_stats = params / std_errors
        
        # p-values from normal distribution (two-tailed test)
        p_values = 2 * (1 - norm.cdf(np.abs(t_stats.numpy())))

        # Calculate 95% confidence intervals
        ci_lower = params - 1.96 * std_errors
        ci_upper = params + 1.96 * std_errors

        # --- 5. Format Output ---
        if feature_names is None:
            index_names = ['Intercept'] + [f'feature_{i+1}' for i in range(X.shape[1])]
        else:
            index_names = ['Intercept'] + feature_names
        
        summary_df = pd.DataFrame({
            'Coefficient': params.numpy(),
            'Std. Error': std_errors.numpy(),
            't-statistic': t_stats.numpy(),
            'p-value': p_values,
            'CI Lower': ci_lower.numpy(),
            'CI Upper': ci_upper.numpy()
        }, index=index_names)

        return summary_df

    def predict(self, X):
        """Predict values for input X."""
        if not self.built:
            raise RuntimeError("Model has not been built. Call fit() first.")
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.squeeze(self.call(X)).numpy()

    @property
    def metrics(self):
        """Return the model's metrics for Keras compatibility."""
        return [
            self.train_loss_tracker,
            self.train_mae_tracker,
            self.val_loss_tracker,
            self.val_mae_tracker,
        ]

    def reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset_states()
