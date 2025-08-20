import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import chi2

from estimators.base_layer.tobit_layer import TobitLayer

# Standard Normal distribution for convenience
tfd = tfp.distributions
STD_NORMAL = tfd.Normal(loc=0.0, scale=1.0)


class TobitIRLS(tf.keras.Model):
    """
    TensorFlow implementation of a Tobit Model for censored regression,
    estimated using an Expectation-Maximization (EM) algorithm which is a
    form of Iteratively Reweighted Least Squares (IRLS).

    This model is designed for a dependent variable that is left-censored at a
    specific point (typically zero). It estimates the parameters of an assumed
    underlying latent linear model.
    """

    def __init__(
        self,
        censor_point=0.0,
        max_iterations=100,
        tolerance=1e-6,
        patience=5,
        regularization=1e-8,
        **kwargs,
    ):
        """
        Initialize the Tobit model.

        Args:
            censor_point (float): The left-censoring point (e.g., 0.0).
            max_iterations (int): Maximum EM/IRLS iterations.
            tolerance (float): Convergence tolerance for parameter changes.
            patience (int): Early stopping patience on validation loss.
            regularization (float): Ridge regularization for numerical stability
                                    in the M-step.
        """
        super().__init__(**kwargs)

        self.censor_point = censor_point
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.regularization = regularization

        # Use TobitLayer for weights
        self.tobit_layer = TobitLayer()

        # Keras-style metric trackers
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        # Add attributes to store statistical results
        self.std_errors = None
        self.chi_square_stats = None
        self.p_values = None
        self.log_likelihood = None
        self.ll_null = None

    def build(self, input_shape):
        """Create the model's weights (parameters)."""
        if not self.tobit_layer.built:
            self.tobit_layer.build(input_shape)
        super().build(input_shape)

    @property
    def sigma(self):
        """Returns the standard deviation sigma from the layer."""
        return self.tobit_layer.scale

    @property
    def log_sigma(self):
        """Returns the log of sigma for calculations."""
        return tf.math.log(self.tobit_layer.scale)

    def call(self, inputs):
        """
        Defines the forward pass, predicting the latent variable y*.

        Returns:
            Predicted latent values (y_star = X * beta).
        """
        y_pred = tf.matmul(inputs, self.tobit_layer.w) + self.tobit_layer.b
        return tf.reshape(y_pred, [-1, 1])

    def _log_likelihood(self, y_true, y_pred_latent, sigma):
        """
        Calculates the log-likelihood for the Tobit model.
        """
        is_censored = y_true <= self.censor_point

        # Standardized latent variable for censored observations
        z_censored = (self.censor_point - y_pred_latent) / sigma

        # Log-likelihood for censored part
        log_prob_censored = STD_NORMAL.log_cdf(z_censored)

        # Log-likelihood for uncensored part
        log_prob_uncensored = (
            STD_NORMAL.log_prob((y_true - y_pred_latent) / sigma) - tf.math.log(sigma)
        )

        log_likelihood = tf.where(is_censored, log_prob_censored, log_prob_uncensored)

        return tf.reduce_sum(log_likelihood)

    def _compute_loss(self, X, y):
        """Computes the Tobit loss for the model."""
        y_pred_latent = self.call(X)
        log_likelihood = self._log_likelihood(y, y_pred_latent, self.sigma)
        return -log_likelihood / tf.cast(tf.shape(y)[0], dtype=tf.float32)

    def _irls_step(self, X, y):
        """
        Performs a single Expectation-Maximization (EM) step, which is a
        form of IRLS for the Tobit model.
        """
        # Store old parameters for convergence check
        old_params = tf.concat(
            [
                tf.reshape(self.tobit_layer.b, [-1]),
                tf.reshape(self.tobit_layer.w, [-1]),
                tf.reshape(self.tobit_layer.scale, [-1]),
            ],
            axis=0,
        )

        # --- E-Step: Impute censored values ---
        y_pred_latent = self.call(X)
        is_censored = y <= self.censor_point

        # Calculate inverse Mills ratio for censored observations
        alpha = (self.censor_point - y_pred_latent) / self.sigma
        imr = STD_NORMAL.prob(alpha) / (STD_NORMAL.cdf(alpha) + 1e-8)

        # Expected value of y* for censored data: E[y*|y*<=c] = Xb - sigma*IMR
        y_star_imputed_censored = y_pred_latent - self.sigma * imr

        # Create the imputed target variable for the M-step
        y_imputed = tf.where(is_censored, y_star_imputed_censored, y)

        # --- M-Step: Update parameters using (weighted) least squares ---
        # 1. Update beta using OLS of imputed y on X
        X_with_intercept = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)
        X_T = tf.transpose(X_with_intercept)
        XTX = tf.matmul(X_T, X_with_intercept)
        XTy = tf.matmul(X_T, y_imputed)

        identity = tf.eye(tf.shape(XTX)[0], dtype=tf.float32) * self.regularization
        params = tf.linalg.solve(XTX + identity, XTy)

        self.tobit_layer.b.assign([params[0, 0]])
        self.tobit_layer.w.assign(params[1:])

        # 2. Update sigma
        # The variance of y* for censored obs: Var(y*|y*<=c) = sigma^2*(1 - alpha*IMR - IMR^2)
        var_y_star_censored = (self.sigma**2) * (1 - alpha * imr - imr**2)

        y_pred_latent_new = self.call(X)
        residuals_uncensored_sq = (y - y_pred_latent_new) ** 2

        # E[(y* - Xb)^2] for censored obs
        residuals_censored_sq_expected = (
            var_y_star_censored + (y_star_imputed_censored - y_pred_latent_new) ** 2
        )

        sum_sq_residuals = tf.reduce_sum(
            tf.where(
                is_censored,
                residuals_censored_sq_expected,
                residuals_uncensored_sq,
            )
        )

        sigma_new = tf.sqrt(
            sum_sq_residuals / tf.cast(tf.shape(X)[0], dtype=tf.float32)
        )
        self.tobit_layer.scale.assign([tf.maximum(sigma_new, 1e-6)])

        # --- Convergence Check ---
        new_params = tf.concat(
            [
                tf.reshape(self.tobit_layer.b, [-1]),
                tf.reshape(self.tobit_layer.w, [-1]),
                tf.reshape(self.tobit_layer.scale, [-1]),
            ],
            axis=0,
        )
        param_change = tf.reduce_max(tf.abs(new_params - old_params))
        loss = self._compute_loss(X, y)

        return loss, param_change

    def _compute_stats(self, X, y):
        """
        Computes statistical properties of the model after fitting.
        """
        variables = [self.tobit_layer.b, self.tobit_layer.w, self.tobit_layer.scale]
        param_shapes = [v.shape for v in variables]
        param_sizes = [tf.reduce_prod(s).numpy() for s in param_shapes]

        def loss_from_flat(flat_params):
            unflattened_params = tf.split(flat_params, param_sizes)
            reshaped_params = [
                tf.reshape(p, s) for p, s in zip(unflattened_params, param_shapes)
            ]
            b, w, scale = reshaped_params[0], reshaped_params[1], reshaped_params[2]
            
            y_pred_latent = tf.matmul(X, w) + b
            log_likelihood = self._log_likelihood(y, y_pred_latent, scale)
            return -log_likelihood

        flat_params = tf.concat(
            [tf.reshape(v, [-1]) for v in variables],
            axis=0
        )

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(flat_params)
            loss = loss_from_flat(flat_params)
            grads = tape.gradient(loss, flat_params)
        
        hessian = tape.jacobian(grads, flat_params)
        del tape

        try:
            if hessian is None:
                raise ValueError("Hessian could not be computed.")
            covariance_matrix = tf.linalg.inv(hessian)
            self.std_errors = tf.sqrt(tf.linalg.diag_part(covariance_matrix)).numpy().flatten()

            params = tf.concat([self.tobit_layer.b, tf.squeeze(self.tobit_layer.w), self.tobit_layer.scale], 0)
            chi_square = (params / self.std_errors) ** 2
            self.chi_square_stats = chi_square.numpy().flatten()
            self.p_values = chi2.sf(self.chi_square_stats, 1).flatten()

        except Exception as e:
            print(f"Could not compute covariance matrix: {e}")
            num_params = len(flat_params)
            self.std_errors = np.full(num_params, np.nan)
            self.chi_square_stats = np.full(num_params, np.nan)
            self.p_values = np.full(num_params, np.nan)

        self.log_likelihood = -loss_from_flat(flat_params).numpy()

        # For Null model (intercept only)
        X_null = tf.zeros((X.shape[0], 0), dtype=tf.float32)
        null_model = TobitIRLS(
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            patience=self.patience,
            regularization=self.regularization
        )
        null_model.fit(X_null, y, verbose=0, compute_stats=False)
        self.ll_null = -null_model._compute_loss(X_null, y).numpy()

    def fit(self, X, y, validation_data=None, verbose=1, compute_stats=True):
        """Fits the model using the custom EM/IRLS training loop."""
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y = tf.expand_dims(tf.squeeze(y), axis=1)

        if not self.built:
            self.build(X.shape)

        if validation_data:
            X_val, y_val = validation_data
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
            y_val = tf.expand_dims(tf.squeeze(y_val), axis=1)
            best_val_loss = np.inf
            epochs_no_improve = 0
            best_weights = None

        if verbose:
            print("Starting Tobit Model Estimation via IRLS (EM Algorithm).")
            print("-" * 60)

        for iteration in range(self.max_iterations):
            loss, param_change = self._irls_step(X, y)
            self.train_loss_tracker.update_state(loss)

            log_line = f"Iter {iteration + 1:2d}: Change = {param_change:.4e} | Train Neg Log-Likelihood = {self.train_loss_tracker.result():.4f}"

            if validation_data:
                val_loss = self._compute_loss(X_val, y_val)
                self.val_loss_tracker.update_state(val_loss)
                log_line += (
                    f" | Val Neg Log-Likelihood = {self.val_loss_tracker.result():.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = self.tobit_layer.get_weights()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    if best_weights is not None:
                        self.tobit_layer.set_weights(best_weights)
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
        else:
            if verbose:
                print(f"\nMax iterations ({self.max_iterations}) reached.")

        if compute_stats:
            if verbose:
                print("-" * 60)
                print("Computing statistics...")
            self._compute_stats(X, y)
            if verbose:
                print("Done.")
                print("-" * 60)

        return self

    def predict(self, X):
        """
        Predicts the expected value of the observed y, E[y|X], which accounts
        for censoring.
        """
        if not self.built:
            raise RuntimeError("Model has not been built. Call fit() first.")
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        y_pred_latent = self.call(X)
        sigma = self.sigma
        z = y_pred_latent / sigma

        # E[y|X] = P(y>0)*E[y|y>0] = CDF(Xb/s)*(Xb + s*IMR)
        prob_positive = STD_NORMAL.cdf(z)
        imr_positive = STD_NORMAL.prob(z) / (prob_positive + 1e-8)

        expected_y = prob_positive * (y_pred_latent + sigma * imr_positive)
        return tf.squeeze(expected_y).numpy()

    def get_coefficients(self):
        """Returns the fitted intercept, coefficients, and sigma."""
        if not self.tobit_layer.built:
            return None, None, None
        return (
            self.tobit_layer.b.numpy()[0],
            self.tobit_layer.w.numpy().flatten(),
            self.tobit_layer.scale.numpy()[0],
        )

    @property
    def metrics(self):
        return [self.train_loss_tracker, self.val_loss_tracker]
