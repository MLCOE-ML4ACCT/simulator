import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
        n_features,
        censor_point=0.0,
        max_iterations=100,
        tolerance=1e-6,
        regularization=1e-8,
        **kwargs,
    ):
        """
        Initialize the Tobit model.

        Args:
            n_features (int): Number of input features.
            censor_point (float): The left-censoring point (e.g., 0.0).
            max_iterations (int): Maximum EM/IRLS iterations.
            tolerance (float): Convergence tolerance for parameter changes.
            regularization (float): Ridge regularization for numerical stability
                                    in the M-step.
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.censor_point = censor_point
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization

        # Model parameters (will be created in build)
        self.coefficients = None
        self.intercept = None
        self.log_sigma = None  # We model log(sigma) to ensure sigma > 0

        # Keras-style metric trackers
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def build(self, input_shape):
        """Create the model's weights (parameters)."""
        self.intercept = self.add_weight(
            name="intercept", shape=(1,), initializer="zeros", trainable=True
        )
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(self.n_features, 1),
            initializer="zeros",
            trainable=True,
        )
        # log(sigma), where sigma is the std. dev. of the error term
        self.log_sigma = self.add_weight(
            name="log_sigma", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    @property
    def sigma(self):
        """Returns the standard deviation sigma, ensuring it's positive."""
        return tf.exp(self.log_sigma)

    def call(self, inputs):
        """
        Defines the forward pass, predicting the latent variable y*.

        Returns:
            Predicted latent values (y_star = X * beta).
        """
        return self.intercept + tf.matmul(inputs, self.coefficients)

    def _log_likelihood(self, y_true, y_pred_latent):
        """
        Calculates the log-likelihood for the Tobit model.

        Args:
            y_true: The observed (and potentially censored) target values.
            y_pred_latent: The predicted latent variable, X*beta.

        Returns:
            The mean negative log-likelihood (loss).
        """
        sigma = self.sigma
        is_censored = tf.squeeze(y_true <= self.censor_point)

        # Standardized latent variable for censored observations
        z_censored = (self.censor_point - y_pred_latent) / sigma

        # Log-likelihood for censored part
        # log[1 - CDF(z)] = log[CDF(-z)]
        log_prob_censored = STD_NORMAL.log_cdf(z_censored)

        # Log-likelihood for uncensored part
        # log[ (1/sigma) * PDF((y - Xb)/sigma) ]
        log_prob_uncensored = (
            STD_NORMAL.log_prob((y_true - y_pred_latent) / sigma) - self.log_sigma
        )

        log_likelihood = tf.where(is_censored, log_prob_censored, log_prob_uncensored)

        # Return the *negative* log-likelihood as the loss to be minimized
        return -tf.reduce_mean(log_likelihood)

    def _compute_loss(self, X, y):
        """Computes the Tobit loss for the model."""
        y_pred_latent = self.call(X)
        return self._log_likelihood(y, y_pred_latent)

    def _irls_step(self, X, y):
        """
        Performs a single Expectation-Maximization (EM) step, which is a
        form of IRLS for the Tobit model.
        """
        # Store old parameters for convergence check
        old_params = tf.concat(
            [self.intercept, tf.squeeze(self.coefficients), self.log_sigma], axis=0
        )

        # --- E-Step: Impute censored values ---
        y_pred_latent = self.call(X)
        is_censored = tf.squeeze(y <= self.censor_point)

        # Calculate inverse Mills ratio for censored observations
        alpha = (self.censor_point - y_pred_latent) / self.sigma
        imr = STD_NORMAL.prob(alpha) / (STD_NORMAL.cdf(alpha) + 1e-8)

        # Expected value of y* for censored data: E[y*|y*<=c] = Xb - sigma*IMR
        y_star_imputed_censored = y_pred_latent - self.sigma * imr

        # Create the imputed target variable for the M-step
        y_imputed = tf.where(
            is_censored, tf.squeeze(y_star_imputed_censored), tf.squeeze(y)
        )
        y_imputed = tf.expand_dims(y_imputed, axis=1)

        # --- M-Step: Update parameters using (weighted) least squares ---
        # 1. Update beta using OLS of imputed y on X
        X_with_intercept = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)
        X_T = tf.transpose(X_with_intercept)
        XTX = tf.matmul(X_T, X_with_intercept)
        XTy = tf.matmul(X_T, y_imputed)

        identity = tf.eye(tf.shape(XTX)[0], dtype=tf.float32) * self.regularization
        params = tf.linalg.solve(XTX + identity, XTy)

        self.intercept.assign([params[0, 0]])
        self.coefficients.assign(params[1:])

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
                tf.squeeze(residuals_censored_sq_expected),
                tf.squeeze(residuals_uncensored_sq),
            )
        )

        sigma_new = tf.sqrt(
            sum_sq_residuals / tf.cast(tf.shape(X)[0], dtype=tf.float32)
        )
        self.log_sigma.assign([tf.math.log(tf.maximum(sigma_new, 1e-6))])

        # --- Convergence Check ---
        new_params = tf.concat(
            [self.intercept, tf.squeeze(self.coefficients), self.log_sigma], axis=0
        )
        param_change = tf.reduce_max(tf.abs(new_params - old_params))
        loss = self._compute_loss(X, y)

        return loss, param_change

    def fit(self, X, y, validation_data=None, verbose=1):
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

        if verbose:
            print("Starting Tobit Model Estimation via IRLS (EM Algorithm)...")
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

            if verbose:
                print(log_line)

            if param_change < self.tolerance:
                if verbose:
                    print(f"\nConvergence reached after {iteration + 1} iterations.")
                break
        else:
            if verbose:
                print(f"\nMax iterations ({self.max_iterations}) reached.")

        if verbose:
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
        if not self.built:
            return None, None, None
        return (
            self.intercept.numpy()[0],
            self.coefficients.numpy().flatten(),
            self.sigma.numpy()[0],
        )

    @property
    def metrics(self):
        return [self.train_loss_tracker, self.val_loss_tracker]
