import numpy as np
import tensorflow as tf


class MultinomialOrdinalIRLS(tf.keras.Model):
    """
    TensorFlow implementation of a Multinomial Ordinal Proportional Odds Model
    with a complementary log-log link, structured to align with Keras best practices.

    This model is refactored to have a similar interface and pattern as the
    BinaryCLogLogIRLS model, emphasizing modularity and reusability.
    """

    def __init__(
        self,
        n_features,
        n_classes,
        max_iterations=100,
        tolerance=1e-6,
        patience=5,
        regularization=1e-6,
        **kwargs,
    ):
        """
        Initialize the Multinomial Ordinal IRLS model.

        Args:
            n_features (int): Number of input features.
            n_classes (int): Number of ordinal outcome categories.
            max_iterations (int): Maximum IRLS iterations.
            tolerance (float): Convergence tolerance for parameter changes.
            patience (int): Early stopping patience on validation loss.
            regularization (float): Ridge regularization for numerical stability.
        """
        super().__init__(**kwargs)

        if n_classes < 3:
            raise ValueError(
                "This ordinal model requires at least 3 outcome categories."
            )

        self.n_features = n_features
        self.n_classes = n_classes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience
        self.regularization = regularization

        # Model parameters (will be created in build)
        self.intercepts = None
        self.coefficients = None

        # Keras-style metric trackers for training and validation
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(
            name="train_accuracy"
        )
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(
            name="val_accuracy"
        )

    def build(self, input_shape):
        """
        Create the model's weights (parameters) using Keras's build method.
        This is called automatically the first time the model is used.
        """
        # Intercepts (thresholds), one for each of the K-1 cumulative probabilities
        self.intercepts = self.add_weight(
            name="intercepts",
            shape=(self.n_classes - 1,),
            initializer="zeros",
            trainable=True,
        )
        # Coefficients (beta), shared across all categories (proportional odds)
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(self.n_features, 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    @tf.function
    def _cloglog(self, eta):
        """Helper for the complementary log-log link function."""
        eta = tf.clip_by_value(eta, -30.0, 30.0)
        return 1.0 - tf.exp(-tf.exp(eta))

    @tf.function
    def _pdf_cloglog(self, eta):
        """Helper for the probability density function of the cloglog link."""
        eta = tf.clip_by_value(eta, -30.0, 30.0)
        return tf.exp(eta - tf.exp(eta))

    def call(self, inputs, training=None):
        """
        Defines the forward pass of the model.

        Args:
            inputs: Input tensor (batch_size, n_features).
            training: Boolean indicating if the model is in training mode.

        Returns:
            Predicted class probabilities (batch_size, n_classes).
        """
        # Ensure intercepts are sorted for monotonic probabilities
        sorted_intercepts = tf.sort(self.intercepts)
        eta = tf.matmul(inputs, self.coefficients)

        # Calculate cumulative probabilities P(y <= k)
        # The linear predictor is theta_k - eta
        linear_predictors = sorted_intercepts - eta
        cumulative_probs = self._cloglog(linear_predictors)

        # Pad with 0 at the beginning and 1 at the end for easier subtraction
        # P(y <= 0) is 0, P(y <= K) is 1
        padded_cum_probs = tf.pad(cumulative_probs, [[0, 0], [1, 1]], constant_values=0)
        padded_cum_probs = tf.concat(
            [padded_cum_probs[:, :-1], tf.ones_like(eta)], axis=1
        )

        # Individual probabilities are P(y=k) = P(y<=k) - P(y<=k-1)
        probs = padded_cum_probs[:, 1:] - padded_cum_probs[:, :-1]
        return tf.clip_by_value(probs, 1e-9, 1.0)

    def _compute_loss(self, X, y_one_hot):
        """
        Computes the negative log-likelihood loss for the model.
        """
        probs = self.call(X)
        log_likelihood = tf.reduce_sum(y_one_hot * tf.math.log(probs))
        return -log_likelihood / tf.cast(tf.shape(X)[0], dtype=tf.float32)

    def _irls_step(self, X, y_one_hot):
        """
        Performs a single Iteratively Reweighted Least Squares (IRLS) step
        to update all model parameters (intercepts and coefficients).
        """
        # Store old parameters for convergence check
        old_params = tf.concat([self.intercepts, tf.squeeze(self.coefficients)], axis=0)

        # --- THE FIX: Squeeze eta to be a 1D vector ---
        eta = tf.squeeze(
            tf.matmul(X, self.coefficients)
        )  # Shape is now (n_samples,) instead of (n_samples, 1)

        n_samples = tf.cast(tf.shape(X)[0], dtype=tf.float32)
        temp_beta = tf.identity(self.coefficients)
        new_intercepts = []

        for j in range(self.n_classes - 1):
            intercept_j = self.intercepts[j]
            # Now all operations are between scalars and 1D vectors, preventing bad broadcasting
            linear_pred = intercept_j - eta

            cum_probs_j = self._cloglog(linear_pred)
            deriv = self._pdf_cloglog(linear_pred)

            observed_cumulative = tf.reduce_sum(y_one_hot[:, : j + 1], axis=1)
            working_response = linear_pred + (observed_cumulative - cum_probs_j) / (
                deriv + 1e-8
            )

            variance = cum_probs_j * (1.0 - cum_probs_j)
            weights = deriv**2 / (variance + 1e-8)
            W = tf.linalg.diag(weights)

            design_matrix = tf.concat([tf.ones((tf.shape(X)[0], 1)), -X], axis=1)

            DTWD = tf.transpose(design_matrix) @ W
            DTWDD = DTWD @ design_matrix
            # working_response is now correctly a 1D vector, so this multiplication will work
            DTWz = DTWD @ tf.expand_dims(working_response, axis=1)

            reg_matrix = self.regularization * tf.eye(tf.shape(DTWDD)[0])

            try:
                params_j = tf.linalg.solve(DTWDD + reg_matrix, DTWz)
            except tf.errors.InvalidArgumentError:
                params_j = tf.linalg.solve(
                    DTWDD + (1e-3 * tf.eye(tf.shape(DTWDD)[0])), DTWz
                )

            new_intercepts.append(params_j[0, 0])
            if j == 0:
                temp_beta = params_j[1:]
            else:
                temp_beta = (temp_beta + params_j[1:]) / 2.0

        self.intercepts.assign(tf.stack(new_intercepts))
        self.coefficients.assign(temp_beta)

        new_params = tf.concat([self.intercepts, tf.squeeze(self.coefficients)], axis=0)
        param_change = tf.reduce_max(tf.abs(new_params - old_params))
        loss = self._compute_loss(X, y_one_hot)

        return loss, param_change

    def fit(self, X, y, validation_data=None, verbose=1):
        """
        Fits the model using the custom IRLS training loop.
        """
        # --- Data Preparation ---
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        if len(y.shape) > 1:
            y = tf.squeeze(y)
        y_one_hot = tf.one_hot(y, self.n_classes, dtype=tf.float32)

        # Build model if not already built
        if not self.built:
            self.build(X.shape)

        # Prepare validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
            if len(y_val.shape) > 1:
                y_val = tf.squeeze(y_val)
            y_val_one_hot = tf.one_hot(y_val, self.n_classes, dtype=tf.float32)

            best_val_loss = np.inf
            epochs_no_improve = 0
            # Store best weights
            best_intercepts = tf.Variable(tf.zeros_like(self.intercepts))
            best_coefficients = tf.Variable(tf.zeros_like(self.coefficients))

        if verbose:
            print("Starting Multinomial Ordinal IRLS Fitting...")
            print("-" * 60)

        # --- IRLS Training Loop ---
        for iteration in range(self.max_iterations):
            loss, param_change = self._irls_step(X, y_one_hot)

            # Update training metrics
            self.train_loss_tracker.update_state(loss)
            self.train_accuracy_tracker.update_state(y_one_hot, self.call(X))

            log_line = f"Iter {iteration + 1:2d}: Change = {param_change:.4e} | Train Loss = {self.train_loss_tracker.result():.4f}"

            # Validation and Early Stopping
            if validation_data:
                val_loss = self._compute_loss(X_val, y_val_one_hot)
                self.val_loss_tracker.update_state(val_loss)
                self.val_accuracy_tracker.update_state(y_val_one_hot, self.call(X_val))
                log_line += f" | Val Loss = {self.val_loss_tracker.result():.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_intercepts.assign(self.intercepts)
                    best_coefficients.assign(self.coefficients)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    self.intercepts.assign(best_intercepts)
                    self.coefficients.assign(best_coefficients)
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

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.built:
            raise RuntimeError("Model has not been built. Call fit() first.")
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return self.call(X).numpy()

    def predict(self, X):
        """Predict the most likely class label."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_coefficients(self):
        """Returns the fitted intercepts and coefficients."""
        if not self.built:
            return None, None
        return self.intercepts.numpy(), self.coefficients.numpy().flatten()
