import tensorflow as tf

from estimators.base_layer.multinomial_layer import MultinomialLayer


class MultinomialCLogLogIRLS(tf.keras.Model):
    """
    TensorFlow model for Multinomial Logit with Complementary Log-Log link using IRLS

    This class wraps the IRLS estimator as a proper TensorFlow model that supports:
    - model.fit(X_train, y_train)
    - tf.function decorators for performance
    - Standard TensorFlow model interface
    """

    def __init__(
        self,
        n_features,
        n_categories=3,
        max_iterations=100,
        tolerance=1e-6,
        damping_factor=0.5,
        regularization=1e-4,
        **kwargs,
    ):
        """
        Initialize the Multinomial CLogLog Newton-Raphson model

        Args:
            n_features: Number of input features
            n_categories: Number of categories (default 3 for -1, 0, +1)
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for Newton steps (0.5-1.0)
            regularization: Ridge regularization for Hessian stability
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_categories = n_categories
        self.n_cutoffs = n_categories - 1
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        self.regularization = regularization

        # Use the existing MultinomialLayer for weight management
        self.multinomial_layer = MultinomialLayer(name="multinomial_layer")

        # Track training and validation metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )

    def build(self, input_shape):
        """Build the layer with the given input shape"""
        # Build the multinomial layer
        self.multinomial_layer.build(input_shape)
        super().build(input_shape)

    @property
    def beta(self):
        """Access to the feature weights from the multinomial layer"""
        return self.multinomial_layer.w

    @property
    def cutoffs(self):
        """Access to the intercepts/cutoffs from the multinomial layer"""
        return self.multinomial_layer.b

    @tf.function
    def _compute_probabilities(self, X):
        """
        Compute class probabilities using complementary log-log link

        Args:
            X: Input features (batch_size, n_features)

        Returns:
            probabilities: Class probabilities (batch_size, n_categories)
        """
        # Linear predictor
        eta = tf.matmul(X, self.beta)  # (batch_size, 1)

        # Create augmented cutoffs with -inf and +inf
        sorted_cutoffs = tf.sort(self.cutoffs)
        padded_cutoffs = tf.concat([[-1e10], sorted_cutoffs, [1e10]], axis=0)

        # Calculate cumulative probabilities for each cutoff
        cutoff_diffs = tf.expand_dims(padded_cutoffs, 1) - tf.transpose(eta)
        cutoff_diffs = tf.clip_by_value(cutoff_diffs, -15.0, 15.0)

        # Cumulative probabilities using cloglog: F(x) = 1 - exp(-exp(x))
        cum_probs = 1.0 - tf.exp(-tf.exp(cutoff_diffs))

        # Individual category probabilities
        cat_probs = cum_probs[1:, :] - cum_probs[:-1, :]
        cat_probs = tf.transpose(cat_probs)

        # Clip for numerical stability
        cat_probs = tf.clip_by_value(cat_probs, 1e-10, 1.0 - 1e-10)

        return cat_probs

    @tf.function
    def call(self, inputs, training=None):
        """
        Forward pass of the model

        Args:
            inputs: Input tensor (batch_size, n_features)
            training: Whether in training mode

        Returns:
            probabilities: Class probabilities (batch_size, n_categories)
        """
        # Build the layer if not built yet
        if not self.multinomial_layer.built:
            self.multinomial_layer.build(inputs.shape)

        return self._compute_probabilities(inputs)

    @tf.function
    def _compute_loss(self, X, y):
        """
        Compute negative log-likelihood loss

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size,)

        Returns:
            loss: Negative log-likelihood
        """
        probs = self._compute_probabilities(X)

        # One-hot encode labels
        y_onehot = tf.one_hot(y, self.n_categories, dtype=tf.float32)

        # Compute negative log-likelihood
        log_likelihood = tf.reduce_sum(y_onehot * tf.math.log(probs + 1e-10))
        loss = -log_likelihood / tf.cast(tf.shape(X)[0], tf.float32)  # Average loss

        return loss

    @tf.function
    def _newton_raphson_step(self, X, y):
        """
        Single Newton-Raphson iteration step using second-order optimization

        This implements the full Newton-Raphson method with Hessian matrix computation,
        similar to what would be used in statistical packages like SAS.

        Args:
            X: Features (batch_size, n_features)
            y: True labels (batch_size,)

        Returns:
            loss: Current loss value
            beta_change: Change in beta parameters
            cutoff_change: Change in cutoff parameters
        """
        # Store old parameters for change calculation
        beta_old = tf.identity(self.beta)
        cutoffs_old = tf.identity(self.cutoffs)

        # Combine parameters into a single vector for joint optimization
        params = tf.concat([tf.reshape(self.beta, [-1]), self.cutoffs], axis=0)

        # Use persistent tape for computing both gradient and Hessian
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(params)

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(params)

                # Unpack parameters
                current_beta = tf.reshape(
                    params[: self.n_features], [self.n_features, 1]
                )
                current_cutoffs = params[self.n_features :]

                # Compute probabilities with current parameters
                eta = tf.matmul(X, current_beta)
                sorted_cutoffs = tf.sort(current_cutoffs)
                padded_cutoffs = tf.concat([[-1e10], sorted_cutoffs, [1e10]], axis=0)

                cutoff_diffs = tf.expand_dims(padded_cutoffs, 1) - tf.transpose(eta)
                cutoff_diffs = tf.clip_by_value(cutoff_diffs, -15.0, 15.0)

                cum_probs = 1.0 - tf.exp(-tf.exp(cutoff_diffs))
                cat_probs = cum_probs[1:, :] - cum_probs[:-1, :]
                cat_probs = tf.transpose(cat_probs)
                cat_probs = tf.clip_by_value(cat_probs, 1e-10, 1.0 - 1e-10)

                # Compute negative log-likelihood
                y_onehot = tf.one_hot(y, self.n_categories, dtype=tf.float32)
                log_likelihood = tf.reduce_sum(y_onehot * tf.math.log(cat_probs))
                loss = -log_likelihood

            # Compute gradient (first derivative)
            gradient = inner_tape.gradient(loss, params)

        # Compute Hessian (second derivative matrix) only if gradient is valid
        if gradient is not None:
            try:
                # Compute Hessian using jacobian of gradient
                hessian = outer_tape.jacobian(gradient, params)

                # Add regularization to Hessian for numerical stability (Levenberg-Marquardt)
                n_params = tf.shape(params)[0]
                hessian_reg = hessian + self.regularization * tf.eye(
                    n_params, dtype=tf.float32
                )

                # Solve Newton system: H * delta = -g
                # Reshape gradient to column vector for matrix solve
                gradient_reshaped = tf.reshape(-gradient, [-1, 1])
                newton_step = tf.linalg.solve(hessian_reg, gradient_reshaped)
                newton_step = tf.reshape(newton_step, [-1])  # Back to 1D

                # Apply damping factor for stability
                newton_step = self.damping_factor * newton_step

                # Clip step size to prevent divergence
                newton_step = tf.clip_by_value(newton_step, -5.0, 5.0)

                # Update parameters
                new_params = params + newton_step

                # Unpack and assign new parameters
                new_beta = tf.reshape(
                    new_params[: self.n_features], [self.n_features, 1]
                )
                new_cutoffs = new_params[self.n_features :]

                # Ensure cutoffs remain ordered
                new_cutoffs = tf.sort(new_cutoffs)

                self.beta.assign(new_beta)
                self.cutoffs.assign(new_cutoffs)

            except (tf.errors.InvalidArgumentError, tf.errors.FailedPreconditionError):
                # Fallback to gradient descent if Hessian computation fails
                gradient_beta = gradient[: self.n_features]
                gradient_cutoffs = gradient[self.n_features :]

                # Small gradient steps as fallback
                self.beta.assign_sub(
                    0.01 * tf.reshape(gradient_beta, [self.n_features, 1])
                )
                cutoffs_update = self.cutoffs - 0.01 * gradient_cutoffs
                self.cutoffs.assign(tf.sort(cutoffs_update))
        else:
            # If gradient is None, make small random adjustments
            noise_beta = tf.random.normal(shape=self.beta.shape, stddev=0.001)
            noise_cutoffs = tf.random.normal(shape=self.cutoffs.shape, stddev=0.001)

            self.beta.assign_add(noise_beta)
            self.cutoffs.assign(tf.sort(self.cutoffs + noise_cutoffs))

        # Clean up persistent tape
        del outer_tape

        # Calculate parameter changes
        beta_change = tf.reduce_sum(tf.abs(self.beta - beta_old))
        cutoff_change = tf.reduce_sum(tf.abs(self.cutoffs - cutoffs_old))

        return loss, beta_change, cutoff_change

    def fit(self, X, y, verbose=1, validation_data=None):
        """
        Fit the model using Newton-Raphson algorithm

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            verbose: Verbosity level
            validation_data: Optional validation data tuple (X_val, y_val)

        Returns:
            self: Fitted model
        """
        # Convert to tensors
        if not isinstance(X, tf.Tensor):
            X = tf.constant(X, dtype=tf.float32)
        if not isinstance(y, tf.Tensor):
            y = tf.constant(y, dtype=tf.int32)

        if len(y.shape) > 1:
            y = tf.squeeze(y)

        # Build the model if not already built
        if not self.built:
            self.build(X.shape)

        if verbose:
            print("Starting Newton-Raphson for Multinomial CLogLog Model...")
            print("-" * 60)

        # Prepare validation data if provided
        X_val, y_val = None, None
        if validation_data is not None:
            X_val, y_val = validation_data
            if not isinstance(X_val, tf.Tensor):
                X_val = tf.constant(X_val, dtype=tf.float32)
            if not isinstance(y_val, tf.Tensor):
                y_val = tf.constant(y_val, dtype=tf.int32)
            if len(y_val.shape) > 1:
                y_val = tf.squeeze(y_val)

        # Training loop
        for iteration in range(self.max_iterations):
            loss, beta_change, cutoff_change = self._newton_raphson_step(X, y)
            total_change = beta_change + cutoff_change

            # Update training metrics
            self.train_loss_tracker.update_state(loss)
            probs = self._compute_probabilities(X)
            predictions = tf.argmax(probs, axis=1)
            # Cast predictions to int32 to match y dtype
            predictions = tf.cast(predictions, tf.int32)
            self.train_accuracy_tracker.update_state(y, predictions)

            # Update validation metrics if validation data provided
            if validation_data is not None:
                val_loss = self._compute_loss(X_val, y_val)
                val_probs = self._compute_probabilities(X_val)
                val_predictions = tf.argmax(val_probs, axis=1)
                val_predictions = tf.cast(val_predictions, tf.int32)

                self.val_loss_tracker.update_state(val_loss)
                self.val_accuracy_tracker.update_state(y_val, val_predictions)

            # Print progress
            if verbose and (iteration % 10 == 0 or iteration < 10):
                if validation_data is not None:
                    print(
                        f"Iteration {iteration + 1:3d}: "
                        f"Loss = {loss.numpy():.6f}, "
                        f"Accuracy = {self.train_accuracy_tracker.result().numpy():.4f}, "
                        f"Val_Loss = {self.val_loss_tracker.result().numpy():.6f}, "
                        f"Val_Accuracy = {self.val_accuracy_tracker.result().numpy():.4f}, "
                        f"Change = {total_change.numpy():.8f}"
                    )
                else:
                    print(
                        f"Iteration {iteration + 1:3d}: "
                        f"Loss = {loss.numpy():.6f}, "
                        f"Accuracy = {self.train_accuracy_tracker.result().numpy():.4f}, "
                        f"Change = {total_change.numpy():.8f}"
                    )

            # Check convergence
            if total_change < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        else:
            if verbose:
                print(f"Maximum iterations ({self.max_iterations}) reached")

        # Print final metrics summary
        if verbose:
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
        Predict class labels

        Args:
            X: Input features

        Returns:
            predictions: Predicted class labels
        """
        if not isinstance(X, tf.Tensor):
            X = tf.constant(X, dtype=tf.float32)

        probs = self._compute_probabilities(X)
        predictions = tf.argmax(probs, axis=1)
        return tf.cast(predictions, tf.int32).numpy()

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

        return self._compute_probabilities(X).numpy()

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
            dict: Dictionary with 'beta' and 'cutoffs'
        """
        return {"beta": self.beta.numpy(), "cutoffs": tf.sort(self.cutoffs).numpy()}

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
