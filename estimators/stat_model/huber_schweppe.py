import tensorflow as tf
import tensorflow_probability as tfp


def _huber_loss(y_true, y_pred, scale, k):
    """Helper function to calculate the Huber loss."""
    residuals = y_true - y_pred
    abs_scaled_residuals = tf.abs(residuals / scale)

    # Quadratic loss for small residuals, linear for large ones
    quadratic_loss = 0.5 * (residuals**2)
    linear_loss = scale * k * (tf.abs(residuals) - 0.5 * scale * k)

    loss = tf.where(abs_scaled_residuals <= k, quadratic_loss, linear_loss)
    return tf.reduce_mean(loss)


def huber_schweppe_estimator(
    X_train_np,
    y_train_np,
    X_val_np=None,
    y_val_np=None,
    max_iterations=100,
    tol=1e-6,
    k=1.345,
):
    """
    Estimates regression coefficients using the Huber-Schweppes robust estimator.

    This method is a type of M-estimator that is robust to outliers in both the
    dependent variable (y) and the independent variables (X). It uses an
    Iteratively Reweighted Least Squares (IRLS) approach where the weight of each
    observation is adjusted based on its residual and its leverage.

    Analogy: Imagine a group of people trying to find the center of a room.
    Standard OLS is like everyone shouting their position with equal volume. If one
    person is standing way off in a corner (an outlier), they can pull the
    perceived center way off. The Huber-Schweppes method is like a moderator
    who listens to everyone, but tells the people furthest from the current consensus
    (high leverage and/or high residual) to speak more softly (down-weights them).
    This process repeats until the group finds a stable, robust center that isn't
    skewed by the outliers.

    Args:
        X_train_np (np.ndarray): The training data features.
        y_train_np (np.ndarray): The training data labels.
        X_val_np (np.ndarray, optional): Validation features for monitoring loss.
        y_val_np (np.ndarray, optional): Validation labels for monitoring loss.
        max_iterations (int): The maximum number of IRLS iterations.
        tol (float): The tolerance for convergence. The algorithm stops when the
                     change in coefficients is less than this value.
        k (float): The tuning constant for the Huber psi function. A standard
                   value is 1.345, which provides 95% efficiency on normal data.

    Returns:
        tf.Tensor: The learned model coefficients (beta), including the intercept.
    """
    # --- 1. Data Preparation and Initialization ---
    # Convert numpy arrays to TensorFlow constants for performance.
    X_train = tf.constant(X_train_np, dtype=tf.float32)
    y_train = tf.constant(y_train_np, dtype=tf.float32)

    # Add a bias term (a column of ones) to the feature matrix for the intercept.
    X_train_b = tf.concat([tf.ones((X_train.shape[0], 1)), X_train], axis=1)
    num_samples, num_features = X_train_b.shape

    # Prepare validation data if it exists
    X_val_b = None
    y_val = None
    if X_val_np is not None and y_val_np is not None:
        X_val = tf.constant(X_val_np, dtype=tf.float32)
        y_val = tf.constant(y_val_np, dtype=tf.float32)
        X_val_b = tf.concat([tf.ones((X_val.shape[0], 1)), X_val], axis=1)

    # Initialize coefficients (beta) with zeros.
    beta = tf.Variable(tf.zeros((num_features, 1), dtype=tf.float32))

    # --- 2. Pre-computation of Leverage ---
    # Calculate the diagonal of the Hat Matrix (h_ii). This measures the leverage
    # of each data point. It's a measure of how far an observation's X values are
    # from the mean of the X values. High-leverage points have the potential to
    # be very influential. This is computed once outside the loop for efficiency.
    try:
        X_T = tf.transpose(X_train_b)
        XTX = tf.matmul(X_T, X_train_b)
        # Calculate (X'X)^-1 * X'
        XTX_inv_XT = tf.linalg.solve(XTX, X_T)
        # Calculate diagonal of H = X * (X'X)^-1 * X' using einsum for efficiency
        hat_matrix_diag = tf.einsum("ij,ji->i", X_train_b, XTX_inv_XT)
    except tf.errors.InvalidArgumentError:
        # If the matrix is singular, add a small identity matrix for numerical stability (regularization)
        X_T = tf.transpose(X_train_b)
        identity = tf.eye(num_features, dtype=tf.float32) * 1e-6
        XTX_reg = tf.matmul(X_T, X_train_b) + identity
        # Calculate (X'X + lambda*I)^-1 * X'
        XTX_inv_XT = tf.linalg.solve(XTX_reg, X_T)
        # Calculate diagonal of H = X * (X'X + lambda*I)^-1 * X'
        hat_matrix_diag = tf.einsum("ij,ji->i", X_train_b, XTX_inv_XT)

    # Reshape for broadcasting in the loop
    h_ii = tf.reshape(hat_matrix_diag, (-1, 1))

    print("Starting Huber-Schweppes Robust Estimation via IRLS...")
    print("-" * 60)

    # --- 3. Iteratively Reweighted Least Squares (IRLS) Loop ---
    for i in range(max_iterations):
        beta_old = tf.identity(beta)

        # --- Step 3a: Calculate Residuals and Scale ---
        # Calculate the residuals from the current coefficient estimates.
        residuals = y_train - tf.matmul(X_train_b, beta)

        # Calculate a robust measure of scale: Median Absolute Deviation (MAD).
        # MAD is used instead of standard deviation because it's not sensitive to outliers.
        # The constant 1.4826 makes it an unbiased estimator for the standard deviation
        # of a normal distribution.
        median_residuals = tfp.stats.percentile(residuals, 50.0)
        mad = tfp.stats.percentile(tf.abs(residuals - median_residuals), 50.0)

        # Avoid division by zero if all residuals are the same
        if mad == 0:
            print("MAD is zero. Stopping iterations.")
            break

        scale = mad * 1.4826

        # --- Step 3b: Calculate Schweppe Weights ---
        # This is the key step that differentiates Schweppe from other estimators.
        # The residuals are scaled not only by the robust scale (MAD) but also by
        # their leverage (h_ii). This ensures that high-leverage points are
        # down-weighted more aggressively.
        # Add a small epsilon to the denominator to avoid division by zero if h_ii is exactly 1.
        scaled_residuals = residuals / (scale * tf.sqrt(1.0 - h_ii + 1e-8))

        # The Huber weight function psi(u)/u is applied to the scaled residuals.
        # It returns 1 for "well-behaved" points and a smaller weight for outliers.
        # |u| <= k: weight is 1
        # |u| > k: weight is k / |u|
        abs_scaled_residuals = tf.abs(scaled_residuals)
        weights = tf.where(
            abs_scaled_residuals <= k, 1.0, k / (abs_scaled_residuals + 1e-8)
        )

        # --- Step 3c: Solve the Weighted Least Squares ---
        # Create a diagonal weight matrix for the WLS calculation.
        W_diag = tf.squeeze(weights)

        # Perform the Weighted Least Squares (WLS) update for beta.
        # beta_new = (X' * W * X)^-1 * (X' * W * y)
        X_T = tf.transpose(X_train_b)
        X_T_W = X_T * W_diag  # Broadcasting for efficiency
        X_T_W_X = tf.matmul(X_T_W, X_train_b)
        X_T_W_y = tf.matmul(X_T_W, y_train)

        try:
            # Add a small identity matrix for numerical stability (regularization)
            identity = tf.eye(num_features, dtype=tf.float32) * 1e-8
            beta.assign(tf.linalg.solve(X_T_W_X + identity, X_T_W_y))
        except tf.errors.InvalidArgumentError:
            print(f"Iteration {i+1}: Matrix is singular. Stopping.")
            beta.assign(beta_old)  # Revert to last good estimate
            break

        # --- Step 3d: Evaluation and Convergence Checks ---
        diff = tf.reduce_sum(tf.abs(beta - beta_old))

        log_message = f"Iter {i+1:2d}: Change in coefficients = {diff:.8f}"

        # Calculate and display validation loss if data is provided
        if X_val_b is not None:
            y_val_pred = tf.matmul(X_val_b, beta)
            val_loss = _huber_loss(y_val, y_val_pred, scale, k)
            log_message += f" | Val Loss = {val_loss:.6f}"

        print(log_message)

        if diff < tol:
            print(f"\nConvergence reached after {i+1} iterations.")
            break
    else:  # This else belongs to the for loop, runs if the loop completes without a 'break'
        print(f"\nMax iterations ({max_iterations}) reached.")

    print("-" * 60)
    return beta
