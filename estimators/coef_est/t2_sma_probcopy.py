import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs


class MultinomialCloglogEstimator:
    """
    A class to estimate a Multinomial Ordinal Model using a
    complementary log-log (cloglog) link function.

    The coefficients are found using Maximum Likelihood Estimation.

    This implementation is designed to replicate the methodology described in
    econometric research papers requiring this specific model structure.
    """

    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-6):
        """
        Initializes the estimator.

        Args:
            fit_intercept (bool): Whether to include intercept terms (thresholds) in the model.
            max_iter (int): Maximum number of iterations for the optimization algorithm.
            tol (float): Tolerance for convergence.
        """
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients_ = None
        self.intercepts_ = None
        self.n_classes_ = None
        self.n_iter_ = None
        self.log_likelihood_ = None

    def _cloglog(self, eta):
        """Complementary log-log link function."""
        # Clip eta to prevent overflow in np.exp(eta)
        eta = np.clip(eta, -30, 30)
        return 1 - np.exp(-np.exp(eta))

    def _inverse_cloglog(self, p):
        """Inverse of the complementary log-log link function."""
        # Add a small epsilon to avoid log(0) for p=1
        epsilon = 1e-9
        p = np.clip(p, epsilon, 1 - epsilon)
        return np.log(-np.log(1 - p))

    def _pdf_cloglog(self, eta):
        """Probability density function for the cloglog link."""
        # Clip eta to prevent overflow in np.exp(eta)
        eta = np.clip(eta, -30, 30)
        return np.exp(eta - np.exp(eta))

    def _calculate_probabilities(self, X, intercepts, beta):
        """
        Calculates the probability for each category.

        Args:
            X (np.ndarray): Feature matrix.
            intercepts (np.ndarray): The threshold parameters (alpha_c).
            beta (np.ndarray): The coefficient parameters.

        Returns:
            np.ndarray: A matrix of probabilities (n_samples, n_classes).
        """
        n_samples = X.shape[0]
        eta = X @ beta

        # Calculate cumulative probabilities P(y <= c)
        cumulative_probs = np.zeros((n_samples, self.n_classes_))
        # The intercepts must be sorted for the ordinal model to be valid
        sorted_intercepts = np.sort(intercepts)
        for j in range(self.n_classes_ - 1):
            # Apply clipping here for numerical stability
            linear_predictor = np.clip(sorted_intercepts[j] - eta, -30, 30)
            cumulative_probs[:, j] = self._cloglog(linear_predictor)

        cumulative_probs[:, -1] = 1.0  # The last cumulative probability is always 1

        # Calculate individual category probabilities P(y = c)
        # P(y=c) = P(y<=c) - P(y<=c-1)
        probs = np.zeros_like(cumulative_probs)
        probs[:, 0] = cumulative_probs[:, 0]
        for j in range(1, self.n_classes_):
            probs[:, j] = cumulative_probs[:, j] - cumulative_probs[:, j - 1]

        # Clip probabilities to avoid numerical instability
        return np.clip(probs, 1e-9, 1 - 1e-9)

    def fit(self, X, y):
        """
        Fits the model to the data using classical IRLS algorithm.

        Args:
            X (pd.DataFrame or np.ndarray): The feature matrix (predictor variables), shape [N, d].
            y (pd.Series or np.ndarray): The target variable (ordinal categories), shape [N, 1] or [N,].
        """

        # Ensure y is a 1D array for indexing, handling the [N, 1] case
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        # Determine the number of classes from the unique values in y
        classes = np.unique(y)
        self.n_classes_ = len(classes)
        if self.n_classes_ < 3:
            raise ValueError(
                "Multinomial models require at least 3 outcome categories."
            )

        # One-hot encode the target variable y
        y_one_hot = np.eye(self.n_classes_)[y]

        n_samples, n_features = X.shape

        # --- Step 1: Initialize Parameters ---
        beta = np.zeros(n_features)
        class_freq = np.mean(y_one_hot, axis=0)
        cumulative_freq = np.cumsum(class_freq)
        intercepts = self._inverse_cloglog(cumulative_freq[:-1])

        print("Fitting model using classical IRLS algorithm...")

        # --- Step 2: IRLS Iterations ---
        for iteration in range(self.max_iter):
            print(f"\n--- IRLS Iteration {iteration + 1} ---")
            print(f"Current intercepts: {intercepts}")
            print(f"Current beta (first 3): {beta[:3] if len(beta) > 3 else beta}")

            # Calculate current probabilities
            probs = self._calculate_probabilities(X, intercepts, beta)
            print(
                f"Probability ranges - Min: {np.min(probs):.6f}, Max: {np.max(probs):.6f}"
            )

            # Calculate working response and weights for IRLS
            eta = X @ beta
            working_response = np.zeros((n_samples, self.n_classes_ - 1))
            weights = np.zeros((n_samples, self.n_classes_ - 1))

            for j in range(self.n_classes_ - 1):
                # For ordinal model: working response for threshold j
                cumulative_probs_j = self._cloglog(intercepts[j] - eta)

                # Calculate derivative of cloglog at current point
                deriv = self._pdf_cloglog(intercepts[j] - eta)

                # Working response: eta + (y - mu) / derivative
                observed_cumulative = np.sum(y_one_hot[:, : j + 1], axis=1)
                working_response[:, j] = (intercepts[j] - eta) + (
                    observed_cumulative - cumulative_probs_j
                ) / deriv

                # Weights: derivative^2 / variance
                variance = cumulative_probs_j * (1 - cumulative_probs_j)
                weights[:, j] = deriv**2 / (
                    variance + 1e-8
                )  # Add small constant for stability

                print(
                    f"  Threshold {j}: deriv range [{np.min(deriv):.6f}, {np.max(deriv):.6f}], "
                    f"weight range [{np.min(weights[:, j]):.6f}, {np.max(weights[:, j]):.6f}]"
                )

            # Update parameters using weighted least squares
            old_intercepts = intercepts.copy()
            old_beta = beta.copy()

            print(f"  Starting parameter update...")

            # Update each threshold separately
            for j in range(self.n_classes_ - 1):
                W = np.diag(weights[:, j])
                z = working_response[:, j]

                # Construct design matrix for threshold j: [1, -X]
                design_matrix = np.column_stack([np.ones(n_samples), -X])

                # Weighted least squares: (D'WD)^-1 D'Wz
                try:
                    WD = W @ design_matrix
                    DTWDD = design_matrix.T @ WD
                    DTWz = design_matrix.T @ (W @ z)

                    # Add small regularization for stability
                    DTWDD += 1e-6 * np.eye(DTWDD.shape[0])

                    params_j = np.linalg.solve(DTWDD, DTWz)
                    intercepts[j] = params_j[0]
                    beta = params_j[1:]

                    print(f"    Threshold {j} updated: {intercepts[j]:.6f}")

                except np.linalg.LinAlgError:
                    print(
                        f"    Warning: Singular matrix at iteration {iteration}, threshold {j}, using regularization"
                    )
                    DTWDD += 1e-3 * np.eye(DTWDD.shape[0])
                    params_j = np.linalg.solve(DTWDD, DTWz)
                    intercepts[j] = params_j[0]
                    beta = params_j[1:]

            # Check convergence
            param_change = np.max(
                np.abs(np.concatenate([intercepts - old_intercepts, beta - old_beta]))
            )

            print(f"  Parameter change: {param_change:.2e}")
            print(f"  New intercepts: {intercepts}")
            print(f"  New beta (first 3): {beta[:3] if len(beta) > 3 else beta}")

            if param_change < self.tol:
                print(
                    f"IRLS converged after {iteration + 1} iterations (change: {param_change:.2e})"
                )
                self.n_iter_ = iteration + 1
                break
        else:
            print(f"IRLS reached maximum iterations ({self.max_iter})")
            self.n_iter_ = self.max_iter

        # --- Step 3: Store Results ---
        self.intercepts_ = np.sort(intercepts)
        self.coefficients_ = beta

        # Calculate final log-likelihood
        final_probs = self._calculate_probabilities(
            X, self.intercepts_, self.coefficients_
        )
        self.log_likelihood_ = np.sum(y_one_hot * np.log(final_probs))

        print(f"Final log-likelihood: {self.log_likelihood_:.4f}")

        return self

    def predict_proba(self, X):
        """
        Predicts the class probabilities for a given feature matrix.

        Args:
            X (pd.DataFrame or np.ndarray): The feature matrix.

        Returns:
            np.ndarray: A matrix of predicted probabilities (n_samples, n_classes).
        """
        if self.coefficients_ is None:
            raise RuntimeError("You must fit the model before making predictions.")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self._calculate_probabilities(X, self.intercepts_, self.coefficients_)

    def predict(self, X):
        """
        Predicts the most likely class for a given feature matrix.

        Returns:
            np.ndarray: An array of predicted class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class TFMultinomialCloglogEstimator(tf.keras.Model):
    """
    TensorFlow implementation of Multinomial Ordinal Model using complementary log-log link.
    Uses classical IRLS algorithm with GPU acceleration.
    """

    def __init__(
        self, n_features, fit_intercept=True, max_iter=100, tol=1e-6, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.n_classes_ = None
        self.n_iter_ = None
        self.log_likelihood_ = None

        # These will be initialized in fit()
        self.intercepts = None
        self.coefficients = None

    @tf.function
    def _cloglog(self, eta):
        """Complementary log-log link function."""
        eta = tf.clip_by_value(eta, -30.0, 30.0)
        return 1.0 - tf.exp(-tf.exp(eta))

    @tf.function
    def _inverse_cloglog(self, p):
        """Inverse of the complementary log-log link function."""
        epsilon = 1e-9
        p = tf.clip_by_value(p, epsilon, 1.0 - epsilon)
        return tf.math.log(-tf.math.log(1.0 - p))

    @tf.function
    def _pdf_cloglog(self, eta):
        """Probability density function for the cloglog link."""
        eta = tf.clip_by_value(eta, -30.0, 30.0)
        return tf.exp(eta - tf.exp(eta))

    @tf.function
    def _calculate_probabilities(self, X, intercepts, beta):
        """Calculate probabilities using TensorFlow operations."""
        n_samples = tf.shape(X)[0]
        # Expand beta to [n_features, 1] for matrix multiplication
        beta_expanded = tf.expand_dims(beta, axis=1)
        eta = tf.squeeze(X @ beta_expanded, axis=1)  # Result: [n_samples]

        # Calculate cumulative probabilities P(y <= c)
        cumulative_probs = tf.zeros((n_samples, self.n_classes_), dtype=tf.float32)
        sorted_intercepts = tf.sort(intercepts)

        # Build cumulative probabilities
        cumulative_list = []
        for j in range(self.n_classes_ - 1):
            linear_predictor = tf.clip_by_value(sorted_intercepts[j] - eta, -30.0, 30.0)
            cumulative_list.append(self._cloglog(linear_predictor))

        # Add the final cumulative probability (always 1.0)
        cumulative_list.append(tf.ones(n_samples, dtype=tf.float32))
        cumulative_probs = tf.stack(cumulative_list, axis=1)

        # Calculate individual category probabilities P(y = c)
        probs_list = []
        probs_list.append(cumulative_probs[:, 0])
        for j in range(1, self.n_classes_):
            probs_list.append(cumulative_probs[:, j] - cumulative_probs[:, j - 1])

        probs = tf.stack(probs_list, axis=1)
        return tf.clip_by_value(probs, 1e-9, 1.0 - 1e-9)

    def fit(self, X, y, verbose=True):
        """
        Fits the model using classical IRLS with TensorFlow acceleration.
        """
        with tf.device("/CPU:0"):  # Force CPU computation to avoid GPU memory issues
            # Convert to TensorFlow tensors
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)

            # Ensure y is 1D
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = tf.squeeze(y, axis=1)

            # Determine number of classes
            classes = tf.unique(y)[0]
            self.n_classes_ = len(classes)
            if self.n_classes_ < 3:
                raise ValueError(
                    "Multinomial models require at least 3 outcome categories."
                )

            # One-hot encode
            y_one_hot = tf.one_hot(y, self.n_classes_, dtype=tf.float32)
            n_samples = tf.shape(X)[0]

            # Initialize parameters
            beta = tf.Variable(tf.zeros(self.n_features, dtype=tf.float32))
            class_freq = tf.reduce_mean(y_one_hot, axis=0)
            cumulative_freq = tf.cumsum(class_freq)
            intercepts = tf.Variable(self._inverse_cloglog(cumulative_freq[:-1]))

            if verbose:
                print("Fitting TF model using classical IRLS algorithm...")

            # IRLS iterations
            for iteration in range(self.max_iter):
                if verbose:
                    print(f"\n--- TF-IRLS Iteration {iteration + 1} ---")
                    print(f"Current intercepts: {intercepts.numpy()}")
                    beta_display = (
                        beta.numpy()[:3] if tf.shape(beta)[0] > 3 else beta.numpy()
                    )
                    print(f"Current beta (first 3): {beta_display}")

                # Store old parameters for convergence check
                old_intercepts = tf.identity(intercepts)
                old_beta = tf.identity(beta)

                # Calculate current probabilities
                probs = self._calculate_probabilities(X, intercepts, beta)
                if verbose:
                    print(
                        f"Probability ranges - Min: {tf.reduce_min(probs):.6f}, Max: {tf.reduce_max(probs):.6f}"
                    )

                # IRLS update step
                beta_expanded = tf.expand_dims(beta, axis=1)
                eta = tf.squeeze(X @ beta_expanded, axis=1)

                # Update each threshold using IRLS
                for j in range(self.n_classes_ - 1):
                    # Calculate working response and weights for threshold j
                    cumulative_probs_j = self._cloglog(intercepts[j] - eta)
                    deriv = self._pdf_cloglog(intercepts[j] - eta)

                    # Working response
                    observed_cumulative = tf.reduce_sum(y_one_hot[:, : j + 1], axis=1)
                    working_response = (intercepts[j] - eta) + (
                        observed_cumulative - cumulative_probs_j
                    ) / deriv

                    # Weights
                    variance = cumulative_probs_j * (1.0 - cumulative_probs_j)
                    weights = deriv**2 / (variance + 1e-8)

                    if verbose:
                        print(
                            f"  Threshold {j}: deriv range [{tf.reduce_min(deriv):.6f}, {tf.reduce_max(deriv):.6f}], "
                            f"weight range [{tf.reduce_min(weights):.6f}, {tf.reduce_max(weights):.6f}]"
                        )

                    # Weighted least squares update
                    W = tf.linalg.diag(weights)
                    design_matrix = tf.concat(
                        [tf.ones((n_samples, 1), dtype=tf.float32), -X], axis=1
                    )

                    # Solve weighted least squares: (D'WD)^-1 D'Wz
                    # Force operations to stay on same device
                    with tf.device(
                        "/CPU:0"
                    ):  # Force CPU computation to avoid GPU memory issues
                        WD = W @ design_matrix
                        DTWDD = tf.transpose(design_matrix) @ WD
                        # Expand working_response to [n_samples, 1] for matrix multiplication
                        working_response_expanded = tf.expand_dims(
                            working_response, axis=1
                        )
                        DTWz = tf.transpose(design_matrix) @ (
                            W @ working_response_expanded
                        )
                        # Squeeze back to 1D
                        DTWz = tf.squeeze(DTWz, axis=1)

                        # Add regularization for numerical stability
                        regularization = 1e-6 * tf.eye(
                            tf.shape(DTWDD)[0], dtype=tf.float32
                        )
                        DTWDD_reg = DTWDD + regularization

                        try:
                            # Ensure DTWz is 2D for matrix solve
                            if len(DTWz.shape) == 1:
                                DTWz = tf.expand_dims(DTWz, axis=1)

                            params_j = tf.linalg.solve(DTWDD_reg, DTWz)
                            params_j = tf.squeeze(
                                params_j, axis=1
                            )  # Convert back to 1D

                            intercepts[j].assign(params_j[0])
                            beta.assign(params_j[1:])

                            if verbose:
                                print(
                                    f"    Threshold {j} updated: {intercepts[j].numpy():.6f}"
                                )

                        except tf.errors.InvalidArgumentError:
                            if verbose:
                                print(
                                    f"    Warning: Singular matrix at iteration {iteration}, threshold {j}"
                                )
                            # Use stronger regularization
                            stronger_reg = 1e-3 * tf.eye(
                                tf.shape(DTWDD)[0], dtype=tf.float32
                            )
                            DTWDD_reg = DTWDD + stronger_reg

                            # Ensure DTWz is 2D for matrix solve
                            if len(DTWz.shape) == 1:
                                DTWz = tf.expand_dims(DTWz, axis=1)

                            params_j = tf.linalg.solve(DTWDD_reg, DTWz)
                            params_j = tf.squeeze(
                                params_j, axis=1
                            )  # Convert back to 1D

                            intercepts[j].assign(params_j[0])
                            beta.assign(params_j[1:])

                # Check convergence
                param_change = tf.reduce_max(
                    tf.abs(
                        tf.concat(
                            [intercepts - old_intercepts, beta - old_beta], axis=0
                        )
                    )
                )

                if verbose:
                    print(f"  Parameter change: {param_change.numpy():.2e}")
                    print(f"  New intercepts: {intercepts.numpy()}")
                    beta_display = (
                        beta.numpy()[:3] if tf.shape(beta)[0] > 3 else beta.numpy()
                    )
                    print(f"  New beta (first 3): {beta_display}")

                if param_change < self.tol:
                    if verbose:
                        print(
                            f"TF-IRLS converged after {iteration + 1} iterations (change: {param_change.numpy():.2e})"
                        )
                    self.n_iter_ = iteration + 1
                    break
            else:
                if verbose:
                    print(f"TF-IRLS reached maximum iterations ({self.max_iter})")
                self.n_iter_ = self.max_iter

            # Store final parameters
            self.intercepts = tf.Variable(tf.sort(intercepts))
            self.coefficients = tf.Variable(beta)

            # Calculate final log-likelihood
            final_probs = self._calculate_probabilities(
                X, self.intercepts, self.coefficients
            )
            self.log_likelihood_ = tf.reduce_sum(y_one_hot * tf.math.log(final_probs))

            if verbose:
                print(f"Final log-likelihood: {self.log_likelihood_.numpy():.4f}")

            return self

    @tf.function
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.coefficients is None:
            raise RuntimeError("You must fit the model before making predictions.")
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return self._calculate_probabilities(X, self.intercepts, self.coefficients)

    @tf.function
    def predict(self, X):
        """Predict most likely class."""
        probs = self.predict_proba(X)
        return tf.argmax(probs, axis=1)

    def call(self, X):
        """Forward pass for tf.keras.Model compatibility."""
        return self.predict_proba(X)


if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t2_sma_prob.json"

    # Feature & Model Parameters
    FEATURES = [
        "TDEPMAt_1",
        "EDEPMAt",
        "EDEPBUt_1",
        "EDEPBUt_12",
        "ddmtdmt_1",
        "FAAB",
        "Public",
    ]
    TEST_SET_SIZE = 0.2
    RANDOM_STATE = 42

    ## 2. Data Loading
    xt_1_npz = np.load(TRAIN_DATA_PATH)
    xt_npz = np.load(TEST_DATA_PATH)

    xt_1 = {key: xt_1_npz[key] for key in xt_1_npz.keys()}
    xt = {key: xt_npz[key] for key in xt_npz.keys()}

    xt_1 = unwrap_inputs(xt_1)
    xt = unwrap_inputs(xt)

    ## 3. Feature Engineering
    ddMTDMt_1 = (xt_1["MTDM"] - xt_1["TDEPMA"]) - (xt_1["MTDMt_1"] - xt_1["TDEPMAt_1"])
    dMPAt_1 = xt_1["MPA"] - xt_1["PALLO"]
    dMPAt_2 = xt_1["MPAt_1"] - xt_1["PALLOt_1"]
    ddMPAt_1 = dMPAt_1 - dMPAt_2
    dCASHt_1 = xt_1["CASHFL"] - xt_1["CASHFLt_1"]
    dmCASHt_1 = xt_1["MCASH"] - xt_1["CASHFL"]
    dmCASHt_2 = xt_1["MCASHt_1"] - xt_1["CASHFLt_1"]
    ddmCASHt_1 = dmCASHt_1 - dmCASHt_2
    sumcasht_1 = ddmCASHt_1 + dCASHt_1
    diffcasht_1 = ddmCASHt_1 - dCASHt_1
    sumCACLt_1 = xt_1["CA"] + xt_1["CL"]
    diffCACLt_1 = xt_1["CA"] - xt_1["CL"]

    # Assemble a dictionary of all potential features
    all_features = {
        "sumcasht_1": sumcasht_1,
        "diffcasht_1": diffcasht_1,
        "TDEPMAt_1": xt_1["TDEPMA"],
        "EDEPMAt": xt["EDEPMA"],
        "EDEPMAt2": xt["EDEPMA"] ** 2,
        "MAt_1": xt_1["MA"],
        "I_BUt_1": xt_1["IBU"],
        "I_BUt_12": xt_1["IBU"] ** 2,
        "EDEPBUt_1": xt_1["EDEPBU"],
        "EDEPBUt_12": xt_1["EDEPBU"] ** 2,
        "ddmtdmt_1": ddMTDMt_1,
        "ddmtdmt_12": ddMTDMt_1**2,
        "dcat_1": xt_1["DCA"],
        "ddmpat_1": ddMPAt_1,
        "ddmpat_12": ddMPAt_1**2,
        "dclt_1": xt_1["DCL"],
        "dclt_12": xt_1["DCL"] ** 2,
        "dgnp": xt_1["dgnp"],
        "FAAB": xt_1["FAAB"],
        "Public": xt_1["Public"],
        "ruralare": xt_1["ruralare"],
        "largcity": xt_1["largcity"],
        "market": xt_1["market"],
        "marketw": xt_1["marketw"],
        "sumcaclt_1": sumCACLt_1,
        "diffcaclt_1": diffCACLt_1,
    }

    ## 4. Data Preparation
    X = assemble_tensor(all_features, FEATURES)
    Y = tf.ones_like(xt["SMA"], dtype=tf.float32)
    Y = tf.where(xt["SMA"] > 0, 2, tf.where(xt["SMA"] < 0, 0, 1))

    Y = tf.cast(Y, tf.int32)
    Y = tf.reshape(Y, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    # # Test both implementations
    # print("="*60)
    # print("COMPARING NUMPY vs TENSORFLOW IMPLEMENTATIONS")
    # print("="*60)

    # # 1. Original NumPy implementation
    # print("\n" + "="*30)
    # print("NUMPY IMPLEMENTATION")
    # print("="*30)
    # estimator_np = MultinomialCloglogEstimator(max_iter=10, tol=1e-5)
    # estimator_np.fit(X_train, y_train)

    # print("\n--- NumPy Results ---")
    # print(f"Estimated Intercepts: {estimator_np.intercepts_}")
    # print(f"Estimated Coefficients: {estimator_np.coefficients_}")
    # print(f"Converged in: {estimator_np.n_iter_} iterations")
    # print(f"Final log-likelihood: {estimator_np.log_likelihood_:.4f}")

    # 2. TensorFlow implementation
    print("\n" + "=" * 30)
    print("TENSORFLOW IMPLEMENTATION")
    print("=" * 30)
    estimator_tf = TFMultinomialCloglogEstimator(
        n_features=X_train.shape[1], max_iter=10, tol=1e-5
    )
    estimator_tf.fit(X_train, y_train)

    print("\n--- TensorFlow Results ---")
    print(f"Estimated Intercepts: {estimator_tf.intercepts.numpy()}")
    print(f"Estimated Coefficients: {estimator_tf.coefficients.numpy()}")
    print(f"Converged in: {estimator_tf.n_iter_} iterations")
    print(f"Final log-likelihood: {estimator_tf.log_likelihood_.numpy():.4f}")

    # Test predictions
    test_probs = estimator_tf.predict_proba(X_test[:100])
    test_predictions = estimator_tf.predict(X_test[:100])
    print(f"\nTest predictions shape: {test_probs.shape}")
    print(f"Sample predictions: {test_predictions.numpy()[:10]}")
    print(f"Sample probabilities (first 5 samples):")
    for i in range(5):
        print(f"  Sample {i}: {test_probs.numpy()[i]}")

    print("\n" + "=" * 60)
    print("FINAL TENSORFLOW MODEL RESULTS")
    print("=" * 60)
