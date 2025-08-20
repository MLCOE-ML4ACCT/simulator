import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from estimators.stat_model.binary_cloglog_irls import BinaryCLogLogIRLS

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t6_dofa_prob_pos.json"

    # Feature & Model Parameters
    FEATURES = [
        "sumcasht_1",
        "diffcasht_1",
        "ddmpat_1",
        "ddmpat_12",
        "ddmpat_13",
        "DIMA",
        "DIBU",
        "realr",
        "FAAB",
        "Public",
        "ruralare",
        "largcity",
        "market",
        "marketw",
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
    dIMAt = tf.cast(xt["IMA"] > 0, dtype=tf.float32)
    dIBUt = tf.cast(xt["IBU"] > 0, dtype=tf.float32)
    # Assemble a dictionary of all potential features
    all_features = {
        "sumcasht_1": sumcasht_1,
        "diffcasht_1": diffcasht_1,
        "ddmpat_1": ddMPAt_1,
        "ddmpat_12": ddMPAt_1**2,
        "ddmpat_13": ddMPAt_1**3,
        "DIMA": dIMAt,
        "DIBU": dIBUt,
        "realr": xt_1["realr"],
        "FAAB": xt_1["FAAB"],
        "Public": xt_1["Public"],
        "ruralare": xt_1["ruralare"],
        "largcity": xt_1["largcity"],
        "market": xt_1["market"],
        "marketw": xt_1["marketw"],
    }

    ## 4. Data Preparation
    X = assemble_tensor(all_features, FEATURES)
    Y = tf.cast(xt["IBU"] > 0, tf.float32)
    Y = tf.reshape(Y, (-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    # Create and fit the TensorFlow model
    print("Creating Binary CLogLog IRLS model...")
    model = BinaryCLogLogIRLS(
        max_iterations=25,
        tolerance=1e-6,
        patience=5,
        regularization=1e-6,
    )

    # Fit the model using the standard TensorFlow interface
    print("Fitting model using model.fit()...")
    model.fit(
        X_train,
        y_train.squeeze(),
        verbose=1,
        validation_data=(X_test, y_test.squeeze()),
    )

    # --- Get Coefficients and Statistical Summary ---
    summary_df, model_stats = model.summary(X_train, y_train, feature_names=FEATURES)

    print("\n--- Model Summary ---")
    print(summary_df)
    print("\n--- Model Statistics ---")
    for key, value in model_stats.items():
        print(f"{key}: {value}")
    print("-" * 20)

    # --- Prepare JSON Output ---
    # Convert summary DataFrame to a dictionary for JSON serialization
    summary_dict = (
        summary_df.reset_index().rename(columns={"index": "feature"}).to_dict("records")
    )

    # Get coefficients from summary_df to ensure consistency
    coeffs = summary_df["Coefficient"].to_dict()
    intercept = coeffs.pop("Intercept")

    # Make predictions using the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_probabilities = model.predict_proba(X_train)
    test_probabilities = model.predict_proba(X_test)

    # Calculate accuracies
    train_accuracy = np.mean(train_predictions.squeeze() == y_train.squeeze())
    test_accuracy = np.mean(test_predictions.squeeze() == y_test.squeeze())

    # Get final metrics from the model
    final_metrics = model.get_metrics()

    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Training Loss: {final_metrics['train_loss']:.6f}")
    print(f"Final Validation Loss: {final_metrics['val_loss']:.6f}")
    print(f"Final Training Accuracy: {final_metrics['train_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {final_metrics['val_accuracy']:.4f}")

    results = {
        "coefficients": {"Intercept": intercept, **coeffs},
        "statistics": {"coefficient_stats": summary_dict, "model_stats": model_stats},
        "model_info": {
            "estimator": "binary_cloglog_irls_tf_model",
            "link_function": "complementary_log_log",
            "method": "iteratively_reweighted_least_squares",
            "framework": "tensorflow_keras_model",
            "optimization": "irls_with_early_stopping",
            "n_features": len(FEATURES),
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "final_train_loss": float(final_metrics["train_loss"]),
            "final_val_loss": float(final_metrics["val_loss"]),
            "final_train_accuracy": float(final_metrics["train_accuracy"]),
            "final_val_accuracy": float(final_metrics["val_accuracy"]),
        },
    }

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Additional model diagnostics
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)

    # Class distribution
    unique_train, counts_train = np.unique(y_train.squeeze(), return_counts=True)
    unique_test, counts_test = np.unique(y_test.squeeze(), return_counts=True)

    print(f"Training set distribution:")
    for cat, count in zip(unique_train, counts_train):
        print(f"  Class {int(cat)}: {count} samples ({count/len(y_train)*100:.1f}%)")

    print(f"\nTest set distribution:")
    for cat, count in zip(unique_test, counts_test):
        print(f"  Class {int(cat)}: {count} samples ({count/len(y_test)*100:.1f}%)")

    # Prediction distribution
    unique_pred_train, counts_pred_train = np.unique(
        train_predictions.squeeze(), return_counts=True
    )
    unique_pred_test, counts_pred_test = np.unique(
        test_predictions.squeeze(), return_counts=True
    )

    print(f"\nTraining predictions distribution:")
    for cat, count in zip(unique_pred_train, counts_pred_train):
        print(
            f"  Class {int(cat)}: {count} predictions ({count/len(train_predictions)*100:.1f}%)"
        )

    print(f"\nTest predictions distribution:")
    for cat, count in zip(unique_pred_test, counts_pred_test):
        print(
            f"  Class {int(cat)}: {count} predictions ({count/len(test_predictions)*100:.1f}%)"
        )

    # Show model summary
    print(f"\nModel Summary:")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Number of parameters: {sum(np.prod(var.shape) for var in model.trainable_variables)}"
    )
    print(f"Weights shape: {model.logistic_layer.w.shape}")

    print(f"\nCoefficients successfully saved to: {output_path}")
