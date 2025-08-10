import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from estimators.stat_model.cloglog_multi_classic_irls import (
    MultinomialCLogLogFisherScoring,
)

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs

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

    # Create and fit the TensorFlow model
    print("Creating Multinomial CLogLog Newton-Raphson model...")
    model = MultinomialCLogLogFisherScoring(
        n_features=X_train.shape[1],
        n_categories=3,
        max_iterations=30,  # Newton-Raphson typically needs fewer iterations
        tolerance=1e-6,
        regularization=1e-5,  # Smaller regularization for less bias
    )

    # Fit the model using the standard TensorFlow interface
    print("Fitting model using model.fit()...")
    model.fit(
        X_train,
        y_train.squeeze(),
        verbose=1,
        validation_data=(X_test, y_test.squeeze()),
    )

    # Get coefficients
    coeffs = model.get_coefficients()
    beta_coefficients = coeffs["beta"]
    cutoff_parameters = coeffs["cutoffs"]

    # Print results
    print("\nEstimated Coefficients (Beta):")
    for i, feature in enumerate(FEATURES):
        print(f"{feature}: {beta_coefficients[i, 0]}")

    print("\nEstimated Cutoffs:")
    for i, cutoff in enumerate(cutoff_parameters):
        print(f"Cutoff {i+1}: {cutoff}")

    # Make predictions using the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_probabilities = model.predict_proba(X_train)
    test_probabilities = model.predict_proba(X_test)

    # Calculate accuracies
    train_accuracy = np.mean(train_predictions == y_train.squeeze())
    test_accuracy = np.mean(test_predictions == y_test.squeeze())

    # Get final metrics from the model
    final_metrics = model.get_metrics()

    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Training Loss: {final_metrics['train_loss']:.6f}")
    print(f"Final Validation Loss: {final_metrics['val_loss']:.6f}")
    print(f"Final Training Accuracy: {final_metrics['train_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {final_metrics['val_accuracy']:.4f}")

    # Prepare results for saving
    results = {
        "coefficients": {
            feature: float(beta_coefficients[i, 0])
            for i, feature in enumerate(FEATURES)
        },
        "cutoffs": [float(cutoff) for cutoff in cutoff_parameters],
        "model_info": {
            "estimator": "multinomial_cloglog_newton_raphson_tf_model",
            "link_function": "complementary_log_log",
            "method": "newton_raphson_with_hessian",
            "framework": "tensorflow_keras_model",
            "optimization": "second_order_newton_raphson",
            "n_features": len(FEATURES),
            "n_categories": 3,
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
        print(f"  Category {cat}: {count} samples ({count/len(y_train)*100:.1f}%)")

    print(f"\nTest set distribution:")
    for cat, count in zip(unique_test, counts_test):
        print(f"  Category {cat}: {count} samples ({count/len(y_test)*100:.1f}%)")

    # Prediction distribution
    unique_pred_train, counts_pred_train = np.unique(
        train_predictions, return_counts=True
    )
    unique_pred_test, counts_pred_test = np.unique(test_predictions, return_counts=True)

    print(f"\nTraining predictions distribution:")
    for cat, count in zip(unique_pred_train, counts_pred_train):
        print(
            f"  Category {cat}: {count} predictions ({count/len(train_predictions)*100:.1f}%)"
        )

    print(f"\nTest predictions distribution:")
    for cat, count in zip(unique_pred_test, counts_pred_test):
        print(
            f"  Category {cat}: {count} predictions ({count/len(test_predictions)*100:.1f}%)"
        )

    # Show model summary
    print(f"\nModel Summary:")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Number of parameters: {sum(np.prod(var.shape) for var in model.trainable_variables)}"
    )
    print(f"Beta shape: {model.beta.shape}")
    print(f"Cutoffs shape: {model.cutoffs.shape}")
