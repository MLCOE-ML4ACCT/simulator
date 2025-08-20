import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from estimators.stat_model.multi_cloglog_irls import MultinomialOrdinalIRLS

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t19_oa_prob.json"

    # Feature & Model Parameters
    FEATURES = [
        "dourt",
        "GCt",
        "DTDEPMA",
        "DZPF",
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
    dZPFt = tf.cast(xt["ZPF"] > 0, dtype=tf.float32)
    dTDEPMAt = tf.cast(xt["TDEPMA"] > 0, dtype=tf.float32)
    dOURt = xt["DOUR"]
    GCt = xt["GC"]
    # Assemble a dictionary of all potential features
    all_features = {
        "dourt": dOURt,
        "GCt": GCt,
        "DTDEPMA": dTDEPMAt,
        "DZPF": dZPFt,
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
    Y = tf.ones_like(xt["OA"], dtype=tf.float32)
    Y = tf.where(xt["OA"] > 0, 2, tf.where(xt["OA"] < 0, 0, 1))
    Y = tf.cast(Y, tf.int32)
    Y = tf.reshape(Y, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    model = MultinomialOrdinalIRLS(
        max_iterations=25,
        tolerance=1e-6,
        patience=5,
        regularization=1e-6,
    )

    model.fit(
        X_train,
        y_train.squeeze(),
        verbose=1,
        validation_data=(X_test, y_test.squeeze()),
    )

    weights, intercepts = (
        model.multinomial_layer.w.numpy().flatten(),
        model.multinomial_layer.b.numpy().flatten(),
    )

    # Debug: Check weight and bias shape
    print("\nEstimated Coefficients:")
    print(f"Intercepts: {intercepts}")
    for i, feature in enumerate(FEATURES):
        print(f"{feature}: {weights[i]}")

    # Assemble statistics
    coeff_names = [f"Intercept{i+1}" for i in range(len(intercepts))] + FEATURES
    coeffs = np.concatenate([intercepts, weights])

    coefficient_stats = []
    for i, name in enumerate(coeff_names):
        coefficient_stats.append(
            {
                "feature": name,
                "Coefficient": float(coeffs[i]),
                "Std. Error": float(model.std_errors[i]),
                "Chi-square": float(model.chi_square_stats[i]),
                "Pr(>ChiSq)": float(model.p_values[i]),
            }
        )

    lr_chi_square = 2 * (model.log_likelihood - model.ll_null)
    lr_df = len(FEATURES)
    lr_p_value = chi2.sf(lr_chi_square, lr_df)

    model_stats = {
        "Log-Likelihood": float(model.log_likelihood),
        "LL-Null": float(model.ll_null),
        "LR Chi-square": float(lr_chi_square),
        "LR df": lr_df,
        "Pr(>ChiSq)": float(lr_p_value),
    }

    result = {
        "coefficients": {
            **{f"Intercept{i+1}": float(intercepts[i]) for i in range(len(intercepts))},
            **{feature: float(weights[i]) for i, feature in enumerate(FEATURES)},
        },
        "statistics": {
            "coefficient_stats": coefficient_stats,
            "model_stats": model_stats,
        },
        "model_info": {
            "n_features": len(FEATURES),
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "train_accuracy": float(model.train_accuracy_tracker.result()),
            "train_loss": float(model.train_loss_tracker.result()),
            "val_accuracy": float(model.val_accuracy_tracker.result()),
            "val_loss": float(model.val_loss_tracker.result()),
        },
    }

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_path}")
