import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from estimators.stat_model.huber_robust import HuberSchweppeIRLS
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t16_zpf_level.json"

    # Feature & Model Parameters
    FEATURES = [
        "sumcasht_1",
        "diffcasht_1",
        "PALLOt_1",
        "ddmpat_1",
        "DTDEPMA",
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
    ddOFAt = tf.cast(xt["DOFA"] != 0, dtype=tf.float32)
    EDEPMAt = xt["EDEPMA"]
    SMAt = xt["SMA"]
    IMAt = xt["IMA"]
    IBUt = xt["IBU"]
    EDEPBUt = xt["EDEPBU"]
    dCAt = xt["DCA"]
    ddLLt = tf.cast(xt["DLL"] != 0, dtype=tf.float32)
    ddSCt = tf.cast(xt["DSC"] != 0, dtype=tf.float32)
    dOFAt = xt["DOFA"]
    dCLt = xt["DCL"]
    dLLt = xt["DLL"]
    sumdCAdCLt = dCAt + dCLt
    diffdCAdCLt = dCAt - dCLt
    sumdOFAdLLt = dOFAt + dLLt
    diffdOFAdLLt = dOFAt - dLLt
    dTDEPMAt = tf.cast(xt["TDEPMA"] > 0, dtype=tf.float32)

    # Assemble a dictionary of all potential features
    all_features = {
        "sumcasht_1": sumcasht_1,
        "diffcasht_1": diffcasht_1,
        "PALLOt_1": xt_1["PALLO"],
        "ddmpat_1": ddMPAt_1,
        "ddmpat_12": ddMPAt_1**2,
        "ddmpat_13": ddMPAt_1**3,
        "DTDEPMA": dTDEPMAt,
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
    Y = xt["ZPF"]
    mask = Y > 0
    X = tf.boolean_mask(X, mask)
    Y = tf.boolean_mask(Y, mask)

    Y = tf.reshape(Y, (-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    model = HuberSchweppeIRLS(
        max_iterations=50,
        tolerance=1e-5,
        patience=5,
        k=1.345,  # Standard Huber constant
        regularization=1e-8,
    )

    model.fit(
        X_train,
        y_train,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    # --- Get Coefficients and Statistical Summary ---
    weights = model.logistic_layer.get_weights()
    weight = weights[0].flatten()
    intercept = weights[1][0]

    print("\n--- Model Summary ---")
    summary_df = model.summary(X_train, y_train, feature_names=FEATURES)
    print(summary_df)
    print("-" * 20)

    # --- Prepare JSON Output ---
    # Convert summary DataFrame to a dictionary for JSON serialization
    summary_dict = (
        summary_df.reset_index().rename(columns={"index": "feature"}).to_dict("records")
    )

    result = {
        "coefficients": {
            "Intercept": float(intercept),
            **{FEATURES[i]: float(coef) for i, coef in enumerate(weight)},
        },
        "statistics": summary_dict,
        "model_info": {
            "n_features": len(FEATURES),
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "n_outliers": len(
                FEATURES
            ),  # This might need to be recalculated based on robust stats
            "huber_k": model.k,
            "train_loss": float(model.train_loss_tracker.result()),
            "train_mae": float(model.train_mae_tracker.result()),
            "val_loss": float(model.val_loss_tracker.result()),
            "val_mae": float(model.val_mae_tracker.result()),
        },
    }

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_path}")
