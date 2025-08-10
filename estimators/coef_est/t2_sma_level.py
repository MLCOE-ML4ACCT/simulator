import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from estimators.stat_model.huber_robust import HuberSchweppeIRLS

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t2_sma_level.json"

    # Feature & Model Parameters
    FEATURES = [
        "EDEPMAt",
        "MAt_1",
        "EDEPBUt_1",
        "ddmtdmt_1",
        "dcat_1",
        "dclt_1",
        "dgnp",
        "FAAB",
        "largcity",
        "market",
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
    Y = xt["SMA"]
    Y = tf.reshape(Y, (-1, 1))

    # Debug: Check Y distribution
    print(
        f"Y stats: min={np.min(Y):.4f}, max={np.max(Y):.4f}, mean={np.mean(Y):.4f}, median={np.median(Y):.4f}"
    )
    print(f"Y unique values: {len(np.unique(Y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    model = HuberSchweppeIRLS(
        n_features=len(FEATURES),
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

    intercept, weight = model.get_coefficients()

    print("\nEstimated Coefficients:")
    print(f"Bias (Intercept): {intercept}")
    for i, feature in enumerate(FEATURES):
        print(f"{feature}: {weight[i]}")

    result = {
        "coefficients": {
            "Intercept": float(intercept),
            **{FEATURES[i]: float(coef) for i, coef in enumerate(weight)},
        },
        "model_info": {
            "n_features": len(FEATURES),
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "n_outliers": len(FEATURES),
            "huber_k": 1.345,
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
