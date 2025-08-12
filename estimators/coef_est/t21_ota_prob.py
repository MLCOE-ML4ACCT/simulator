import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from estimators.stat_model.binary_logistic_irls import BinaryLogisticIRLS
from estimators.stat_model.huber_robust import HuberSchweppeIRLS
from estimators.stat_model.multi_cloglog_irls import MultinomialOrdinalIRLS

# Assuming these are your custom modules
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t21_ota_prob.json"

    # Feature & Model Parameters
    FEATURES = [
        "PALLOt_1",
        "ZPFt",
        "TDEPMAt",
        "TDEPMAt2",
        "OIBDt",
        "OIBDt2",
        "EDEPBUt",
        "EDEPBUt2",
        "dourt",
        "TLt",
        "FIt",
        "FEt",
        "dgnp",
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
    TDEPMAt = xt["TDEPMA"]
    OIBDt = xt["OIBD"]
    FIt = xt["FI"]
    FEt = xt["FE"]
    EDEPBUt = xt["EDEPBU"]
    ZPFt = xt["ZPF"]
    TLt = xt["TL"]
    # Assemble a dictionary of all potential features
    all_features = {
        "PALLOt_1": xt_1["PALLO"],
        "ZPFt": ZPFt,
        "TDEPMAt": TDEPMAt,
        "TDEPMAt2": TDEPMAt**2,
        "OIBDt": OIBDt,
        "OIBDt2": OIBDt**2,
        "EDEPBUt": EDEPBUt,
        "EDEPBUt2": EDEPBUt**2,
        "dourt": dOURt,
        "TLt": TLt,
        "FIt": FIt,
        "FEt": FEt,
        "dgnp": xt_1["dgnp"],
        "FAAB": xt_1["FAAB"],
        "Public": xt_1["Public"],
        "ruralare": xt_1["ruralare"],
        "largcity": xt_1["largcity"],
        "market": xt_1["market"],
        "marketw": xt_1["marketw"],
    }
    ## 4. Data Preparation
    X = assemble_tensor(all_features, FEATURES)
    Y = tf.ones_like(xt["OTA"], dtype=tf.float32)
    Y = tf.where(xt["OTA"] > 0, 2, tf.where(xt["OTA"] < 0, 0, 1))
    Y = tf.cast(Y, tf.int32)

    Y = tf.reshape(Y, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    model = MultinomialOrdinalIRLS(
        n_features=X_train.shape[1],
        n_classes=3,
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

    bias, weight = model.get_weights()

    print("\nEstimated Coefficients:")
    print(f"Bias (Intercept): {bias}")
    for i, feature in enumerate(FEATURES):
        print(f"{feature}: {weight[i][0]}")

    print(model.train_loss_tracker.result())
    print(model.train_accuracy_tracker.result())
    print(model.val_loss_tracker.result())
    print(model.val_accuracy_tracker.result())

    result = {
        "coefficients": {
            "Intercept1": float(bias[0]),
            "Intercept2": float(bias[1]),
            **{feature: float(weight[i][0]) for i, feature in enumerate(FEATURES)},
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
