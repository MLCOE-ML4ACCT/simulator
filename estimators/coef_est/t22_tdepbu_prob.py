import json
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Assuming these are your custom modules
from estimators.stat_model.tobit import TobitIRLS
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t22_tdepbu_prob.json"

    # Feature & Model Parameters
    FEATURES = [
        "sumcasht_1",
        "diffcasht_1",
        "EDEPMAt",
        "EDEPMAt2",
        "SMAt",
        "I_MAt",
        "BUt_1",
        "BUt_12",
        "dcat",
        "dcat2",
        "dclt",
        "ddmpat_1",
        "ddmpat_12",
        "ddmpat_13",
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
    EDEPMAt = xt["EDEPMAt"]
    SMAt = xt["SMAt"]
    IMAt = xt["IMAt"]
    dCAt = xt["DCA"]
    dCLt = xt["DCL"]
    # Assemble a dictionary of all potential features
    all_features = {
        "sumcasht_1": sumcasht_1,
        "diffcasht_1": diffcasht_1,
        "EDEPMAt": EDEPMAt,
        "EDEPMAt2": EDEPMAt**2,
        "SMAt": SMAt,
        "I_MAt": IMAt,
        "BUt_1": xt_1["BU"],
        "BUt_12": xt_1["BU"] ** 2,
        "dcat": dCAt,
        "dcat2": dCAt**2,
        "dclt": dCLt,
        "ddmpat_1": ddMPAt_1,
        "ddmpat_12": ddMPAt_1**2,
        "ddmpat_13": ddMPAt_1**3,
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
    Y = xt["TDEPBU"]

    Y = tf.reshape(Y, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    print(X_train.shape, y_train.shape)

    model = TobitIRLS(
        max_iterations=50,
    )

    model.fit(
        X_train,
        y_train,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    weight, intercept, sig = model.get_weights()
    weight = weight.flatten()
    intercept = intercept[0]
    sig = sig[0]
    print("\nEstimated Coefficients:")
    print(f"Bias (Intercept): {intercept}")
    for i, feature in enumerate(FEATURES):
        print(f"{feature}: {weight[i]}")

    result = {
        "coefficients": {
            "Intercept": float(intercept),
            "LogScale": float(math.log(sig)),
            "Scale": float(sig),
            **{FEATURES[i]: float(coef) for i, coef in enumerate(weight)},
        },
        "model_info": {
            "n_features": len(FEATURES),
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "n_outliers": len(FEATURES),
            "train_loss": float(model.train_loss_tracker.result()),
            "val_loss": float(model.val_loss_tracker.result()),
        },
    }

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_path}")
