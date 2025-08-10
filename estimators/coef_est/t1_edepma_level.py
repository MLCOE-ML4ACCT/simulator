import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Assuming these are your custom modules
from estimators.layers.edepma_layer import EDEPMALayer
from estimators.stat_model.huber_schweppe import huber_schweppe_estimator
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "data/coefficient_estimates"
    OUTPUT_FILENAME = "t1_edepma_level.json"

    # Feature & Model Parameters
    FEATURES = [
        "diffcasht_1",
        "TDEPMAt_1",
        "MAt_1",
        "I_MAt_1",
        "FAAB",
        "Public",
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

    # Assemble a dictionary of all potential features
    all_features = {
        "sumcasht_1": sumcasht_1,
        "diffcasht_1": diffcasht_1,
        "TDEPMAt_1": xt_1["TDEPMA"],
        "MAt_1": xt_1["MA"],
        "I_MAt_1": xt_1["IMA"],
        "I_MAt_12": xt_1["IMA"] ** 2,
        "EDEPBUt_1": xt_1["EDEPBU"],
        "EDEPBUt_12": xt_1["EDEPBU"] ** 2,
        "ddmtdmt_1": ddMTDMt_1,
        "ddmtdmt_12": ddMTDMt_1**2,
        "dcat_1": xt_1["DCA"],
        "ddmpat_1": ddMPAt_1,
        "ddmpat_12": ddMPAt_1**2,
        "dclt_1": xt_1["DCL"],
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
    Y = xt["EDEPMA"]
    Y = tf.reshape(Y, (-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    ## 5. Model Training
    learned_weights = huber_schweppe_estimator(X_train, y_train, X_test, y_test)

    ## 6. Report and Save Results
    print("\nLearned Coefficients (Beta):")
    print(f"Bias (Intercept): {learned_weights[0].numpy()[0]}")
    for i, feature_name in enumerate(FEATURES):
        weight = learned_weights[i + 1].numpy()[0]
        print(f"Weight for Feature '{feature_name}': {weight}")

    # Save coefficients to a JSON file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coefficients = {"Intercept": float(learned_weights[0].numpy()[0])}
    for i, feature_name in enumerate(FEATURES):
        coefficients[feature_name] = float(learned_weights[i + 1].numpy()[0])

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        json.dump(coefficients, f, indent=4)

    print(f"\nCoefficients successfully saved to: {output_path}")
