import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Assuming these are your custom modules
from estimators.stat_model.huber_robust import HuberSchweppeIRLS
from utils.data_loader import assemble_tensor, unwrap_inputs

if __name__ == "__main__":
    ## 1. Configuration
    # File Paths
    TRAIN_DATA_PATH = "data/simulation_outputs/synthetic_data/train.npz"
    TEST_DATA_PATH = "data/simulation_outputs/synthetic_data/test.npz"
    OUTPUT_DIR = "estimators/coef"
    OUTPUT_FILENAME = "t1_edepma_level.json"

    # Feature & Model Parameters
    FEATURES = [
        "sumcasht_1",
        "diffcasht_1",
        "TDEPMAt_1",
        "MAt_1",
        "I_MAt_1",
        "I_MAt_12",
        "EDEPBUt_1",
        "EDEPBUt_12",
        "ddmtdmt_1",
        "ddmtdmt_12",
        "dcat_1",
        "ddmpat_1",
        "ddmpat_12",
        "dclt_1",
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
    mask = Y > 0
    X = tf.boolean_mask(X, mask)
    Y = tf.boolean_mask(Y, mask)
    Y = tf.reshape(Y, (-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), Y.numpy(), test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )

    model = HuberSchweppeIRLS(
        n_features=len(FEATURES),
        max_iterations=50,
        tolerance=1e-6,
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

    intercept, coefficients = model.get_coefficients()

    print("\nEstimated Coefficients:")
    print(f"Intercept: {intercept:.6f}")
    for i, coef in enumerate(coefficients):
        print(f"Feature_{i+1}: {coef:.6f}")

    ## 6. Model Performance
    print(f"\nModel Performance:")
    print(f"Train Loss: {model.train_loss_tracker.result():.4f}")
    print(f"Train MAE: {model.train_mae_tracker.result():.4f}")
    print(f"Val Loss: {model.val_loss_tracker.result():.4f}")
    print(f"Val MAE: {model.val_mae_tracker.result():.4f}")

    ## 7. Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate R-squared for additional evaluation
    ss_res_train = np.sum((y_train - y_pred_train) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1 - (ss_res_train / ss_tot_train)

    ss_res_test = np.sum((y_test - y_pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - (ss_res_test / ss_tot_test)

    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")

    ## 8. Save Results
    result = {
        "coefficients": {
            "Intercept": float(intercept),
            **{FEATURES[i]: float(coef) for i, coef in enumerate(coefficients)},
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
            "train_r2": float(r2_train),
            "test_r2": float(r2_test),
        },
    }

    # Save results to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_path}")
