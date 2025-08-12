# Create simulator
import csv
import datetime
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data.macro_var_1998_2010 import MACRO_VAR
from theoretical.simple_theoretical import SimulatorEngine
from utils.data_loader import create_firm_data_tensors
from visualization.visualize_simulation import main as generate_ratio_plots
from visualization.visualize_simulation import plot_firm_state_scatter

NUM_FIRMS = 30000
NUM_YEARS_TO_SIMULATE = 10
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_simulator_engine(model_instance, num_firms):
    """
    Build the SimulatorEngine by passing dummy input tensors.

    Args:
        model_instance: SimulatorEngine instance.
        num_firms (int): Number of firms (defines input shape).
    """
    print(f"Building engine for {num_firms} firms...")

    # Define input tensor shape: (num_firms, 1)
    tensor_shape = (num_firms, 1)

    # List of all feature names expected by the model
    feature_names = [
        "CA",
        "MA",
        "BU",
        "OFA",
        "CL",
        "LL",
        "ASD",
        "OUR",
        "SC",
        "RR",
        "URE",
        "PFt",
        "PFt_1",
        "PFt_2",
        "PFt_3",
        "PFt_4",
        "PFt_5",
        "OIBD",
        "EDEPMA",
        "EDEPBU",
        "OIAD",
        "FI",
        "FE",
        "EBA",
        "TDEPMA",
        "OA",
        "ZPF",
        "PALLO",
        "EBT",
        "TL",
        "NI",
        "OTA",
        "TDEPBU",
        "OLT_1T",
        "TAX",
        "ROT",
        "FTAX",
        "OLT",
        "NBI",
        "MTDM",
        "MCASH",
        "IMA",
        "IBU",
        "CMA",
        "DCA",
        "DOFA",
        "DCL",
        "DLL",
        "DOUR",
        "DSC",
        "DRR",
        "DURE",
        "DIV",
        "CASHFL",
        "SMA",
        "MPA",
        "realr",
        "dgnp",
        "Public",
        "FAAB",
        "ruralare",
        "largcity",
        "market",
        "marketw",
        "MTDMt_1",
        "TDEPMAt_1",
        "MPAt_1",
        "PALLOt_1",
        "MCASHt_1",
        "CASHFLt_1",
    ]

    # Create dummy input dictionaries for t_1
    dummy_input_t_1 = {
        name: tf.zeros(tensor_shape, dtype=tf.float32) for name in feature_names
    }

    # Call the model once with dummy data to trigger build
    _ = model_instance(dummy_input_t_1)

    print("Engine built successfully.")


def convert_output_to_input(result, previous_input_t_1):
    """Converts the output of a simulation year into the input for the next year."""
    new_input = {}

    # Map state variables
    new_input["CA"] = result["CAt"]
    new_input["MA"] = result["MAt"]
    new_input["BU"] = result["BUt"]
    new_input["OFA"] = result["OFAt"]
    new_input["CL"] = result["CLt"]
    new_input["LL"] = result["LLt"]
    new_input["ASD"] = result["ASDt"]
    new_input["OUR"] = result["OURt"]
    new_input["SC"] = result["SCt"]
    new_input["RR"] = result["RRt"]
    new_input["URE"] = result["UREt"]

    # Map periodical reserves
    new_input["PFt"] = result["PFt_t"]
    new_input["PFt_1"] = result["PFt_t_1"]
    new_input["PFt_2"] = result["PFt_t_2"]
    new_input["PFt_3"] = result["PFt_t_3"]
    new_input["PFt_4"] = result["PFt_t_4"]
    new_input["PFt_5"] = result["PFt_t_5"]

    # Map income statement items
    new_input["OIBD"] = result["OIBDt"]
    new_input["EDEPMA"] = result["EDEPMAt"]
    new_input["EDEPBU"] = result["EDEPBUt"]
    new_input["OIAD"] = new_input["OIBD"] - new_input["EDEPMA"] - new_input["EDEPBU"]
    new_input["FI"] = result["FIt"]
    new_input["FE"] = result["FEt"]
    new_input["EBA"] = new_input["OIAD"] + new_input["FI"] - new_input["FE"]
    new_input["TDEPMA"] = result["TDEPMAt"]
    new_input["OA"] = result["OAt"]
    new_input["ZPF"] = result["ZPFt"]
    new_input["PALLO"] = result["PALLOt"]
    new_input["EBT"] = result["EBTt"]
    new_input["TL"] = result["TLt"]
    new_input["NI"] = result["NIt"]
    new_input["OTA"] = result["OTAt"]
    new_input["TDEPBU"] = result["TDEPBUt"]
    new_input["OLT_1T"] = previous_input_t_1["OLT"]
    new_input["TAX"] = result["TAXt"]
    new_input["ROT"] = result["ROTt"]
    new_input["FTAX"] = result["FTAXt"]
    new_input["OLT"] = result["OLt"]
    new_input["NBI"] = result["NBIt"]

    # Map flow variables
    new_input["MTDM"] = result["MTDMt"]
    new_input["MCASH"] = result["MCASHt"]
    new_input["IMA"] = result["IMAt"]
    new_input["IBU"] = result["IBUt"]
    new_input["CMA"] = result["CMAt"]
    new_input["DCA"] = result["dCAt"]
    new_input["DOFA"] = result["dOFAt"]
    new_input["DCL"] = result["dCLt"]
    new_input["DLL"] = result["dLLt"]
    new_input["DOUR"] = result["dOURt"]
    new_input["DSC"] = result["dSCt"]
    new_input["DRR"] = result["dRRt"]
    new_input["DURE"] = result["UREt"] - previous_input_t_1["URE"]
    new_input["DIV"] = result["DIVt"]
    new_input["CASHFL"] = result["CASHFLt"]
    new_input["SMA"] = result["SMAt"]
    new_input["MPA"] = result["MPAt"]

    # Carry over environmental variables
    new_input["realr"] = previous_input_t_1["realr"]
    new_input["dgnp"] = previous_input_t_1["dgnp"]
    new_input["Public"] = previous_input_t_1["Public"]
    new_input["FAAB"] = previous_input_t_1["FAAB"]
    new_input["ruralare"] = previous_input_t_1["ruralare"]
    new_input["largcity"] = previous_input_t_1["largcity"]
    new_input["market"] = previous_input_t_1["market"]
    new_input["marketw"] = previous_input_t_1["marketw"]

    new_input["MTDMt_1"] = previous_input_t_1["MTDM"]
    new_input["TDEPMAt_1"] = previous_input_t_1["TDEPMA"]
    new_input["MPAt_1"] = previous_input_t_1["MPA"]
    new_input["PALLOt_1"] = previous_input_t_1["PALLO"]
    new_input["MCASHt_1"] = previous_input_t_1["MCASH"]
    new_input["CASHFLt_1"] = previous_input_t_1["CASHFL"]
    new_input["GC"] = result["GCt"]

    return new_input


def get_realr(year):
    ten_yr_yield = MACRO_VAR[str(year)]["ten_yr_yield"]
    realr = MACRO_VAR[str(year)]["realr"]
    return (ten_yr_yield / 100 - realr / 100) / (1 + realr / 100)


def get_dgnp(year):
    gnp = MACRO_VAR[str(year)]["GDP"] * MACRO_VAR[str(year)]["fx"] * 1000
    gnp_1 = MACRO_VAR[str(year - 1)]["GDP"] * MACRO_VAR[str(year - 1)]["fx"] * 1000
    return gnp - gnp_1


def main():
    # Create a timestamped directory for this simulation run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_dir = Path(f"data/simulation_outputs/{timestamp}")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"CSV simulation outputs will be saved in: {csv_output_dir}")

    all_features_list = []
    all_targets_list = []

    simulator = SimulatorEngine(NUM_FIRMS)

    build_simulator_engine(simulator, NUM_FIRMS)
    simulator.load_weights_from_cfg()

    # Initial data load
    xt_1 = create_firm_data_tensors(2000, NUM_FIRMS)
    xt_1["realr"] = tf.constant(get_realr(2000), shape=(NUM_FIRMS, 1), dtype=tf.float32)
    xt_1["dgnp"] = tf.constant(get_dgnp(2000), shape=(NUM_FIRMS, 1), dtype=tf.float32)

    xt_2 = create_firm_data_tensors(1999, NUM_FIRMS)

    xt_1["MTDMt_1"] = xt_2["MTDM"]
    xt_1["TDEPMAt_1"] = xt_2["TDEPMA"]
    xt_1["MPAt_1"] = xt_2["MPA"]
    xt_1["PALLOt_1"] = xt_2["PALLO"]
    xt_1["MCASHt_1"] = xt_2["MCASH"]
    xt_1["CASHFLt_1"] = xt_2["CASHFL"]
    for year in range(NUM_YEARS_TO_SIMULATE):
        current_year = 2001 + year
        print(f"Simulating year t+{year} ({current_year})...")

        y_hat = simulator(xt_1)

        # Convert tensors to a DataFrame for the current year's output
        output_data = {key: np.squeeze(value.numpy()) for key, value in y_hat.items()}
        df_year = pd.DataFrame(output_data)

        # Save to CSV
        csv_path = csv_output_dir / f"{current_year}.csv"
        df_year.to_csv(csv_path, index=False)
        print(f"    Saved simulation data for year {current_year} to {csv_path}")

        xt_1["year"] = tf.constant(current_year, shape=(NUM_FIRMS, 1), dtype=tf.int32)

        # Update marketw based on the new asset values
        total_assets_all_firms = (
            y_hat["CAt"] + y_hat["MAt"] + y_hat["BUt"] + y_hat["OFAt"]
        )
        total_market_assets = tf.reduce_sum(total_assets_all_firms)
        new_marketw = total_assets_all_firms / total_market_assets

        xt_1_hat = convert_output_to_input(y_hat, xt_1)
        xt_1["marketw"] = new_marketw
        xt_1["realr"] = tf.constant(
            get_realr(current_year), shape=(NUM_FIRMS, 1), dtype=tf.float32
        )
        xt_1["dgnp"] = tf.constant(
            get_dgnp(current_year), shape=(NUM_FIRMS, 1), dtype=tf.float32
        )

        if year >= 7:
            all_targets_list.append(xt_1)
            all_features_list.append(xt_1_hat)
        xt_1 = xt_1_hat

        total_assets = (
            y_hat["CAt"].numpy()[0]
            + y_hat["MAt"].numpy()[0]
            + y_hat["BUt"].numpy()[0]
            + y_hat["OFAt"].numpy()[0]
        )[0]

        total_liabilities_and_equity = (
            y_hat["CLt"].numpy()[0]
            + y_hat["LLt"].numpy()[0]
            + y_hat["ASDt"].numpy()[0]
            + y_hat["OURt"].numpy()[0]
            + y_hat["SCt"].numpy()[0]
            + y_hat["RRt"].numpy()[0]
            + y_hat["UREt"].numpy()[0]
            + y_hat["PFt_t"].numpy()[0]
            + y_hat["PFt_t_1"].numpy()[0]
            + y_hat["PFt_t_2"].numpy()[0]
            + y_hat["PFt_t_3"].numpy()[0]
            + y_hat["PFt_t_4"].numpy()[0]
            + y_hat["PFt_t_5"].numpy()[0]
        )[0]
        print(f"    Sample Total Assets: {total_assets}")
        print(f"    Sample Total Liabilities: {total_liabilities_and_equity}")

        # Calculate balance metrics for monitoring (balance validation now happens in TF model)
        absolute_diff = abs(total_assets - total_liabilities_and_equity)
        relative_diff = absolute_diff / abs(total_assets) if total_assets != 0 else 0
        print(
            f"    Balance difference: {absolute_diff:.2f} (relative: {relative_diff:.2e})"
        )

    all_features_df = pd.DataFrame()
    for x in all_targets_list:
        df = pd.DataFrame({key: np.squeeze(value) for key, value in x.items()})
        all_features_df = pd.concat([all_features_df, df], axis=0)
    all_targets_df = pd.DataFrame()

    for y in all_features_list:
        df = pd.DataFrame({key: np.squeeze(value) for key, value in y.items()})
        all_targets_df = pd.concat([all_targets_df, df], axis=0)

    all_features_df.reset_index(drop=True, inplace=True)
    all_targets_df.reset_index(drop=True, inplace=True)

    output_dir = "./data/simulation_outputs/synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    train_to_save = {
        col: all_features_df[col].to_numpy(dtype=np.float32)
        for col in all_features_df.columns
    }
    target_to_save = {
        col: all_targets_df[col].to_numpy(dtype=np.float32)
        for col in all_targets_df.columns
    }

    np.savez_compressed(f"{output_dir}/train.npz", **train_to_save)
    np.savez_compressed(f"{output_dir}/test.npz", **target_to_save)

    print("\nGenerating financial ratio plots...")
    generate_ratio_plots(str(csv_output_dir))
    print("\nGenerating firm state scatter plots...")
    plot_firm_state_scatter(str(csv_output_dir))


if __name__ == "__main__":
    main()
