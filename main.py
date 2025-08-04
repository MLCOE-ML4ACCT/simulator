# Create simulator
import csv
import datetime
import json
import math
import os
import random
from pathlib import Path

import tensorflow as tf

from theoretical.simple_theoretical import SimulatorEngine
from utils.data_loader import create_firm_data_tensors
from visualization.financial_ratios import FinancialRatioCalculator


def convert_output_to_input(result, previous_input_t_1, realr, num_firms):
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
    new_input["NI"] = result["NBIt"]  # Using NBI for NI as per model logic
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

    return new_input


def save_results_to_csv(results, timestamp, num_firms):
    """Saves the simulation results to CSV files."""
    output_dir = Path(__file__).parent / "data" / "simulation_outputs" / f"{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        year = 2001 + i
        filename = output_dir / f"{year}.csv"

        # Prepare data for CSV writing
        header = list(result.keys())
        # Create a list of dictionaries, where each dictionary represents a firm
        rows = []
        for firm_index in range(num_firms):
            row = {key: result[key].numpy().flatten()[firm_index] for key in header}
            rows.append(row)

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved simulation results for year {year} to {filename}")


def main():
    # Seeding for reproducibility
    SEED = random.randint(0, 100000)
    # SEED = 62011
    # SEED = 42
    tf.random.set_seed(SEED)
    # np.random.seed(SEED) # Uncomment if you use numpy.random directly
    random.seed(SEED)
    print(f"Running simulation with seed: {SEED}")

    NUM_FIRMS = 3000
    NUM_YEARS_TO_SIMULATE = 10
    RANTA10 = 0.045
    INFLATION_RATE = 0.006  # Example inflation rate

    realr = (RANTA10 - INFLATION_RATE) / (1 + INFLATION_RATE)

    # Initial data load
    input_t_1 = create_firm_data_tensors(2000, realr, NUM_FIRMS)
    input_t_2 = create_firm_data_tensors(1999, realr, NUM_FIRMS)

    # Initialize the simulator engine
    simulator = SimulatorEngine(NUM_FIRMS)

    simulation_results = []
    relative_diff_ls = []
    # Main simulation loop
    for year in range(NUM_YEARS_TO_SIMULATE):
        print(f"Simulating year {2001 + year}...")
        result = simulator.run_one_year(input_t_1, input_t_2)
        simulation_results.append(result)

        # Calculate and print financial ratios
        ratio_calculator = FinancialRatioCalculator(result)
        ratios = ratio_calculator.calculate_all_ratios(realr)
        print("    Financial Ratios (Mean):")
        for ratio_name, ratio_value in ratios.items():
            mean_ratio = tf.reduce_mean(ratio_value).numpy()
            print(f"    - {ratio_name}: {mean_ratio:.3f}")

        # Prepare inputs for the next iteration
        new_input_t_1 = convert_output_to_input(result, input_t_1, realr, NUM_FIRMS)
        input_t_2 = input_t_1
        input_t_1 = new_input_t_1

        total_assets = (
            result["CAt"].numpy()[0]
            + result["MAt"].numpy()[0]
            + result["BUt"].numpy()[0]
            + result["OFAt"].numpy()[0]
        )[0]
        total_liabilities_and_equity = (
            result["CLt"].numpy()[0]
            + result["LLt"].numpy()[0]
            + result["ASDt"].numpy()[0]
            + result["OURt"].numpy()[0]
            + result["SCt"].numpy()[0]
            + result["RRt"].numpy()[0]
            + result["UREt"].numpy()[0]
            + result["PFt_t"].numpy()[0]
            + result["PFt_t_1"].numpy()[0]
            + result["PFt_t_2"].numpy()[0]
            + result["PFt_t_3"].numpy()[0]
            + result["PFt_t_4"].numpy()[0]
            + result["PFt_t_5"].numpy()[0]
        )[0]
        print(f"    Sample Total Assets: {total_assets}")
        print(f"    Sample Total Liabilities: {total_liabilities_and_equity}")

        # Calculate balance metrics
        absolute_diff = abs(total_assets - total_liabilities_and_equity)
        relative_diff = absolute_diff / abs(total_assets) if total_assets != 0 else 0
        relative_diff_ls.append(absolute_diff)
        print(
            f"    Balance difference: {absolute_diff:.2f} (relative: {relative_diff:.2e})"
        )

        # Define acceptability thresholds
        REL_TOLERANCE = 1e-6  # 0.0001% relative error
        ABS_TOLERANCE = max(1.0, abs(total_assets) * 1e-6)  # Dynamic absolute tolerance

        is_acceptable = math.isclose(
            total_assets,
            total_liabilities_and_equity,
            rel_tol=REL_TOLERANCE,
            abs_tol=ABS_TOLERANCE,
        )

        print(f"    Balance acceptable: {is_acceptable}")
        if not is_acceptable:
            print(
                f"    WARNING: Balance exceeds tolerance (rel_tol={REL_TOLERANCE}, abs_tol={ABS_TOLERANCE:.2f})"
            )

        # print out how many value in the tenor are negative
        # need to check for MA, CMA, BU, OFA, CA, RR, ASD, PF, OUR, LL, CL
        # count how many values are negative in each of these tensors and print out
        negative_counts = {
            "MA": tf.reduce_sum(tf.cast(result["MAt"] < 0, tf.int32)).numpy(),
            "CMA": tf.reduce_sum(tf.cast(result["CMAt"] < 0, tf.int32)).numpy(),
            "BU": tf.reduce_sum(tf.cast(result["BUt"] < 0, tf.int32)).numpy(),
            "OFA": tf.reduce_sum(tf.cast(result["OFAt"] < 0, tf.int32)).numpy(),
            "CA": tf.reduce_sum(tf.cast(result["CAt"] < 0, tf.int32)).numpy(),
            "RR": tf.reduce_sum(tf.cast(result["RRt"] < 0, tf.int32)).numpy(),
            "ASD": tf.reduce_sum(tf.cast(result["ASDt"] < 0, tf.int32)).numpy(),
            "PF": tf.reduce_sum(tf.cast(result["PFt_t"].numpy() < 0, tf.int32)).numpy(),
            "OUR": tf.reduce_sum(tf.cast(result["OURt"] < 0, tf.int32)).numpy(),
            "LL": tf.reduce_sum(tf.cast(result["LLt"] < 0, tf.int32)).numpy(),
            "CL": tf.reduce_sum(tf.cast(result["CLt"] < 0, tf.int32)).numpy(),
        }
        print("Negative counts in tensors:")
        for key, count in negative_counts.items():
            print(f"    {key}: {count}")
    print(relative_diff_ls)
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_to_csv(simulation_results, timestamp, NUM_FIRMS)

    # Plot results
    from visualization.plot import plot_simulation_results

    plot_simulation_results(timestamp, realr)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
