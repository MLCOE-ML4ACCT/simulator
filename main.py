# Create simulator
import json
import math
from pathlib import Path

import tensorflow as tf

from theoretical.simple_theoretical import SimulatorEngine


def create_firm_data_tensors(base_year, realr, num_firms, data_file_path=None):
    """Create Dict of firm data tensors for simulation.

    Args:
        base_year: Year to load data for (e.g., 1999)
        realr: Real interest rate value
        num_firms: Number of firms to simulate
        data_file_path: Optional path to data file. If None, uses default path.

    Returns:
        Dict containing TensorFlow tensors for all firm variables
    """
    # Set default data path if not provided
    if data_file_path is None:
        data_file_path = Path(__file__).parent / "data" / "data_table25.json"

    # Load and parse data file
    with open(data_file_path) as f:
        data = json.load(f)

    year_str = str(base_year)
    if year_str not in data:
        raise ValueError(f"Data for year {base_year} not found in {data_file_path}")

    year_data = data[year_str]
    assets = year_data["Assets"]
    liabilities = year_data["Liabilities"]
    income = year_data["Income statement"]
    flow = year_data["Flow variabels"]

    # Helper to convert to float tensor
    def to_tensor(value):
        return tf.constant(float(value), shape=(num_firms, 1), dtype=tf.float32)

    return dict(
        {
            "CA": to_tensor(assets["CA"]),
            "MA": to_tensor(assets["MA"]),
            "BU": to_tensor(assets["BU"]),
            "OFA": to_tensor(assets["OFA"]),
            "CL": to_tensor(liabilities["CL"]),
            "LL": to_tensor(liabilities["LL"]),
            "ASD": to_tensor(liabilities["ASD"]),
            "OUR": to_tensor(liabilities["OUR"]),
            "SC": to_tensor(liabilities["SC"]),
            "RR": to_tensor(liabilities["RR"]),
            "URE": to_tensor(liabilities["URE"]),
            "PFt": to_tensor(liabilities["PFt"]),
            "PFt_1": to_tensor(liabilities["PFt-1t"]),
            "PFt_2": to_tensor(liabilities["PFt-2t"]),
            "PFt_3": to_tensor(liabilities["PFt-3t"]),
            "PFt_4": to_tensor(liabilities["PFt-4t"]),
            "PFt_5": to_tensor(liabilities["PFt-5t"]),
            "OIBD": to_tensor(income["OIBD"]),
            "EDEPMA": to_tensor(income["EDEPma"]),
            "EDEPBU": to_tensor(income["EDEPbu"]),
            "OIAD": to_tensor(income["OIAD"]),
            "FI": to_tensor(income["FI"]),
            "FE": to_tensor(income["FE"]),
            "EBA": to_tensor(income["EBA"]),
            "TDEPMA": to_tensor(income["TDEPma-EDEPma"]) + to_tensor(income["EDEPma"]),
            "OA": to_tensor(income["OA"]),
            "ZPF": to_tensor(income["zPF"]),
            "PALLO": to_tensor(income["Pallo"]),
            "EBT": to_tensor(income["EBT"]),
            "TL": to_tensor(income["TL"]),
            "NI": to_tensor(income["NI"]),
            "OTA": to_tensor(income["OTA"]),
            "TDEPBU": to_tensor(income["TDEPbu"]),
            "OLT_1T": to_tensor(income["Olt-1t"]),
            "TAX": to_tensor(income["TAX"]),
            "ROT": to_tensor(income["ROT"]),
            "FTAX": to_tensor(income["FTAX"]),
            "OLT": to_tensor(income["Olt"]),
            "NBI": to_tensor(income["NBI"]),
            "MTDM": to_tensor(flow["MTDM"]),
            "MCASH": to_tensor(flow["MCASH"]),
            "IMA": to_tensor(flow["I_ma"]),
            "IBU": to_tensor(flow["I_bu"]),
            "CMA": to_tensor(flow["CMA"]),
            "DCA": to_tensor(flow["dCA"]),
            "DOFA": to_tensor(flow["dOFA"]),
            "DCL": to_tensor(flow["dCL"]),
            "DLL": to_tensor(flow["dLL"]),
            "DOUR": to_tensor(flow["dOUR"]),
            "DSC": to_tensor(flow["dSC"]),
            "DRR": to_tensor(flow["dRR"]),
            "DURE": to_tensor(flow["dURE"]),
            "DIV": to_tensor(flow["DIV"]),
            "CASHFL": to_tensor(flow["CASHFL"]),
            "SMA": to_tensor(flow["SMA"]),
            "MPA": to_tensor(flow["MPA"]),
            "realr": tf.constant(realr, shape=(num_firms, 1), dtype=tf.float32),
            "dgnp": tf.constant(0.0, shape=(num_firms, 1), dtype=tf.float32),
            "Public": tf.cast(
                tf.random.uniform((num_firms, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "FAAB": tf.cast(
                tf.random.uniform((num_firms, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "ruralare": tf.cast(
                tf.random.uniform((num_firms, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "largcity": tf.cast(
                tf.random.uniform((num_firms, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "market": tf.constant(
                1 / num_firms, shape=(num_firms, 1), dtype=tf.float32
            ),
            "marketw": tf.constant(
                1 / num_firms, shape=(num_firms, 1), dtype=tf.float32
            ),
        }
    )


def main():
    NUM_FIRMS = 3
    RANTA10 = 0.045
    INFLATION_RATE = 0.006  # Example inflation rate
    GNP1998 = 265.02e9
    GNP1999 = 273.92e9

    realr = (RANTA10 - INFLATION_RATE) / (1 + INFLATION_RATE)

    input_t_1 = create_firm_data_tensors(2001, realr, NUM_FIRMS)
    input_t_2 = create_firm_data_tensors(2000, realr, NUM_FIRMS)
    input_t_3 = create_firm_data_tensors(1999, realr, NUM_FIRMS)
    # Initialize the simulator engine
    simulator = SimulatorEngine(NUM_FIRMS)

    # Run one year of simulation (currently returns placeholder)
    result = simulator.run_one_year(input_t_1, input_t_2)

    print("complete")
    # print(f"\nSimulation result: {result}")


if __name__ == "__main__":
    main()
