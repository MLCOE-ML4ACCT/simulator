# Create simulator
import json
import math
from pathlib import Path
from typing import OrderedDict

import tensorflow as tf

from theoretical.simple_theoretical import SimulatorEngine
from utils.data_loader import load_data_for_year


def main():
    NUM_FIRMS = 10
    BASE_YEAR = 1999
    RANTA10 = 0.045
    INFLATION_RATE = 0.006  # Example inflation rate
    GNP1998 = 265.02e9
    GNP1999 = 273.92e9

    realr = (RANTA10 - INFLATION_RATE) / (1 + INFLATION_RATE)

    data_path = Path(__file__).parent / "data" / "data_table25.json"
    with open(data_path) as f:
        data = json.load(f)

    year_str = str(BASE_YEAR)
    if year_str not in data:
        raise ValueError(f"Data for year {BASE_YEAR} not found in {data_path}")

    year_data = data[year_str]
    assets = year_data["Assets"]
    liabilities = year_data["Liabilities"]
    income = year_data["Income statement"]
    flow = year_data["Flow variabels"]

    # Helper to convert to float tensor
    def to_tensor(value):
        return tf.constant(float(value), shape=(NUM_FIRMS, 1), dtype=tf.float32)

    base_year_input = OrderedDict(
        {
            "CA": to_tensor(assets["CA"]),
            "MA": to_tensor(assets["MA"]),
            "BU": to_tensor(assets["BU"]),
            "OFA": to_tensor(assets["OFA"]),
            "CL": to_tensor(liabilities["CL"]),
            "LL": to_tensor(liabilities["LL"]),
            "SC": to_tensor(liabilities["SC"]),
            "ASD": to_tensor(liabilities["ASD"]),
            "OUR": to_tensor(liabilities["OUR"]),
            "RR": to_tensor(liabilities["RR"]),
            "URE": to_tensor(liabilities["URE"]),
            "PFt": to_tensor(liabilities["PFt"]),
            "PFt_1": to_tensor(liabilities["PFt-1t"]),
            "PFt_2": to_tensor(liabilities["PFt-2t"]),
            "PFt_3": to_tensor(liabilities["PFt-3t"]),
            "PFt_4": to_tensor(liabilities["PFt-4t"]),
            "PFt_5": to_tensor(liabilities["PFt-5t"]),
            "OL": to_tensor(income["Olt"]),
            "DIV": to_tensor(flow["DIV"]),
            "realr": tf.constant(realr, shape=(NUM_FIRMS, 1), dtype=tf.float32),
            "dgnp": tf.constant(0.0, shape=(NUM_FIRMS, 1), dtype=tf.float32),
            "Public": tf.cast(
                tf.random.uniform((NUM_FIRMS, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "FAAB": tf.cast(
                tf.random.uniform((NUM_FIRMS, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "ruralare": tf.cast(
                tf.random.uniform((NUM_FIRMS, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "largcity": tf.cast(
                tf.random.uniform((NUM_FIRMS, 1), minval=0, maxval=1, dtype=tf.float32)
                < 0.5,
                tf.float32,
            ),
            "market": tf.constant(
                1 / NUM_FIRMS, shape=(NUM_FIRMS, 1), dtype=tf.float32
            ),
            "marketw": tf.constant(
                1 / NUM_FIRMS, shape=(NUM_FIRMS, 1), dtype=tf.float32
            ),
        }
    )

    # Initialize the simulator engine
    simulator = SimulatorEngine(NUM_FIRMS)

    # Run one year of simulation (currently returns placeholder)
    result = simulator.run_one_year(base_year_input)

    print(f"\nSimulation result: {result}")


if __name__ == "__main__":
    main()
