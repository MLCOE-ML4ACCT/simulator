import json
from pathlib import Path

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


def create_firm_data_tensors(base_year, num_firms, data_file_path=None):
    """Create Dict of firm data tensors for simulation.

    Args:
        base_year: Year to load data for (e.g., 1999)
        num_firms: Number of firms to simulate
        data_file_path: Optional path to data file. If None, uses default path.

    Returns:
        Dict containing TensorFlow tensors for all firm variables
    """
    # Set default data path if not provided
    if data_file_path is None:
        data_file_path = Path(__file__).parent.parent / "data" / "data_table25.json"

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
            "realr": tf.constant(0.0, shape=(num_firms, 1), dtype=tf.float32),
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


def assemble_tensor(ipt_dict, feature_ls):
    feature_tensors = [tf.reshape(ipt_dict[name], (-1, 1)) for name in feature_ls]
    return tf.concat(feature_tensors, axis=1)


def unwrap_inputs(input_dict: dict) -> dict:
    """Unwrap tensor inputs and extract individual variables.

    Args:
        input_dict (dict): A dictionary containing tensor fields matching
            the input signature.

    Returns:
        dict: A dictionary containing all unwrapped variables with their names as keys.
    """
    # unwrap all the input tensors
    unwrapped_inputs = {key: tensor for key, tensor in input_dict.items()}
    # Create a dictionary with all variables for easy access
    variables = {}

    # Assets
    variables.update(
        {
            "CA": unwrapped_inputs["CA"],
            "MA": unwrapped_inputs["MA"],
            "BU": unwrapped_inputs["BU"],
            "OFA": unwrapped_inputs["OFA"],
        }
    )

    # Liabilities & Equity
    variables.update(
        {
            "CL": unwrapped_inputs["CL"],
            "LL": unwrapped_inputs["LL"],
            "ASD": unwrapped_inputs["ASD"],
            "OUR": unwrapped_inputs["OUR"],
            "SC": unwrapped_inputs["SC"],
            "RR": unwrapped_inputs["RR"],
            "URE": unwrapped_inputs["URE"],
        }
    )

    # Periodical Reserves (current and lagged)
    variables.update(
        {
            "PFt": unwrapped_inputs["PFt"],
            "PFt_1": unwrapped_inputs["PFt_1"],
            "PFt_2": unwrapped_inputs["PFt_2"],
            "PFt_3": unwrapped_inputs["PFt_3"],
            "PFt_4": unwrapped_inputs["PFt_4"],
            "PFt_5": unwrapped_inputs["PFt_5"],
        }
    )

    # Income Statement Items
    variables.update(
        {
            "OIBD": unwrapped_inputs["OIBD"],
            "EDEPMA": unwrapped_inputs["EDEPMA"],
            "EDEPBU": unwrapped_inputs["EDEPBU"],
            "OIAD": unwrapped_inputs["OIAD"],
            "FI": unwrapped_inputs["FI"],
            "FE": unwrapped_inputs["FE"],
            "EBA": unwrapped_inputs["EBA"],
            "TDEPMA": unwrapped_inputs["TDEPMA"],
            "OA": unwrapped_inputs["OA"],
            "ZPF": unwrapped_inputs["ZPF"],
            "PALLO": unwrapped_inputs["PALLO"],
            "EBT": unwrapped_inputs["EBT"],
            "TL": unwrapped_inputs["TL"],
            "NI": unwrapped_inputs["NI"],
            "OTA": unwrapped_inputs["OTA"],
            "TDEPBU": unwrapped_inputs["TDEPBU"],
            "OLT_1T": unwrapped_inputs["OLT_1T"],
            "TAX": unwrapped_inputs["TAX"],
            "ROT": unwrapped_inputs["ROT"],
            "FTAX": unwrapped_inputs["FTAX"],
            "OLT": unwrapped_inputs["OLT"],
            "NBI": unwrapped_inputs["NBI"],
        }
    )

    # Flow Variables
    variables.update(
        {
            "MTDM": unwrapped_inputs["MTDM"],
            "MCASH": unwrapped_inputs["MCASH"],
            "IMA": unwrapped_inputs["IMA"],
            "IBU": unwrapped_inputs["IBU"],
            "CMA": unwrapped_inputs["CMA"],
            "DCA": unwrapped_inputs["DCA"],
            "DOFA": unwrapped_inputs["DOFA"],
            "DCL": unwrapped_inputs["DCL"],
            "DLL": unwrapped_inputs["DLL"],
            "DOUR": unwrapped_inputs["DOUR"],
            "DSC": unwrapped_inputs["DSC"],
            "DRR": unwrapped_inputs["DRR"],
            "DURE": unwrapped_inputs["DURE"],
            "DIV": unwrapped_inputs["DIV"],
            "CASHFL": unwrapped_inputs["CASHFL"],
            "SMA": unwrapped_inputs["SMA"],
            "MPA": unwrapped_inputs["MPA"],
        }
    )

    # Economic Environment
    variables.update(
        {
            "realr": unwrapped_inputs["realr"],
            "dgnp": unwrapped_inputs["dgnp"],
        }
    )

    # Firm Characteristics (Binary indicators)
    variables.update(
        {
            "Public": unwrapped_inputs["Public"],
            "FAAB": unwrapped_inputs["FAAB"],
            "ruralare": unwrapped_inputs["ruralare"],
            "largcity": unwrapped_inputs["largcity"],
        }
    )

    # Market Variables
    variables.update(
        {
            "market": unwrapped_inputs["market"],
            "marketw": unwrapped_inputs["marketw"],
        }
    )

    variables.update(
        {
            "MTDMt_1": unwrapped_inputs["MTDMt_1"],
            "TDEPMAt_1": unwrapped_inputs["TDEPMAt_1"],
            "MPAt_1": unwrapped_inputs["MPAt_1"],
            "PALLOt_1": unwrapped_inputs["PALLOt_1"],
            "MCASHt_1": unwrapped_inputs["MCASHt_1"],
            "CASHFLt_1": unwrapped_inputs["CASHFLt_1"],
        }
    )

    return variables
