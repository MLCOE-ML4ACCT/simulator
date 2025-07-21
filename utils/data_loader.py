import json
from pathlib import Path

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


def load_data_for_year(year: int, num_firms: int) -> tuple[FirmState, FlowVariables]:
    """
    Loads data for a specific year from the JSON file and populates
    FirmState and FlowVariables dataclasses.

    Args:
        year (int): The year for which to load the data.
        num_firms (int): The number of firms to create data for.

    Returns:
        tuple[FirmState, FlowVariables]: A tuple containing the populated
                                         FirmState and FlowVariables objects.
    """
    data_path = Path(__file__).parent.parent / "data" / "data_table25.json"
    with open(data_path) as f:
        data = json.load(f)

    year_str = str(year)
    if year_str not in data:
        raise ValueError(f"Data for year {year} not found in {data_path}")

    year_data = data[year_str]
    assets = year_data["Assets"]
    liabilities = year_data["Liabilities"]
    income = year_data["Income statement"]
    flow = year_data["Flow variabels"]

    # Helper to convert to float tensor
    def to_tensor(value):
        return tf.constant(float(value), shape=(num_firms, 1), dtype=tf.float32)

    firm_state = FirmState(
        CA=to_tensor(assets["CA"]),
        MA=to_tensor(assets["MA"]),
        BU=to_tensor(assets["BU"]),
        OFA=to_tensor(assets["OFA"]),
        CMA=to_tensor(flow["CMA"]),
        CL=to_tensor(liabilities["CL"]),
        LL=to_tensor(liabilities["LL"]),
        SC=to_tensor(liabilities["SC"]),
        RR=to_tensor(liabilities["RR"]),
        URE=to_tensor(liabilities["URE"]),
        ASD=to_tensor(liabilities["ASD"]),
        PFt_5=to_tensor(liabilities["PFt-5t"]),
        PFt_4=to_tensor(liabilities["PFt-4t"]),
        PFt_3=to_tensor(liabilities["PFt-3t"]),
        PFt_2=to_tensor(liabilities["PFt-2t"]),
        PFt_1=to_tensor(liabilities["PFt-1t"]),
        PFt_0=to_tensor(liabilities["PFt"]),
        OUR=to_tensor(liabilities["OUR"]),
    )

    flow_vars = FlowVariables(
        OIBD=to_tensor(income["OIBD"]),
        FI=to_tensor(income["FI"]),
        FE=to_tensor(income["FE"]),
        EDEP_MA=to_tensor(income["EDEPma"]),
        EDEP_BU=to_tensor(income["EDEPbu"]),
        S_MA=to_tensor(flow["SMA"]),
        I_MA=to_tensor(flow["I_ma"]),
        I_BU=to_tensor(flow["I_bu"]),
        dofa=to_tensor(flow["dOFA"]),
        dca=to_tensor(flow["dCA"]),
        dcl=to_tensor(flow["dCL"]),
        dll=to_tensor(flow["dLL"]),
        dsc=to_tensor(flow["dSC"]),
        drr=to_tensor(flow["dRR"]),
        TDEP_MA=to_tensor(income["TDEPma-EDEPma"] + income["EDEPma"]),
        TDEP_BU=to_tensor(income["TDEPbu"]),
        dASD=to_tensor(income["TDEPma-EDEPma"]),
        p_allo=to_tensor(income["Pallo"]),
        zpf=to_tensor(income["zPF"]),
        zpf_t1=to_tensor(liabilities["PFt-1t"]),
        zpf_t2=to_tensor(liabilities["PFt-2t"]),
        zpf_t3=to_tensor(liabilities["PFt-3t"]),
        zpf_t4=to_tensor(liabilities["PFt-4t"]),
        zpf_t5=to_tensor(liabilities["PFt-5t"]),
        dour=to_tensor(flow["dOUR"]),
        GC=to_tensor(0),  # Not in JSON, assuming 0
        OA=to_tensor(income["OA"]),
        TL=to_tensor(income["TL"]),
        ROT=to_tensor(income["ROT"]),
        DIV=to_tensor(flow["DIV"]),
    )

    return firm_state, flow_vars
