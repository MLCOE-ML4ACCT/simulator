import math

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


def dummy_generator(num_firms: int = 1) -> tuple[FirmState, FlowVariables]:
    """Generates dummy data for baseyear

    Return firm state and flow variables of a firm according to page 255, table 25.
    The data in the first dimention must be same as the value on table 25.
    For subsequent dim, the data is randomly generated.

    Args:
        num_firms (int): Number of firms to generate data for. Defaults to 10.

    Returns:
        tuple[FirmState, FlowVariables]: A tuple containing FirmState and FlowVariables instances.
    """

    assert num_firms > 0, "Number of firms must be greater than 0"

    # Define baseline values from table 25 (page 255)
    base_firm_values = {
        "CA": 2032624.0,
        "MA": 590199.0,
        "BU": 526432.0,
        "OFA": 3087234.0,
        "CMA": 376070.0,
        "CL": 1596495.0,
        "LL": 2543005.0,
        "SC": 317548.0,
        "RR": 350500.0,
        "URE": 1052945.0,
        "ASD": 214129.0,
        "PFt_5": 19941.0,
        "PFt_4": 23350.0,
        "PFt_3": 25432.0,
        "PFt_2": 23364.0,
        "PFt_1": 27497.0,
        "PFt_0": 31417.0,
        "OUR": 10862.0,
    }

    base_flow_values = {
        "OBID": 270101.0,
        "FI": 259111.0,
        "FE": 178090.0,
        "EDEP_MA": 93798.0,
        "EDEP_BU": 15125,
        "S_MA": 1199.0,
        "I_MA": 139449.0,
        "I_BU": 9201.0,
        "dofa": 1444874.0,
        "dca": 103801.0,
        "dcl": 138027.0,
        "dll": 1237940.0,
        "dsc": 21499.0,
        "drr": 73096.0,
        "TDEP_MA": math.inf,
        "EDEP_MA": math.inf,
        # TDEP^MA - EDEP^MA = dASD
        # table 25 does not explicitly provides TDEP^MA and EDEP^MA
        # only have the difference
        "dASD": 20271.0,
        "TDEP_BU": 14844.0,
        "p_allo": 31418.0,
        "zpf": 11363.0,
        "zpf_t1": math.inf,
        "zpf_t2": math.inf,
        "zpf_t3": math.inf,
        "zpf_t4": math.inf,
        "zpf_t5": math.inf,
        "dour": -9665.0,
        "GC": math.inf,  # table 25 do not include this
        "OA": 11725.0,
        "TL": 36709.0,
        "ROT": 1507.0,
        "DIV": 0.0,
    }

    # Create tensors with the correct values for the first firm
    # For subsequent firms, initialize with zeros (as mentioned in docstring)
    def create_tensor(value: float) -> tf.Tensor:
        tensor = tf.zeros((num_firms, 1), dtype=tf.float32)
        if num_firms > 0:
            # Set the first firm's value to the baseline value
            tensor = tf.tensor_scatter_nd_update(tensor, [[0, 0]], [value])
        return tensor

    # Create FirmState with properly initialized tensors
    firm_state = FirmState(
        CA=create_tensor(base_firm_values["CA"]),
        MA=create_tensor(base_firm_values["MA"]),
        BU=create_tensor(base_firm_values["BU"]),
        OFA=create_tensor(base_firm_values["OFA"]),
        CMA=create_tensor(base_firm_values["CMA"]),
        CL=create_tensor(base_firm_values["CL"]),
        LL=create_tensor(base_firm_values["LL"]),
        SC=create_tensor(base_firm_values["SC"]),
        RR=create_tensor(base_firm_values["RR"]),
        URE=create_tensor(base_firm_values["URE"]),
        ASD=create_tensor(base_firm_values["ASD"]),
        PFt_5=create_tensor(base_firm_values["PFt_5"]),
        PFt_4=create_tensor(base_firm_values["PFt_4"]),
        PFt_3=create_tensor(base_firm_values["PFt_3"]),
        PFt_2=create_tensor(base_firm_values["PFt_2"]),
        PFt_1=create_tensor(base_firm_values["PFt_1"]),
        PFt_0=create_tensor(base_firm_values["PFt_0"]),
        OUR=create_tensor(base_firm_values["OUR"]),
    )
    # Create FlowVariables with dummy data
    flow_variables = FlowVariables(
        OIBD=create_tensor(base_flow_values["OBID"]),
        FI=create_tensor(base_flow_values["FI"]),
        FE=create_tensor(base_flow_values["FE"]),
        EDEP_MA=create_tensor(base_flow_values["EDEP_MA"]),
        EDEP_BU=create_tensor(base_flow_values["EDEP_BU"]),
        S_MA=create_tensor(base_flow_values["S_MA"]),
        I_MA=create_tensor(base_flow_values["I_MA"]),
        I_BU=create_tensor(base_flow_values["I_BU"]),
        dofa=create_tensor(base_flow_values["dofa"]),
        dca=create_tensor(base_flow_values["dca"]),
        dcl=create_tensor(base_flow_values["dcl"]),
        dll=create_tensor(base_flow_values["dll"]),
        dsc=create_tensor(base_flow_values["dsc"]),
        drr=create_tensor(base_flow_values["drr"]),
        TDEP_MA=create_tensor(base_flow_values["TDEP_MA"]),
        TDEP_BU=create_tensor(base_flow_values["TDEP_BU"]),
        dASD=create_tensor(base_flow_values["dASD"]),
        p_allo=create_tensor(base_flow_values["p_allo"]),
        zpf=create_tensor(base_flow_values["zpf"]),
        zpf_t5=create_tensor(base_flow_values["zpf_t5"]),
        zpf_t4=create_tensor(base_flow_values["zpf_t4"]),
        zpf_t3=create_tensor(base_flow_values["zpf_t3"]),
        zpf_t2=create_tensor(base_flow_values["zpf_t2"]),
        zpf_t1=create_tensor(base_flow_values["zpf_t1"]),
        dour=create_tensor(base_flow_values["dour"]),
        GC=create_tensor(base_flow_values["GC"]),
        OA=create_tensor(base_flow_values["OA"]),
        TL=create_tensor(base_flow_values["TL"]),
        ROT=create_tensor(base_flow_values["ROT"]),
        DIV=create_tensor(base_flow_values["DIV"]),
    )

    return firm_state, flow_variables
