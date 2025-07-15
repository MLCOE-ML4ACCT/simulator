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
    baseline_values = {
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
        CA=create_tensor(baseline_values["CA"]),
        MA=create_tensor(baseline_values["MA"]),
        BU=create_tensor(baseline_values["BU"]),
        OFA=create_tensor(baseline_values["OFA"]),
        CMA=create_tensor(baseline_values["CMA"]),
        CL=create_tensor(baseline_values["CL"]),
        LL=create_tensor(baseline_values["LL"]),
        SC=create_tensor(baseline_values["SC"]),
        RR=create_tensor(baseline_values["RR"]),
        URE=create_tensor(baseline_values["URE"]),
        ASD=create_tensor(baseline_values["ASD"]),
        PFt_5=create_tensor(baseline_values["PFt_5"]),
        PFt_4=create_tensor(baseline_values["PFt_4"]),
        PFt_3=create_tensor(baseline_values["PFt_3"]),
        PFt_2=create_tensor(baseline_values["PFt_2"]),
        PFt_1=create_tensor(baseline_values["PFt_1"]),
        PFt_0=create_tensor(baseline_values["PFt_0"]),
        OUR=create_tensor(baseline_values["OUR"]),
    )
    # Create FlowVariables with dummy data
    flow_variables = FlowVariables(
        OIBD=tf.zeros((num_firms, 1), dtype=tf.float32),
        FI=tf.zeros((num_firms, 1), dtype=tf.float32),
        FE=tf.zeros((num_firms, 1), dtype=tf.float32),
        EDEP_MA=tf.zeros((num_firms, 1), dtype=tf.float32),
        EDEP_BU=tf.zeros((num_firms, 1), dtype=tf.float32),
        S_MA=tf.zeros((num_firms, 1), dtype=tf.float32),
        I_MA=tf.zeros((num_firms, 1), dtype=tf.float32),
        I_BU=tf.zeros((num_firms, 1), dtype=tf.float32),
        dofa=tf.zeros((num_firms, 1), dtype=tf.float32),
        dca=tf.zeros((num_firms, 1), dtype=tf.float32),
        dcl=tf.zeros((num_firms, 1), dtype=tf.float32),
        dll=tf.zeros((num_firms, 1), dtype=tf.float32),
        dsc=tf.zeros((num_firms, 1), dtype=tf.float32),
        drr=tf.zeros((num_firms, 1), dtype=tf.float32),
        TDEP_MA=tf.zeros((num_firms, 1), dtype=tf.float32),
        TDEP_BU=tf.zeros((num_firms, 1), dtype=tf.float32),
        p_allo=tf.zeros((num_firms, 1), dtype=tf.float32),
        zpf_t5=tf.zeros((num_firms, 1), dtype=tf.float32),
        zpf_t4=tf.zeros((num_firms, 1), dtype=tf.float32),
        zpf_t3=tf.zeros((num_firms, 1), dtype=tf.float32),
        zpf_t2=tf.zeros((num_firms, 1), dtype=tf.float32),
        zpf_t1=tf.zeros((num_firms, 1), dtype=tf.float32),
        dour=tf.zeros((num_firms, 1), dtype=tf.float32),
        GC=tf.zeros((num_firms, 1), dtype=tf.float32),
        OA=tf.zeros((num_firms, 1), dtype=tf.float32),
        TL=tf.zeros((num_firms, 1), dtype=tf.float32),
        ROT=tf.zeros((num_firms, 1), dtype=tf.float32),
        DIV=tf.zeros((num_firms, 1), dtype=tf.float32),
    )

    return firm_state, flow_variables
