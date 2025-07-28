import math
from collections import OrderedDict

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


class SimulatorEngine:
    """A basic simulator engine for theoretical financial modeling.

    This is a simplified version of the Shahnazarian (2004) model

    Args:
        num_firms (int): The number of firms to simulate.
    """

    def __init__(self, num_firms: int):
        assert num_firms > 0, "Number of firms must be positive"
        assert isinstance(num_firms, int), "Number of firms must be an integer"
        self.num_firms = num_firms
        self._input_signature = self._create_input_signature()

        # Create the tf.function with input signature
        self.run_one_year = tf.function(input_signature=[self._input_signature])(
            self._run_one_year_impl
        )

        print(f"--> [SimulatorEngine]: Initialized for {self.num_firms} firms")

    def _create_input_signature(self) -> dict:
        """Create the input signature for TensorFlow function compilation.

        This signature matches the structure used in main.py for the base_year_input.

        Returns:
            dict: Input signature specification for tf.function
        """
        tensor_spec = tf.TensorSpec(shape=(self.num_firms, 1), dtype=tf.float32)

        return {
            # Assets
            "CA": tensor_spec,  # Current Assets
            "MA": tensor_spec,  # Machinery and Equipment
            "BU": tensor_spec,  # Buildings
            "OFA": tensor_spec,  # Other Fixed Assets
            # Liabilities & Equity
            "CL": tensor_spec,  # Current Liabilities
            "LL": tensor_spec,  # Long-Term Liabilities
            "SC": tensor_spec,  # Share Capital
            "ASD": tensor_spec,  # Accumulated Supplementary Depreciation
            "OUR": tensor_spec,  # Other Untaxed Reserves
            "RR": tensor_spec,  # Restricted Reserves
            "URE": tensor_spec,  # Unrestricted Equity
            # Periodical Reserves (current and lagged)
            "PFt": tensor_spec,  # Periodical Reserves in Current Period t
            "PFt_1": tensor_spec,  # Periodical Reserves t-1
            "PFt_2": tensor_spec,  # Periodical Reserves t-2
            "PFt_3": tensor_spec,  # Periodical Reserves t-3
            "PFt_4": tensor_spec,  # Periodical Reserves t-4
            "PFt_5": tensor_spec,  # Periodical Reserves t-5
            # Income Statement Items
            "OL": tensor_spec,  # Operating Loss/Income
            # Flow Variables
            "DIV": tensor_spec,  # Dividends Paid to Shareholders
            # Economic Environment
            "realr": tensor_spec,  # Real Interest Rate
            "dgnp": tensor_spec,  # GNP Growth Rate
            # Firm Characteristics (Binary indicators)
            "Public": tensor_spec,  # Public vs Private Company
            "FAAB": tensor_spec,  # Foreign Affiliate of a Foreign Company
            "ruralare": tensor_spec,  # Rural Area Indicator
            "largcity": tensor_spec,  # Large City Indicator
            # Market Variables
            "market": tensor_spec,  # Market Share
            "marketw": tensor_spec,  # Weighted Market Share
        }

    @property
    def input_signature(self) -> dict:
        """Get the input signature for this simulator engine."""
        return self._input_signature

    def _run_one_year_impl(self, input_dict: OrderedDict) -> dict:
        """Implementation of run_one_year that will be wrapped with tf.function.

        Args:
            input_dict (OrderedDict): A dictionary containing tensor fields matching
                the input signature. This includes firm state variables, flow variables,
                and environmental/characteristic variables.

        Returns:
            dict: A dictionary containing the updated firm state and flow variables
                after one year of simulation
        """
        # TODO: Implement the actual simulation logic
        # unwrap all the input tensors
        unwrapped_inputs = {
            key: tf.squeeze(tensor) for key, tensor in input_dict.items()
        }
        CA_t_1 = unwrapped_inputs["CA"]
        MA_t_1 = unwrapped_inputs["MA"]
        BU_t_1 = unwrapped_inputs["BU"]
        OFA_t_1 = unwrapped_inputs["OFA"]
        CL_t_1 = unwrapped_inputs["CL"]
        LL_t_1 = unwrapped_inputs["LL"]
        SC_t_1 = unwrapped_inputs["SC"]
        ASD_t_1 = unwrapped_inputs["ASD"]
        OUR_t_1 = unwrapped_inputs["OUR"]
        RR_t_1 = unwrapped_inputs["RR"]
        URE_t_1 = unwrapped_inputs["URE"]
        PFt_t_1 = unwrapped_inputs["PFt"]
        PFt_1_t_1 = unwrapped_inputs["PFt_1"]
        PFt_2_t_1 = unwrapped_inputs["PFt_2"]
        PFt_3_t_1 = unwrapped_inputs["PFt_3"]
        PFt_4_t_1 = unwrapped_inputs["PFt_4"]
        PFt_5_t_1 = unwrapped_inputs["PFt_5"]
        OL_t_1 = unwrapped_inputs["OL"]
        DIV_t_1 = unwrapped_inputs["DIV"]
        realr_t_1 = unwrapped_inputs["realr"]
        dgnp_t_1 = unwrapped_inputs["dgnp"]
        Public_t_1 = unwrapped_inputs["Public"]
        FAAB_t_1 = unwrapped_inputs["FAAB"]
        ruralare_t_1 = unwrapped_inputs["ruralare"]
        largcity_t_1 = unwrapped_inputs["largcity"]
        market_t_1 = unwrapped_inputs["market"]
        marketw_t_1 = unwrapped_inputs["marketw"]

        return {"status": "not_implemented"}
