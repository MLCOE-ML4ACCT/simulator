import math

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.debug_utils import debug_tf_input_signature
from estimators.factory import EstimatorFactory


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
        self.estimator_factory = EstimatorFactory(num_firms=num_firms)
        self.allocation_rate = 0.25
        self.corporate_tax_rate = 0.28
        self.db_rate = 0.30
        self.rv_rate = 0.25

        self.edep_ma_est = self.estimator_factory.get_estimator("EDEPMA")
        self.s_ma_est = self.estimator_factory.get_estimator("SMA")
        self.i_ma_est = self.estimator_factory.get_estimator("IMA")
        self.edep_bu_est = self.estimator_factory.get_estimator("EDEPBU")
        self.i_bu_est = self.estimator_factory.get_estimator("IBU")
        self.dofa_est = self.estimator_factory.get_estimator("DOFA")
        self.dca_est = self.estimator_factory.get_estimator("DCA")
        self.dll_est = self.estimator_factory.get_estimator("DLL")
        self.dcl_est = self.estimator_factory.get_estimator("DCL")
        self.dsc_est = self.estimator_factory.get_estimator("DSC")
        self.drr_est = self.estimator_factory.get_estimator("DRR")
        self.oibd_est = self.estimator_factory.get_estimator("OIBD")
        self.fi_est = self.estimator_factory.get_estimator("FI")
        self.fe_est = self.estimator_factory.get_estimator("FE")
        self.tdep_ma_est = self.estimator_factory.get_estimator("TDEPMA")
        self.zpf_est = self.estimator_factory.get_estimator("ZPF")
        self.dour_est = self.estimator_factory.get_estimator("DOUR")
        self.gc_est = self.estimator_factory.get_estimator("GC")
        self.oa_est = self.estimator_factory.get_estimator("OA")
        self.tl_est = self.estimator_factory.get_estimator("TL")
        self.ota_est = self.estimator_factory.get_estimator("OTA")
        self.tdep_bu_est = self.estimator_factory.get_estimator("TDEPBU")
        self.p_allo_est = self.estimator_factory.get_estimator("PALLO")
        self.rot_est = self.estimator_factory.get_estimator("ROT")

        print(f"--> [SimulatorEngine]: Initialized for {self.num_firms} firms")

    def _unwrap_inputs(self, input_dict: dict) -> dict:
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

        return variables

    def _prepare_edepma_inputs(
        self, sumcasht_1, diffcasht_1, vars_t_1, ddMTDMt_1, ddMPAt_1
    ):
        """Prepares the input dictionary for the EDEPMA estimator."""
        return {
            "sumcasht_1": sumcasht_1,
            "diffcasht_1": diffcasht_1,
            "TDEPMAt_1": vars_t_1["TDEPMA"],
            "MAt_1": vars_t_1["MA"],
            "I_MAt_1": vars_t_1["IMA"],
            "I_MAt_12": vars_t_1["IMA"] ** 2,
            "EDEPBUt_1": vars_t_1["EDEPBU"],
            "EDEPBUt_12": vars_t_1["EDEPBU"] ** 2,
            "ddmtdmt_1": ddMTDMt_1,
            "ddmtdmt_12": ddMTDMt_1**2,
            "dcat_1": vars_t_1["DCA"],
            "ddmpat_1": ddMPAt_1,
            "ddmpat_12": ddMPAt_1**2,
            "dclt_1": vars_t_1["DCL"],
            "dgnp": vars_t_1["dgnp"],
            "FAAB": vars_t_1["FAAB"],
            "Public": vars_t_1["Public"],
            "ruralare": vars_t_1["ruralare"],
            "largcity": vars_t_1["largcity"],
            "market": vars_t_1["market"],
            "marketw": vars_t_1["marketw"],
        }

    def run_one_year(self, input_t_1: dict, input_t_2: dict) -> dict:
        """Implementation of run_one_year.

        Args:
            input_t_1 (dict): A dictionary containing tensor fields for time t-1
            input_t_2 (dict): A dictionary containing tensor fields for time t-2

        Returns:
            dict: A dictionary containing the updated firm state and flow variables
                after one year of simulation
        """
        # Unwrap all inputs into individual variables
        vars_t_1 = self._unwrap_inputs(input_t_1)
        vars_t_2 = self._unwrap_inputs(input_t_2)

        # Example placeholder for future implementation:
        ddMTDMt_1 = (vars_t_1["MTDM"] - vars_t_1["TDEPMA"]) - (
            vars_t_2["MTDM"] - vars_t_2["TDEPMA"]
        )
        dMPAt_1 = vars_t_1["MPA"] - vars_t_1["PALLO"]
        dMPAt_2 = vars_t_2["MPA"] - vars_t_2["PALLO"]
        ddMPAt_1 = dMPAt_1 - dMPAt_2
        dCASHt_1 = vars_t_1["CASHFL"] - vars_t_2["CASHFL"]
        dmCASHt_1 = vars_t_1["MCASH"] - vars_t_1["CASHFL"]
        dmCASHt_2 = vars_t_2["MCASH"] - vars_t_2["CASHFL"]
        ddmCASHt_1 = dmCASHt_1 - dmCASHt_2

        sumcasht_1 = ddmCASHt_1 + dCASHt_1
        diffcasht_1 = ddmCASHt_1 - dCASHt_1

        edepma_inputs = self._prepare_edepma_inputs(
            sumcasht_1, diffcasht_1, vars_t_1, ddMTDMt_1, ddMPAt_1
        )
        EDEPMAt = self.edep_ma_est.predict(edepma_inputs)
        sumCACLt_1 = vars_t_1["CA"] + vars_t_1["CL"]
        diffCACLt_1 = vars_t_1["CA"] - vars_t_1["CL"]

        SMAt = self.s_ma_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "TDEPMAt_1": vars_t_1["TDEPMA"],
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "MAt_1": vars_t_1["MA"],
                "I_BUt_1": vars_t_1["IBU"],
                "I_BUt_12": vars_t_1["IBU"] ** 2,
                "EDEPBUt_1": vars_t_1["EDEPBU"],
                "EDEPBUt_12": vars_t_1["EDEPBU"] ** 2,
                "ddmtdmt_1": ddMTDMt_1,
                "ddmtdmt_12": ddMTDMt_1**2,
                "dcat_1": vars_t_1["DCA"],
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "dclt_1": vars_t_1["DCL"],
                "dclt_12": vars_t_1["DCL"] ** 2,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
                "sumcaclt_1": sumCACLt_1,
                "diffcaclt_1": diffCACLt_1,
            }
        )

        IMAt = self.i_ma_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "smat": SMAt,
                "I_BUt_1": vars_t_1["IBU"],
                "EDEPBUt_1": vars_t_1["EDEPBU"],
                "EDEPBUt_12": vars_t_1["EDEPBU"] ** 2,
                "EDEPMAt": EDEPMAt,
                "TDEPMAt_1": vars_t_1["TDEPMA"],
                "TDEPMAt_12": vars_t_1["TDEPMA"] ** 2,
                "ddmtdmt_1": ddMTDMt_1,
                "dcat_1": vars_t_1["DCA"],
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "dclt_1": vars_t_1["DCL"],
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        dIMAt = tf.cast(IMAt > 0, dtype=tf.float32)

        # Declining Balance Method (Formula 2.49)
        TDDBMAt = self.db_rate * (vars_t_1["CMA"] + IMAt - SMAt)
        TDDBMAt = tf.maximum(0.0, TDDBMAt)  # Depreciation cannot be negative

        # Rest Value Method (Formula 2.51)
        TDRVMAt = self.rv_rate * (vars_t_1["CMA"] + IMAt - SMAt)
        TDRVMAt = tf.maximum(0.0, TDRVMAt)  # Depreciation cannot be negative

        # Firms choose the method that gives the highest deduction.
        # This simplifies the conditional logic in formulas 2.53 and 2.54.
        MTDMt = tf.maximum(TDDBMAt, TDRVMAt)

        EDEPBUt = self.edep_bu_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "IMAt": IMAt,
                "BUt_1": vars_t_1["BU"],
                "BUt_12": vars_t_1["BU"] ** 2,
                "dcat_1": vars_t_1["DCA"],
                "ddmpat_1": ddMPAt_1,
                "dclt_1": vars_t_1["DCL"],
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
                "sumcaclt_1": sumCACLt_1,
                "diffcaclt_1": diffCACLt_1,
                "SMAt2": SMAt**2,
            }
        )
        IBUt = self.i_bu_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "IMAt": IMAt,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "dcat_1": vars_t_1["DCA"],
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "dclt_1": vars_t_1["DCL"],
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
                "sumcaclt_1": sumCACLt_1,
                "diffcaclt_1": diffCACLt_1,
            }
        )
        dIBUt = tf.cast(IBUt != 0, dtype=tf.float32)

        dOFAt = self.dofa_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "DIMA": dIMAt,
                "DIBU": dIBUt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        dOFAt = tf.maximum(dOFAt, -vars_t_1["OFA"])
        ddOFAt = tf.cast(dOFAt != 0, dtype=tf.float32)

        dCAt = self.dca_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "IMAt": IMAt,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "IBUt": IBUt,
                "IBUt2": IBUt**2,
                "IBUt3": IBUt**3,
                "dclt_1": vars_t_1["DCL"],
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        dCAt = tf.maximum(dCAt, -vars_t_1["DCA"])
        dLLt = self.dll_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "DIMA": dIMAt,
                "DIBU": dIBUt,
                "Ddofa": ddOFAt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        ddLLt = tf.cast(dLLt != 0, dtype=tf.float32)

        dCLt = self.dcl_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "IMAt": IMAt,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "IBUt": IBUt,
                "IBUt2": IBUt**2,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "dcat": dCAt,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        dSCt = self.dsc_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "DIMA": dIMAt,
                "DIBU": dIBUt,
                "Ddofa": ddOFAt,
                "Ddll": ddLLt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        # constraint eq 3.56
        min_dSCt = 100000.0 - vars_t_1["SC"]
        dSCt = tf.maximum(dSCt, min_dSCt)
        ddSCt = tf.cast(dSCt != 0, dtype=tf.float32)

        dRRt = self.drr_est.predict(
            {
                "ddmcasht_1": ddmCASHt_1,
                "ddmcasht_12": ddmCASHt_1**2,
                "DIMA": dIMAt,
                "DIBU": dIBUt,
                "Ddofa": ddOFAt,
                "Ddll": ddLLt,
                "Ddsc": ddSCt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        dRRt = tf.maximum(dRRt, -vars_t_1["RR"])
        OIBDt = self.oibd_est.predict(
            {
                "sumcaclt_1": sumCACLt_1,
                "diffcaclt_1": diffCACLt_1,
                "MAt_1": vars_t_1["MA"],
                "I_MAt": IMAt,
                "SMAt": SMAt,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "BUt_1": vars_t_1["BU"],
                "I_BUt": IBUt,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "dcat": dCAt,
                "dcat2": dCAt**2,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "dcasht_1": dCASHt_1,
                "dcasht_12": dCASHt_1**2,
                "dclt": dCLt,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        FIt = self.fi_est.predict(
            {
                "I_BUt": IBUt,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "I_MAt": IMAt,
                "I_MAt2": IMAt**2,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "dcat": dCAt,
                "dcat2": dCAt**2,
                "dofat": dOFAt,
                "OFAt_1": vars_t_1["OFA"],
                "CAt_1": vars_t_1["CA"],
                "MAt_1": vars_t_1["MA"],
                "BUt_1": vars_t_1["BU"],
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        sumdCAdCLt = dCAt + dCLt
        diffdCAdCLt = dCAt - dCLt
        sumdOFAdLLt = dOFAt + dLLt
        diffdOFAdLLt = dOFAt - dLLt
        FEt = self.fe_est.predict(
            {
                "I_BUt": IBUt,
                "EDEPMAt": EDEPMAt,
                "SMAt": SMAt,
                "I_MAt": IMAt,
                "EDEPBUt": EDEPBUt,
                "OFAt_1": vars_t_1["OFA"],
                "MAt_1": vars_t_1["MA"],
                "BUt_1": vars_t_1["BU"],
                "LLt_1": vars_t_1["LL"],
                "sumcaclt_1": sumCACLt_1,
                "diffcaclt_1": diffCACLt_1,
                "sumdcadclt": sumdCAdCLt,
                "diffdcadclt": diffdCAdCLt,
                "sumdofadllt": sumdOFAdLLt,
                "diffdofadllt": diffdOFAdLLt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        TDEPMAt = self.tdep_ma_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "SMAt": SMAt,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "I_MAt": IMAt,
                "I_MAt2": IMAt**2,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        TDEPMAt = tf.minimum(TDEPMAt, MTDMt)  # Enforce the MTDM constraint
        dTDEPMAt = tf.cast(TDEPMAt > 0, dtype=tf.float32)

        mandatory_reversal = vars_t_1["PFt_5"]
        ZPFt = self.zpf_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "PALLOt_1": vars_t_1["PALLO"],
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "DTDEPMA": dTDEPMAt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        ZPFt = tf.maximum(ZPFt, mandatory_reversal)

        dZPFt = tf.cast(ZPFt > 0, dtype=tf.float32)

        dOURt = self.dour_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "DTDEPMA": dTDEPMAt,
                "DZPF": dZPFt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        dOURt = tf.maximum(dOURt, -vars_t_1["OUR"])
        ddOURt = tf.cast(dOURt != 0, dtype=tf.float32)

        GCt = self.gc_est.predict(
            {
                "OIBDt": OIBDt,
                "OIBDt2": OIBDt**2,
                "OIBDt3": OIBDt**3,
                "FIt": FIt,
                "FEt": FEt,
                "TDEPMAt": TDEPMAt,
                "TDEPMAt2": TDEPMAt**2,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "ZPFt": ZPFt,
                "dourt": dOURt,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "EDEPBU": vars_t_1["EDEPBU"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        OAt = self.oa_est.predict(
            {
                "dourt": dOURt,
                "GCt": GCt,
                "DTDEPMA": dTDEPMAt,
                "DZPF": dZPFt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )
        TLt = self.tl_est.predict(
            {
                "OIBDt": OIBDt,
                "OIBDt2": OIBDt**2,
                "FIt": FIt,
                "FIt2": FIt**2,
                "FEt": FEt,
                "FEt2": FEt**2,
                "TDEPMAt": TDEPMAt,
                "TDEPMAt2": TDEPMAt**2,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "dourt": dOURt,
                "dourt2": dOURt**2,
                "ZPFt": ZPFt,
                "PALLOt_1": vars_t_1["PALLO"],
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        OTAt = self.ota_est.predict(
            {
                "PALLOt_1": vars_t_1["PALLO"],
                "ZPFt": ZPFt,
                "TDEPMAt": TDEPMAt,
                "TDEPMAt2": TDEPMAt**2,
                "OIBDt": OIBDt,
                "OIBDt2": OIBDt**2,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "dourt": dOURt,
                "TLt": TLt,
                "FIt": FIt,
                "FEt": FEt,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        TDEPBUt = self.tdep_bu_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "EDEPMAt": EDEPMAt,
                "EDEPMAt2": EDEPMAt**2,
                "SMAt": SMAt,
                "I_MAt": IMAt,
                "BUt_1": vars_t_1["BU"],
                "BUt_12": vars_t_1["BU"] ** 2,
                "dcat": dCAt,
                "dcat2": dCAt**2,
                "dclt": dCLt,
                "ddmpat_1": ddMPAt_1,
                "ddmpat_12": ddMPAt_1**2,
                "ddmpat_13": ddMPAt_1**3,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        TAt = OTAt - TDEPBUt - vars_t_1["OLT"]
        PBASEt = OIBDt - EDEPBUt + FIt + FEt - TDEPMAt + ZPFt + OAt - TLt + TAt
        MPAt = tf.maximum(0.0, (self.allocation_rate * PBASEt))

        PALLOt = self.p_allo_est.predict(
            {
                "sumcasht_1": sumcasht_1,
                "diffcasht_1": diffcasht_1,
                "ZPFt": ZPFt,
                "dmpat_1": dMPAt_1,
                "MPAt": MPAt,
                "realr": vars_t_1["realr"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        PALLOt = tf.maximum(0.0, tf.minimum(PALLOt, (self.allocation_rate * PBASEt)))

        sumALLOZPFt = PALLOt + ZPFt
        diffALLOZPFt = PALLOt - ZPFt
        ROTt = self.rot_est.predict(
            {
                "sumallozpft": sumALLOZPFt,
                "diffallozpft": diffALLOZPFt,
                "TDEPMAt": TDEPMAt,
                "TDEPMAt2": TDEPMAt**2,
                "OIBDt": OIBDt,
                "OIBDt2": OIBDt**2,
                "EDEPBUt": EDEPBUt,
                "EDEPBUt2": EDEPBUt**2,
                "OTAt": OTAt,
                "OTAt2": OTAt**2,
                "TDEPBUt": TDEPBUt,
                "TDEPBUt2": TDEPBUt**2,
                "dourt": dOURt,
                "TLt": TLt,
                "FIt": FIt,
                "FEt": FEt,
                "dgnp": vars_t_1["dgnp"],
                "FAAB": vars_t_1["FAAB"],
                "Public": vars_t_1["Public"],
                "ruralare": vars_t_1["ruralare"],
                "largcity": vars_t_1["largcity"],
                "market": vars_t_1["market"],
                "marketw": vars_t_1["marketw"],
            }
        )

        # checkout section 2.8
        MAt = vars_t_1["MA"] + IMAt - SMAt - EDEPMAt
        BUt = vars_t_1["BU"] + IBUt - EDEPBUt
        OFAt = vars_t_1["OFA"] + dOFAt
        CAt = vars_t_1["CA"] + dCAt
        SCt = vars_t_1["SC"] + dSCt
        RRt = vars_t_1["RR"] + dRRt
        OURt = vars_t_1["OUR"] + dOURt
        CMAt = vars_t_1["CMA"] + IMAt - SMAt - TDEPMAt
        ASDt = vars_t_1["ASD"] + (TDEPMAt - EDEPMAt)
        dMPAt = MPAt - PALLOt
        ddMPAt = dMPAt - dMPAt_1
        LLt = vars_t_1["LL"] + dLLt
        CLt = vars_t_1["CL"] + dCLt
        PFt_t = PALLOt
        mandatory_reversal = vars_t_1["PFt_5"]
        voluntary_reversal = tf.maximum(0.0, ZPFt - mandatory_reversal)
        ZPFt_t_5 = tf.minimum(voluntary_reversal, vars_t_1["PFt_4"])
        remaining_reversal = voluntary_reversal - ZPFt_t_5
        ZPFt_t_4 = tf.minimum(remaining_reversal, vars_t_1["PFt_3"])
        remaining_reversal = remaining_reversal - ZPFt_t_4
        ZPFt_t_3 = tf.minimum(remaining_reversal, vars_t_1["PFt_2"])
        remaining_reversal = remaining_reversal - ZPFt_t_3
        ZPFt_t_2 = tf.minimum(remaining_reversal, vars_t_1["PFt_1"])
        remaining_reversal = remaining_reversal - ZPFt_t_2
        ZPFt_t_1 = tf.minimum(remaining_reversal, vars_t_1["PFt"])

        PFt_t_5 = vars_t_1["PFt_4"] - ZPFt_t_5
        PFt_t_4 = vars_t_1["PFt_3"] - ZPFt_t_4
        PFt_t_3 = vars_t_1["PFt_2"] - ZPFt_t_3
        PFt_t_2 = vars_t_1["PFt_1"] - ZPFt_t_2
        PFt_t_1 = vars_t_1["PFt"] - ZPFt_t_1

        PFt = PFt_t + PFt_t_1 + PFt_t_2 + PFt_t_3 + PFt_t_4 + PFt_t_5

        EBTt = OIBDt - EDEPBUt + FIt - FEt - TDEPMAt - PALLOt + ZPFt + OAt
        TAXt = self.corporate_tax_rate * tf.maximum(0.0, (EBTt - TLt + TAt))
        FTAXt = TAXt - ROTt
        NBIt = EBTt - FTAXt
        OLt = tf.abs(tf.minimum(0.0, (EBTt - TLt + TAt)))
        CASHFLt = (
            OIBDt
            + FIt
            - FEt
            + OAt
            - FTAXt
            - vars_t_1["DIV"]
            + dSCt
            + dCLt
            + dLLt
            + dOURt
            - IMAt
            + SMAt
            - IBUt
            - dOFAt
            - dCAt
        )
        UREt = vars_t_1["URE"] + NBIt - vars_t_1["DIV"] - dRRt - CASHFLt
        MCASHt = vars_t_1["URE"] + NBIt - dRRt
        DIVt = tf.maximum(0.0, tf.minimum(CASHFLt, MCASHt))

        return {
            "ddMTDMt_1": ddMTDMt_1,
            "dMPAt_1": dMPAt_1,
            "dMPAt_2": dMPAt_2,
            "ddMPAt_1": ddMPAt_1,
            "dCASHt_1": dCASHt_1,
            "dmCASHt_1": dmCASHt_1,
            "dmCASHt_2": dmCASHt_2,
            "ddmCASHt_1": ddmCASHt_1,
            "sumcasht_1": sumcasht_1,
            "diffcasht_1": diffcasht_1,
            "EDEPMAt": EDEPMAt,
            "sumCACLt_1": sumCACLt_1,
            "diffCACLt_1": diffCACLt_1,
            "SMAt": SMAt,
            "IMAt": IMAt,
            "dIMAt": dIMAt,
            "TTDDBMAt": TDDBMAt,
            "TDRVMAt": TDRVMAt,
            "MTDMt": MTDMt,
            "EDEPBUt": EDEPBUt,
            "IBUt": IBUt,
            "dIBUt": dIBUt,
            "dOFAt": dOFAt,
            "ddOFAt": ddOFAt,
            "dCAt": dCAt,
            "dLLt": dLLt,
            "ddLLt": ddLLt,
            "dCLt": dCLt,
            "dSCt": dSCt,
            "ddSCt": ddSCt,
            "dRRt": dRRt,
            "OIBDt": OIBDt,
            "FIt": FIt,
            "sumdCAdCLt": sumdCAdCLt,
            "diffdCAdCLt": diffdCAdCLt,
            "sumdOFAdLLt": sumdOFAdLLt,
            "diffdOFAdLLt": diffdOFAdLLt,
            "FEt": FEt,
            "TDEPMAt": TDEPMAt,
            "dTDEPMAt": dTDEPMAt,
            "ZPFt": ZPFt,
            "dZPFt": dZPFt,
            "dOURt": dOURt,
            "ddOURt": ddOURt,
            "GCt": GCt,
            "OAt": OAt,
            "TLt": TLt,
            "OTAt": OTAt,
            "TDEPBUt": TDEPBUt,
            "TAt": TAt,
            "PBASEt": PBASEt,
            "MPAt": MPAt,
            "PALLOt": PALLOt,
            "sumALLOZPFt": sumALLOZPFt,
            "diffALLOZPFt": diffALLOZPFt,
            "ROTt": ROTt,
            "MAt": MAt,
            "BUt": BUt,
            "OFAt": OFAt,
            "CAt": CAt,
            "SCt": SCt,
            "RRt": RRt,
            "OURt": OURt,
            "CMAt": CMAt,
            "ASDt": ASDt,
            "dMPAt": dMPAt,
            "ddMPAt": ddMPAt,
            "LLt": LLt,
            "CLt": CLt,
            "PFt_t": PFt_t,
            "mandatory_reversal": mandatory_reversal,
            "voluntary_reversal": voluntary_reversal,
            "ZPFt_t_5": ZPFt_t_5,
            "remaining_reversal": remaining_reversal,
            "ZPFt_t_4": ZPFt_t_4,
            "ZPFt_t_3": ZPFt_t_3,
            "ZPFt_t_2": ZPFt_t_2,
            "ZPFt_t_1": ZPFt_t_1,
            "PFt_t_5": PFt_t_5,
            "PFt_t_4": PFt_t_4,
            "PFt_t_3": PFt_t_3,
            "PFt_t_2": PFt_t_2,
            "PFt_t_1": PFt_t_1,
            "PFt": PFt,
            "EBTt": EBTt,
            "TAXt": TAXt,
            "FTAXt": FTAXt,
            "NBIt": NBIt,
            "OLt": OLt,
            "CASHFLt": CASHFLt,
            "UREt": UREt,
            "MCASHt": MCASHt,
            "DIVt": DIVt,
        }
