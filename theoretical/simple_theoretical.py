import math

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.base_estimator import BaseEstimator
from theoretical.base_theoretical import BaseTheoretical


class SimulatorEngine(BaseTheoretical):
    def __init__(self, estimator: BaseEstimator, tax_rate: float = 0.28):
        self.estimator = estimator
        self.tax_rate = tax_rate
        print(f"--> [SimulatorEngine]: Initialized with tax rate τ = {self.tax_rate}")

    def run_one_year(self, initial_state: FirmState) -> FirmState:
        flow_vars = self.estimator.get_flow_variables(initial_state)

        # Calculate intermediate values with equation references from Shahnazarian (2004)

        # EBT_t = EBA_t - ΔASD - ΔPF_t - OA_t (Eq. 2.11)
        # where EBA_t = OIAD_t + FI_t - FE_t (Eq. 2.10)
        # and OIAD_t = OIBD_t - EDEP_t^MA - EDEP_t^BU (Eq. 2.9)
        # Combining: EBT_t = OIBD_t - EDEP_t^BU + FI_t - FE_t - ΔASD - ΔPF_t - OA_t
        # Where:    ΔASD = TDEP_MA - EDEP_t^MA (Eq. 2.12)
        #           ΔPF_t = p_t^allo - zpf_t (Eq. 2.14)

        # Eq. 2.11: EBA_t = OIAD_t + FI_t - FE_t
        EBA_t = (
            flow_vars.OIBD
            - flow_vars.EDEP_MA
            - flow_vars.EDEP_BU
            + flow_vars.FI
            - flow_vars.FE
        )

        # Eq. 2.15
        # p_allo <-> PE_t (2.14)
        # zpf_t = zpf_t^t-5 + ... + zpf_t^t-1 + PF_t-1^t-5 (2.15)
        EBT_t = (
            EBA_t
            - flow_vars.TDEP_MA
            + flow_vars.EDEP_MA
            - flow_vars.p_allo
            + flow_vars.zpf_total
            - flow_vars.OA
        )

        # TL_t (Tax Liability), estimate/assumption in the paper
        TL_t = EBT_t * 0.5  # Assuming a 50% tax liability for simplicity

        # Related to Eq. 2.16
        # seems not used in the later calculation
        NI_t = EBT_t - TL_t  # Net Income before tax adjustments

        # OTA_t (Other Tax Adjustments) - Section 2.2
        # For baseline model, we assume OTA_t = 0 (no other tax adjustments)
        OTA_t = 0.0

        # Eq. 2.18: TA_t = OTA_t - TDEP_t^BU - OL_{t-1}
        TA_t = OTA_t - flow_vars.TDEP_BU - initial_state.OL

        # Eq. 2.17: TAX_t = τ × max[0, (EBT_t - TL_t + TA_t)]
        TAX_t = self.tax_rate * max(0, EBT_t - TL_t + TA_t)

        # Eq. 2.19: FTAX_t = TAX_t - ROT_t
        FTAX_t = TAX_t - flow_vars.ROT

        # Eq. 2.20: NBI_t = EBT_t - FTAX_t
        NBI_t = EBT_t - FTAX_t

        # Calculate the CHANGE (Delta) for every balance sheet account
        # These equations show how each balance sheet component evolves from t-1 to t

        # Eq. 2.28: MA_t = MA_{t-1} + I_t^MA - S_t^MA - EDEP_t^MA
        # Therefore: ΔMA = I_t^MA - S_t^MA - EDEP_t^MA
        delta_MA = flow_vars.I_MA - flow_vars.S_MA - flow_vars.EDEP_MA

        # Eq. 2.29: BU_t = BU_{t-1} + I_t^BU - EDEP_t^BU
        # Therefore: ΔBU = I_t^BU - EDEP_t^BU
        delta_BU = flow_vars.I_BU - flow_vars.EDEP_BU

        # Eq. 2.30: OFA_t = OFA_{t-1} + dofa_t
        # Therefore: ΔOFA = dofa_t
        delta_OFA = flow_vars.dofa

        # Eq. 2.24: CL_t = CL_{t-1} + dcl_t
        # Therefore: ΔCL = dcl_t
        delta_CL = flow_vars.dcl

        # Eq. 2.23: LL_t = LL_{t-1} + dll_t
        # Therefore: ΔLL = dll_t
        delta_LL = flow_vars.dll

        # Eq. 2.25: SC_t = SC_{t-1} + dsc_t
        # Therefore: ΔSC = dsc_t
        delta_SC = flow_vars.dsc

        # Eq. 2.26: RR_t = RR_{t-1} + drr_t
        # Therefore: ΔRR = drr_t
        delta_RR = flow_vars.drr

        # Eq. 2.22: URE_t = URE_{t-1} + NBI_t - DIV_{t-1} - ΔRR_t - cashfl_t
        # ΔRR = drr_t (Eq. 2.26)
        delta_URE = (
            NBI_t
            - flow_vars.DIV
            - flow_vars.drr
            - self.calculate_cash_flow(flow_vars, FTAX_t)
        )

        # Eq. 2.31: ASD_t = ASD_{t-1} + (TDEP_t^MA - EDEP_t^MA)
        # Therefore: ΔASD = TDEP_t^MA - EDEP_t^MA
        delta_ASD = flow_vars.TDEP_MA - flow_vars.EDEP_MA

        # Eq. 2.32: OUR_t = OUR_{t-1} + dour_t
        # Therefore: ΔOUR = dour_t
        delta_OUR = flow_vars.dour

        # Periodical Reserves change: ΔPF = p_t^allo - zpf_t
        # From equations 2.33-2.38 for periodical reserves management
        delta_PF = flow_vars.p_allo - flow_vars.zpf_total

        # !!!NOTE!!!:
        # some constraints, i.e. Section 2.5-2.7 are ignored

        # Balance Sheet Balancing: Solve for ΔCA (Change in Current Assets)
        # The balance sheet identity K = B must hold, so:
        # ΔK = ΔB
        # ΔCA + ΔMA + ΔBU + ΔOFA = ΔCL + ΔLL + ΔSC + ΔRR + ΔURE + ΔASD + ΔOUR + ΔPF
        # Therefore: ΔCA = (ΔCL + ΔLL + ΔSC + ΔRR + ΔURE + ΔASD + ΔOUR + ΔPF) - (ΔMA + ΔBU + ΔOFA)

        delta_RHS = (
            delta_CL
            + delta_LL
            + delta_SC
            + delta_RR
            + delta_URE
            + delta_ASD
            + delta_OUR
            + delta_PF
        )
        delta_Non_CA = delta_MA + delta_BU + delta_OFA
        delta_CA = delta_RHS - delta_Non_CA

        # Update balance sheet state variables for period t
        # Each state variable X_t = X_{t-1} + ΔX_t

        # Construct the new state
        new_state = FirmState()
        new_state.MA = initial_state.MA + delta_MA  # Eq. 2.28
        new_state.BU = initial_state.BU + delta_BU  # Eq. 2.29
        new_state.OFA = initial_state.OFA + delta_OFA  # Eq. 2.30
        new_state.CA = (
            initial_state.CA + delta_CA
        )  # Eq. 2.27 (derived from balance sheet identity)
        new_state.CL = initial_state.CL + delta_CL  # Eq. 2.24
        new_state.LL = initial_state.LL + delta_LL  # Eq. 2.23
        new_state.SC = initial_state.SC + delta_SC  # Eq. 2.25
        new_state.RR = initial_state.RR + delta_RR  # Eq. 2.26
        new_state.URE = initial_state.URE + delta_URE  # Eq. 2.22
        new_state.ASD = initial_state.ASD + delta_ASD  # Eq. 2.31
        new_state.OUR = initial_state.OUR + delta_OUR  # Eq. 2.32

        # Periodical Reserves Management (Eq.s 2.33-2.38)
        # PF_t^t = p_t^allo (new reserves allocated in period t)
        # PF_t^{t-1} = PF_{t-1}^{t-1} - zpf_t^{t-1} (reserves from previous period, minus releases)
        # And so on for older reserves (they age and eventually get released)

        # Shift periodical reserves: new allocation goes to PFt_0, others age by one period
        new_state.PFt_0 = flow_vars.p_allo
        (
            new_state.PFt_1,
            new_state.PFt_2,
            new_state.PFt_3,
            new_state.PFt_4,
            new_state.PFt_5,
        ) = (
            initial_state.PFt_0 - flow_vars.zpf_t1,
            initial_state.PFt_1 - flow_vars.zpf_t2,
            initial_state.PFt_2 - flow_vars.zpf_t3,
            initial_state.PFt_3 - flow_vars.zpf_t4,
            initial_state.PFt_4 - flow_vars.zpf_t5,
        )

        # Loss Carryforward Logic (Eq. 2.21)
        # OL_t = min[0, (EBT_t - TL_t + TA_t)]
        # If (EBT_t - TL_t + TA_t) < 0, then losses increase by the absolute value
        # If (EBT_t - TL_t + TA_t) >= 0, then losses remain unchanged (they don't decrease automatically)
        if (EBT_t - TL_t + TA_t) < 0:
            new_state.OL = initial_state.OL + abs(EBT_t - TL_t + TA_t)
        else:
            new_state.OL = initial_state.OL

        # Balance Sheet Integrity Check
        # Verify that the fundamental accounting identity K = B holds (Eq.s 2.2 and 2.3)
        # K_t = CA_t + FA_t = CA_t + MA_t + BU_t + OFA_t (Eq. 2.2)
        # B_t = CL_t + LL_t + UR_t + EC_t (Eq. 2.3)
        # where UR_t = ASD_t + PF_t + OUR_t (Eq. 2.4)
        # and EC_t = SC_t + RR_t + URE_t (Eq. 2.6)

        print(f"--> [SimulatorEngine]: Calculated Total Assets (K) = {new_state.K:.2f}")
        print(
            f"--> [SimulatorEngine]: Calculated Total Liab. & Equity (B) = {new_state.B:.2f}"
        )

        assert math.isclose(
            new_state.K, new_state.B, rel_tol=1e-9
        ), "Balance Sheet does not balance! K != B"

        return new_state

    def calculate_cash_flow(self, flow_vars: FlowVariables, FTAX_t: float) -> float:
        """
        Calculate cash flow according to equation 2.39 in the paper.
        This is useful for analysis and validation.
        """
        cashfl_t = (
            flow_vars.OIBD
            + flow_vars.FI
            - flow_vars.FE
            + flow_vars.OA
            - FTAX_t
            - flow_vars.DIV
            + flow_vars.dsc
            + flow_vars.dcl
            + flow_vars.dll
            + flow_vars.dour
            - flow_vars.I_MA
            + flow_vars.S_MA
            - flow_vars.I_BU
            - flow_vars.dofa
            - flow_vars.dca  # This should be calculated, not predetermined
        )
        return cashfl_t
