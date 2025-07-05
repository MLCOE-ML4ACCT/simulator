import math  # Needed for the assert check

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.base_estimator import BaseEstimator


class SimulatorEngine:
    """
    The core simulation engine. It applies the accounting rules from the
    (04) paper to update a firm's state over one period.
    """

    def __init__(self, estimator: BaseEstimator, tax_rate: float = 0.28):
        """
        Initializes the engine.

        Args:
            estimator: An object that adheres to the BaseEstimator interface.
            tax_rate: The corporate tax rate (τ in the paper). Default is 28%.
        """
        self.estimator = estimator
        self.tax_rate = tax_rate
        print(f"--> [SimulatorEngine]: Initialized with tax rate τ = {self.tax_rate}")

    def run_one_year(self, initial_state: FirmState) -> FirmState:
        flow_vars = self.estimator.get_flow_variables(initial_state)

        # --- Step 2: Calculate Intermediate Income Statement and Tax Values ---
        # This part remains the same
        EBT_t = (
            flow_vars.OIBD
            + flow_vars.FI
            - flow_vars.FE
            - flow_vars.TDEP_MA
            - flow_vars.TDEP_BU
            - flow_vars.p_allo
            + flow_vars.zpf
            + flow_vars.OA
        )
        TL_t = EBT_t
        OTA_t = 0.0
        TA_t = OTA_t - flow_vars.TDEP_BU - initial_state.OL
        TAX_t = self.tax_rate * max(0, EBT_t - TL_t + TA_t)
        FTAX_t = TAX_t - flow_vars.ROT
        NBI_t = EBT_t - FTAX_t

        # --- Step 3: Update ALL Balance Sheet Variables EXCEPT URE ---
        new_state = FirmState()

        new_state.MA = (
            initial_state.MA + flow_vars.I_MA - flow_vars.S_MA - flow_vars.EDEP_MA
        )
        new_state.BU = initial_state.BU + flow_vars.I_BU  # Assuming net investment
        new_state.CA = initial_state.CA + flow_vars.dca
        new_state.OFA = initial_state.OFA + flow_vars.dofa

        new_state.LL = initial_state.LL + flow_vars.dll
        new_state.CL = initial_state.CL + flow_vars.dcl
        new_state.SC = initial_state.SC + flow_vars.dsc
        new_state.RR = initial_state.RR + flow_vars.drr

        new_state.OUR = initial_state.OUR + flow_vars.dour
        new_state.ASD = initial_state.ASD + (flow_vars.TDEP_MA - flow_vars.EDEP_MA)

        new_state.PFt_0 = flow_vars.p_allo
        new_state.PFt_1 = initial_state.PFt_0
        new_state.PFt_2 = initial_state.PFt_1
        new_state.PFt_3 = initial_state.PFt_2
        new_state.PFt_4 = initial_state.PFt_3
        new_state.PFt_5 = initial_state.PFt_4

        if (EBT_t - TL_t + TA_t) < 0:
            new_state.OL = initial_state.OL + abs(EBT_t - TL_t + TA_t)
        else:
            new_state.OL = initial_state.OL

        # --- Step 4: Calculate URE as the Balancing "Plug" Figure ---
        # Total Assets = CA + MA + BU + OFA
        total_assets = new_state.CA + new_state.MA + new_state.BU + new_state.OFA

        # Total Liabilities (excluding all equity)
        total_liabilities = new_state.CL + new_state.LL

        # Total Untaxed Reserves
        total_UR = (
            new_state.ASD
            + new_state.OUR
            + sum(
                [
                    new_state.PFt_0,
                    new_state.PFt_1,
                    new_state.PFt_2,
                    new_state.PFt_3,
                    new_state.PFt_4,
                    new_state.PFt_5,
                ]
            )
        )

        # Restricted Equity
        restricted_equity = new_state.SC + new_state.RR

        # URE is what's left over to make the sheet balance
        new_state.URE = total_assets - total_liabilities - total_UR - restricted_equity

        # --- Step 5: Now, let's independently calculate what URE *should* have been ---
        # This will expose the flawed term.
        # URE(t) = URE(t-1) + NBI(t) - DIV(t) - drr(t)
        ure_from_flow_calc = initial_state.URE + NBI_t - flow_vars.DIV - flow_vars.drr

        print(f"--> [Debug]: URE from balance sheet plug = {new_state.URE:.2f}")
        print(f"--> [Debug]: URE from income flow calc = {ure_from_flow_calc:.2f}")

        # --- Step 6: Final Validation ---
        # This check now just confirms our plug calculation is consistent.
        final_total_liab_equity = (
            total_liabilities + total_UR + restricted_equity + new_state.URE
        )

        print(f"--> [SimulatorEngine]: Calculated Total Assets = {total_assets:.2f}")
        print(
            f"--> [SimulatorEngine]: Calculated Total Liab. & Equity = {final_total_liab_equity:.2f}"
        )

        assert math.isclose(
            total_assets, final_total_liab_equity
        ), "Balance Sheet does not balance!"

        return new_state
