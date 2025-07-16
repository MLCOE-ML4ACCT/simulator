import math

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.base_estimator import BaseEstimator
from theoretical.base_theoretical import BaseTheoretical


class SimulatorEngine(BaseTheoretical):
    """A basic simulator engine for theoretical financial modeling.

    This is a simplified version of the Shahnazarian (2004) model

    Args:
        tax_rate (float): The tax rate to apply to earnings before tax (EBT).
            Default is 0.28 (28%).
    """

    def __init__(self, tax_rate: float = 0.28):
        self.tax_rate = tax_rate

    def run_one_year(
        self, curr_firm_state: FirmState, curr_flow_vars: FlowVariables
    ) -> tuple[FirmState, FlowVariables]:
        """Run one year of the simulation.

        Args:
            curr_firm_state (FirmState): The current state of the firm
            curr_flow_vars (FlowVariables): The current flow variables for each firm

        Returns:
            tuple[FirmState, FlowVariables]: The updated firm state and flow variables after one year
        """

        return curr_firm_state, curr_flow_vars
