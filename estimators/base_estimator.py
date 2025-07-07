from abc import ABC, abstractmethod

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


class BaseEstimator(ABC):
    """
    Abstract Base Class for all estimator strategies.
    This defines the 'contract' that any estimator must follow.
    The simulator will interact with this interface, not the concrete implementation.
    """

    @abstractmethod
    def get_flow_variables(self, current_state: FirmState) -> FlowVariables:
        """
        Takes the firm's current state and estimates the flow variables for the next period.

        Args:
            current_state: A FirmState object representing the firm's financials at t-1.

        Returns:
            A FlowVariables object containing all the decisions/events for period t.
        """
        pass
