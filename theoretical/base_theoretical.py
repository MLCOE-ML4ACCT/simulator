from abc import ABC, abstractmethod

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.base_estimator import BaseEstimator


class BaseTheoretical(ABC):
    """
    Abstract Base Class for all theoretical models.
    This defines the 'contract' that any theoretical model must follow.
    The simulator will interact with this interface, not the concrete implementation.
    """

    def __init__(self, estimator: BaseEstimator, tax_rate: float = 0.28):
        self.estimator = estimator
        self.tax_rate = tax_rate
        print(f"--> [BaseTheoretical]: Initialized with tax rate τ = {self.tax_rate}")

    @abstractmethod
    def run_one_year(self, initial_state: FirmState) -> FirmState:
        """
        Run one year of simulation based on the initial firm state.
        """
        raise NotImplementedError("Subclasses should implement this method.")
