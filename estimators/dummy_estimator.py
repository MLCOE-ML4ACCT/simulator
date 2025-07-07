# In estimators/dummy_estimator.py
from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.base_estimator import BaseEstimator


class DummyEstimator(BaseEstimator):
    """
    A simple, concrete implementation of the BaseEstimator.
    This estimator returns a hardcoded, constant set of flow variables,
    ignoring the input state. It's used for testing the simulator engine.
    """

    def get_flow_variables(self, current_state: FirmState) -> FlowVariables:
        """
        Returns a pre-defined set of flow variables for a single period.
        """
        print("--> [DummyEstimator]: Providing hardcoded flow variables...")

        # These values are placeholders. They can be adjusted for testing different scenarios.
        # It's important to provide a value for EVERY flow variable.

        return FlowVariables(
            OIBD=100000.0,
            FI=5000.0,
            FE=3000.0,
            I_MA=15000.0,
            S_MA=2000.0,
            I_BU=10000.0,
            dofa=500.0,
            EDEP_MA=8000.0,
            EDEP_BU=3500.0,
            TDEP_MA=9000.0,
            TDEP_BU=4000.0,
            dll=5000.0,
            dcl=2000.0,
            dsc=1000.0,
            drr=500.0,
            dour=100.0,
            p_allo=1500.0,
            zpf_t5=100.0,
            zpf_t4=200.0,
            zpf_t3=300.0,
            zpf_t2=400.0,
            zpf_t1=400.0,
            OA=300.0,
            DIV=10000.0,
            ROT=50.0,
            GC=0.0,
        )
