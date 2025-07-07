"""
Pytest configuration and shared fixtures for the simulator tests.
"""

import pytest

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from estimators.dummy_estimator import DummyEstimator
from theoretical.simple_theoretical import SimulatorEngine


@pytest.fixture
def basic_firm_state():
    """Basic balanced firm state for testing."""
    return FirmState(
        MA=100000.0,
        CA=50000.0,
        BU=25000.0,
        OFA=5000.0,
        CL=20000.0,
        LL=30000.0,
        SC=80000.0,
        URE=50000.0,
    )


@pytest.fixture
def minimal_firm_state():
    """Minimal firm state for edge case testing."""
    return FirmState(
        MA=1000.0,
        CA=500.0,
        CL=200.0,
        SC=1000.0,
        URE=300.0,
    )


@pytest.fixture
def firm_state_with_reserves():
    """Firm state with periodical reserves for testing."""
    return FirmState(
        MA=100000.0,
        CA=50000.0,
        CL=20000.0,
        LL=30000.0,
        SC=80000.0,
        URE=40000.0,
        PFt_0=1000.0,
        PFt_1=800.0,
        PFt_2=600.0,
        PFt_3=400.0,
        PFt_4=200.0,
        PFt_5=100.0,
        ASD=5000.0,
        OUR=2000.0,
    )


@pytest.fixture
def firm_state_with_losses():
    """Firm state with loss carryforward for testing."""
    return FirmState(
        MA=100000.0,
        CA=50000.0,
        CL=20000.0,
        LL=30000.0,
        SC=80000.0,
        URE=50000.0,
        OL=5000.0,
    )


@pytest.fixture
def basic_flow_variables():
    """Basic flow variables for testing."""
    return FlowVariables(
        OIBD=100000.0,
        FI=5000.0,
        FE=3000.0,
        EDEP_MA=8000.0,
        EDEP_BU=3500.0,
        TDEP_MA=9000.0,
        TDEP_BU=4000.0,
        I_MA=15000.0,
        S_MA=2000.0,
        I_BU=10000.0,
        dofa=500.0,
        dcl=2000.0,
        dll=5000.0,
        dsc=1000.0,
        drr=500.0,
        dour=100.0,
        p_allo=1500.0,
        zpf_t5=100.0,
        zpf_t4=200.0,
        zpf_t3=300.0,
        zpf_t2=400.0,
        zpf_t1=500.0,
        OA=300.0,
        DIV=10000.0,
        ROT=50.0,
        GC=0.0,
    )


@pytest.fixture
def dummy_estimator():
    """Dummy estimator for testing."""
    return DummyEstimator()


@pytest.fixture
def simulator_engine(dummy_estimator):
    """Basic simulator engine for testing."""
    return SimulatorEngine(estimator=dummy_estimator, tax_rate=0.28)


@pytest.fixture
def zero_flow_variables():
    """Flow variables with all zeros for boundary testing."""
    return FlowVariables()


class CustomEstimator(DummyEstimator):
    """Custom estimator for testing specific scenarios."""

    def __init__(self, custom_flows=None):
        super().__init__()
        self.custom_flows = custom_flows or {}

    def get_flow_variables(self, current_state):
        flow_vars = super().get_flow_variables(current_state)

        # Apply custom overrides
        for attr, value in self.custom_flows.items():
            if hasattr(flow_vars, attr):
                setattr(flow_vars, attr, value)

        return flow_vars


@pytest.fixture
def custom_estimator():
    """Factory for creating custom estimators."""
    return CustomEstimator


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Helper functions for assertions
def assert_balance_sheet_balanced(firm_state, tolerance=1e-6):
    """Assert that a firm state has a balanced balance sheet."""
    assert (
        abs(firm_state.K - firm_state.B) < tolerance
    ), f"Balance sheet not balanced: K={firm_state.K}, B={firm_state.B}"


def assert_non_negative_reserves(firm_state):
    """Assert that all reserves are non-negative."""
    assert firm_state.PFt_0 >= 0, "PFt_0 should be non-negative"
    assert firm_state.PFt_1 >= 0, "PFt_1 should be non-negative"
    assert firm_state.PFt_2 >= 0, "PFt_2 should be non-negative"
    assert firm_state.PFt_3 >= 0, "PFt_3 should be non-negative"
    assert firm_state.PFt_4 >= 0, "PFt_4 should be non-negative"
    assert firm_state.PFt_5 >= 0, "PFt_5 should be non-negative"
    assert firm_state.ASD >= 0, "ASD should be non-negative"
    assert firm_state.OUR >= 0, "OUR should be non-negative"


# Make helper functions available globally
pytest.assert_balance_sheet_balanced = assert_balance_sheet_balanced
pytest.assert_non_negative_reserves = assert_non_negative_reserves
