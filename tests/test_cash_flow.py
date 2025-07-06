"""
Test cash flow calculations and equation 2.39.
"""

import pytest

from data_models.flow_variables import FlowVariables
from estimators.dummy_estimator import DummyEstimator
from theoretical.simple_theoretical import SimulatorEngine


class TestCashFlowCalculations:
    """Test cash flow calculations from Shahnazarian (2004)."""

    def test_cash_flow_calculation_eq_2_39(self):
        """Test equation 2.39: Cash flow calculation."""
        # Create test flow variables
        flow_vars = FlowVariables(
            OIBD=100000.0,
            FI=5000.0,
            FE=3000.0,
            OA=300.0,
            DIV=10000.0,
            dsc=1000.0,
            dcl=2000.0,
            dll=5000.0,
            dour=100.0,
            I_MA=15000.0,
            S_MA=2000.0,
            I_BU=10000.0,
            dofa=500.0,
            dca=0.0,  # This should be calculated, not predetermined
        )

        # Mock FTAX calculation for testing
        FTAX_t = 20000.0

        simulator = SimulatorEngine(estimator=DummyEstimator())
        cashfl_t = simulator.calculate_cash_flow(flow_vars, FTAX_t)

        # Calculate expected cash flow according to equation 2.39
        expected_cashfl = (
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
            - flow_vars.dca
        )

        assert (
            abs(cashfl_t - expected_cashfl) < 1e-6
        ), f"Cash flow calculation incorrect: {cashfl_t} != {expected_cashfl}"

    def test_cash_flow_with_zero_values(self):
        """Test cash flow calculation with zero values."""
        flow_vars = FlowVariables()  # All zeros by default
        FTAX_t = 0.0

        simulator = SimulatorEngine(estimator=DummyEstimator())
        cashfl_t = simulator.calculate_cash_flow(flow_vars, FTAX_t)

        assert (
            cashfl_t == 0.0
        ), f"Cash flow should be zero when all inputs are zero: {cashfl_t}"

    def test_cash_flow_with_negative_values(self):
        """Test cash flow calculation with negative values."""
        flow_vars = FlowVariables(
            OIBD=-10000.0,  # Operating loss
            FI=1000.0,
            FE=5000.0,  # High financial expenses
            OA=2000.0,
            DIV=0.0,  # No dividends
            dsc=0.0,
            dcl=-1000.0,  # Decrease in current liabilities
            dll=-2000.0,  # Decrease in long-term liabilities
            dour=0.0,
            I_MA=5000.0,
            S_MA=0.0,
            I_BU=3000.0,
            dofa=0.0,
            dca=0.0,
        )

        FTAX_t = 0.0  # No tax in loss scenario

        simulator = SimulatorEngine(estimator=DummyEstimator())
        cashfl_t = simulator.calculate_cash_flow(flow_vars, FTAX_t)

        expected_cashfl = (
            -10000.0
            + 1000.0
            - 5000.0
            + 2000.0
            - 0.0
            - 0.0
            + 0.0
            + (-1000.0)
            + (-2000.0)
            + 0.0
            - 5000.0
            + 0.0
            - 3000.0
            - 0.0
            - 0.0
        )

        assert (
            abs(cashfl_t - expected_cashfl) < 1e-6
        ), f"Cash flow calculation incorrect with negative values: {cashfl_t} != {expected_cashfl}"

    def test_cash_flow_components_validation(self):
        """Test that all components of cash flow equation are correctly included."""
        flow_vars = FlowVariables(
            OIBD=100000.0,  # +
            FI=5000.0,  # +
            FE=3000.0,  # -
            OA=300.0,  # +
            DIV=10000.0,  # -
            dsc=1000.0,  # +
            dcl=2000.0,  # +
            dll=5000.0,  # +
            dour=100.0,  # +
            I_MA=15000.0,  # -
            S_MA=2000.0,  # +
            I_BU=10000.0,  # -
            dofa=500.0,  # -
            dca=1500.0,  # -
        )

        FTAX_t = 20000.0  # -

        simulator = SimulatorEngine(estimator=DummyEstimator())
        cashfl_t = simulator.calculate_cash_flow(flow_vars, FTAX_t)

        # Manually calculate to verify each component
        manual_calculation = (
            100000.0  # OIBD
            + 5000.0  # FI
            - 3000.0  # FE
            + 300.0  # OA
            - 20000.0  # FTAX_t
            - 10000.0  # DIV
            + 1000.0  # dsc
            + 2000.0  # dcl
            + 5000.0  # dll
            + 100.0  # dour
            - 15000.0  # I_MA
            + 2000.0  # S_MA
            - 10000.0  # I_BU
            - 500.0  # dofa
            - 1500.0  # dca
        )

        assert (
            abs(cashfl_t - manual_calculation) < 1e-6
        ), f"Cash flow calculation incorrect: {cashfl_t} != {manual_calculation}"

    def test_cash_flow_integration_with_simulator(self):
        """Test that cash flow calculation integrates properly with the simulator."""
        from data_models.firm_state import FirmState

        initial_state = FirmState(
            MA=100000.0,
            CA=50000.0,
            CL=20000.0,
            LL=30000.0,
            SC=80000.0,
            URE=50000.0,
        )

        estimator = DummyEstimator()
        simulator = SimulatorEngine(estimator=estimator)

        # Get flow variables to test cash flow calculation
        flow_vars = estimator.get_flow_variables(initial_state)

        # This should not raise any errors
        cashfl_t = simulator.calculate_cash_flow(flow_vars, 20000.0)

        # Cash flow should be a reasonable number (not NaN or infinity)
        assert isinstance(cashfl_t, float), "Cash flow should be a float"
        assert not (cashfl_t != cashfl_t), "Cash flow should not be NaN"
        assert abs(cashfl_t) < 1e12, "Cash flow should be a reasonable magnitude"
