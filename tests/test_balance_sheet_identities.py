"""
Test balance sheet identities and core accounting relationships.
"""

import pytest

from data_models.firm_state import FirmState
from estimators.dummy_estimator import DummyEstimator
from theoretical.simple_theoretical import SimulatorEngine


class TestBalanceSheetIdentities:
    """Test core balance sheet identities from Shahnazarian (2004)."""

    def test_balance_sheet_identity_k_equals_b(self):
        """Test that K = B (equations 2.2 and 2.3)."""
        initial_state = FirmState(
            MA=100000.0,
            CA=50000.0,
            BU=25000.0,
            OFA=5000.0,
            CL=20000.0,
            LL=30000.0,
            SC=80000.0,
            URE=50000.0,
        )

        # Test initial balance
        assert (
            abs(initial_state.K - initial_state.B) < 1e-6
        ), f"Initial balance sheet doesn't balance: K={initial_state.K}, B={initial_state.B}"

        # Test after simulation
        estimator = DummyEstimator()
        simulator = SimulatorEngine(estimator=estimator)
        final_state = simulator.run_one_year(initial_state)

        assert (
            abs(final_state.K - final_state.B) < 1e-6
        ), f"Final balance sheet doesn't balance: K={final_state.K}, B={final_state.B}"

    def test_fixed_assets_calculation_eq_2_1(self):
        """Test equation 2.1: FA = MA + BU + OFA."""
        state = FirmState(
            MA=100000.0,
            BU=25000.0,
            OFA=5000.0,
        )

        expected_FA = state.MA + state.BU + state.OFA
        assert (
            abs(state.FA - expected_FA) < 1e-6
        ), f"FA calculation incorrect: {state.FA} != {expected_FA}"

    def test_total_assets_calculation_eq_2_2(self):
        """Test equation 2.2: K = CA + FA."""
        state = FirmState(
            CA=50000.0,
            MA=100000.0,
            BU=25000.0,
            OFA=5000.0,
        )

        expected_K = state.CA + state.FA
        assert (
            abs(state.K - expected_K) < 1e-6
        ), f"K calculation incorrect: {state.K} != {expected_K}"

    def test_untaxed_reserves_calculation_eq_2_4(self):
        """Test equation 2.4: UR = ASD + PF + OUR."""
        state = FirmState(
            ASD=5000.0,
            PFt_0=1000.0,
            PFt_1=800.0,
            PFt_2=600.0,
            PFt_3=400.0,
            PFt_4=200.0,
            PFt_5=100.0,
            OUR=1500.0,
        )

        expected_UR = (
            state.ASD
            + (
                state.PFt_0
                + state.PFt_1
                + state.PFt_2
                + state.PFt_3
                + state.PFt_4
                + state.PFt_5
            )
            + state.OUR
        )
        assert (
            abs(state.UR - expected_UR) < 1e-6
        ), f"UR calculation incorrect: {state.UR} != {expected_UR}"

    def test_equity_capital_calculation_eq_2_6(self):
        """Test equation 2.6: EC = SC + RR + URE."""
        state = FirmState(
            SC=80000.0,
            RR=10000.0,
            URE=50000.0,
        )

        expected_EC = state.SC + state.RR + state.URE
        assert (
            abs(state.EC - expected_EC) < 1e-6
        ), f"EC calculation incorrect: {state.EC} != {expected_EC}"

    def test_working_capital_calculation(self):
        """Test working capital calculation: WC = CA - CL."""
        state = FirmState(
            CA=50000.0,
            CL=20000.0,
        )

        expected_WC = state.CA - state.CL
        assert (
            abs(state.WC - expected_WC) < 1e-6
        ), f"WC calculation incorrect: {state.WC} != {expected_WC}"
