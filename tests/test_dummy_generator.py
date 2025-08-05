import unittest

import numpy as np
import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from synthetic_generator.dummy_generator import dummy_generator


class TestDummyGenerator(tf.test.TestCase):
    """Tests for dummy_generator"""

    def test_invalid_num_firms_raises_error(self):
        """
        Number of firms should be larger than zero
        """
        with self.assertRaisesRegex(
            AssertionError, "Number of firms must be greater than 0"
        ):
            dummy_generator(num_firms=0)

        with self.assertRaisesRegex(
            AssertionError, "Number of firms must be greater than 0"
        ):
            dummy_generator(num_firms=-5)

    def test_accounting_principles(self):
        """
        Check for asset and liability accounting principle
        """
        firm_state, _ = dummy_generator(num_firms=1)

        # Calculate total assets
        total_assets = (
            firm_state.CA[0][0]
            + firm_state.MA[0][0]
            + firm_state.BU[0][0]
            + firm_state.OFA[0][0]
        )

        # Calculate total liabilities and equity
        total_liabilities_equity = (
            firm_state.CL[0][0]
            + firm_state.LL[0][0]
            + firm_state.ASD[0][0]
            + firm_state.OUR[0][0]
            + firm_state.SC[0][0]
            + firm_state.RR[0][0]
            + firm_state.URE[0][0]
            + firm_state.PFt_0[0][0]
            + firm_state.PFt_1[0][0]
            + firm_state.PFt_2[0][0]
            + firm_state.PFt_3[0][0]
            + firm_state.PFt_4[0][0]
            + firm_state.PFt_5[0][0]
        )

        self.assertAllClose(
            total_assets,
            tf.constant(6236489.0),
            msg="Total assets do not equal to the number at table 25 (1999 yr).",
        )

        self.assertEqual(
            total_liabilities_equity,
            tf.constant(6236485.0),
            msg="Total liabilities do not equal to the number at table 25 (1999 yr).",
        )

    def test_tensor_shape(self):
        """Check for tensor shape correct"""

        for n in range(1, 4):
            firm_state, flow_variables = dummy_generator(num_firms=n)
            self.assertEqual(
                firm_state.CA.shape,
                (n, 1),
                f"The tensor shape is wrong when num_firms={n}",
            )
            self.assertEqual(
                flow_variables.OIBD.shape,
                (n, 1),
                f"The tensor shape is wrong when num_firms={n}",
            )


# This allows the test to be run from the command line
if __name__ == "__main__":
    unittest.main()
