import unittest

import tensorflow as tf

from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables
from utils.data_loader import load_data_for_year


class TestDataLoader(unittest.TestCase):
    def test_load_data_for_year(self):
        """
        Tests the load_data_for_year function for correctness using unittest.
        """
        # 1. Setup
        year = 1999
        num_firms = 10

        # 2. Execute
        firm_state, flow_vars = load_data_for_year(year, num_firms)

        # 3. Assert
        # Check types
        self.assertIsInstance(firm_state, FirmState)
        self.assertIsInstance(flow_vars, FlowVariables)

        # Check shape of a sample of tensors
        self.assertEqual(firm_state.CA.shape, (num_firms, 1))
        self.assertEqual(firm_state.MA.shape, (num_firms, 1))
        self.assertEqual(flow_vars.OIBD.shape, (num_firms, 1))
        self.assertEqual(flow_vars.FI.shape, (num_firms, 1))

        # Check values of a sample of tensors
        # Values from data/data_table25.json for year 1999
        expected_ca = 2032624.0
        expected_oibd = 270101.0

        # Use tf.debugging.assert_near for comparing floating point tensors
        tf.debugging.assert_near(
            firm_state.CA,
            tf.constant(expected_ca, shape=(num_firms, 1), dtype=tf.float32),
            rtol=1e-6,
        )
        tf.debugging.assert_near(
            flow_vars.OIBD,
            tf.constant(expected_oibd, shape=(num_firms, 1), dtype=tf.float32),
            rtol=1e-6,
        )

        # Test with a different number of firms to ensure it's dynamic
        num_firms_2 = 5
        firm_state_2, _ = load_data_for_year(year, num_firms_2)
        self.assertEqual(firm_state_2.CA.shape, (num_firms_2, 1))

        # Test for a year that doesn't exist in the data
        with self.assertRaisesRegex(ValueError, "Data for year 1998 not found"):
            load_data_for_year(1998, num_firms)


if __name__ == "__main__":
    unittest.main()
