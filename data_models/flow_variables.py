from dataclasses import dataclass

import tensorflow as tf


@dataclass
class FlowVariables(tf.experimental.ExtensionType):
    """
    Represents the financial flows (decisions and events) within a single period.
    """

    # Income & Expense Flow
    OIBD: tf.Tensor  # Operating Income Before Depreciation
    FI: tf.Tensor  # Financial Income
    FE: tf.Tensor  # Financial Expenses

    # Asset-Related Flow
    EDEP_MA: tf.Tensor  # Economic Depreciation
    EDEP_BU: tf.Tensor  # Economic Depreciation of Buildings
    S_MA: tf.Tensor  # Sales of Machinery and Equipment
    I_MA: tf.Tensor  # Net Investment in Machinery and Equipment
    I_BU: tf.Tensor  # Net Investment in Buildings
    dofa: tf.Tensor  # Net Change in Other Fixed Assets
    dca: tf.Tensor  # Net Change in Current Assets

    # Financial Flow
    dcl: tf.Tensor  # Net Change in Current Liabilities
    dll: tf.Tensor  # Net Change in Long-Term Liabilities
    dsc: tf.Tensor  # Net Change in Share Capital
    drr: tf.Tensor  # Net Changes in Restricted Reserves

    # Tax & Allocation Flows
    TDEP_MA: tf.Tensor  # Tax Depreciation of M&E
    TDEP_BU: tf.Tensor  # Tax Depreciation of Buildings
    p_allo: tf.Tensor  # Allocations to Periodical Reserves
    zpf_t5: tf.Tensor  # Change in Periodical Reserves from t-5 (released in period t)
    zpf_t4: tf.Tensor  # Change in Periodical Reserves from t-4 (released in period t)
    zpf_t3: tf.Tensor  # Change in Periodical Reserves from t-3 (released in period t)
    zpf_t2: tf.Tensor  # Change in Periodical Reserves from t-2 (released in period t)
    zpf_t1: tf.Tensor  # Change in Periodical Reserves from t-1 (released in period t)
    dour: tf.Tensor  # Net Change in Other Untaxed Reserves
    GC: tf.Tensor  # Net Group Contribution
    OA: tf.Tensor  # Other Allocations
    TL: tf.Tensor  # Tax Liability
    ROT: tf.Tensor  # Reduction Of Taxes
    DIV: tf.Tensor  # Dividends Paid to Shareholders
