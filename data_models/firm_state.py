from dataclasses import dataclass

import tensorflow as tf


@dataclass
class FirmState(tf.experimental.ExtensionType):
    """
    Represents the state of a firm's balance sheet at a single point in time.
    This version is a more direct replication, including summary variables
    from the paper's Appendix A (page 146) as calculated properties.
    """

    # === Core Stored Attributes (The Fundamental State) ===
    # Assets
    CA: tf.Tensor  # Current Assets
    MA: tf.Tensor  # Machinery and Equipment
    BU: tf.Tensor  # Buildings
    OFA: tf.Tensor  # Other Fixed Assets
    CMA: tf.Tensor  # The Taxable Residual Value of M&A

    # Liabilities & Equity Components
    CL: tf.Tensor  # Current Liabilities
    LL: tf.Tensor  # Long-Term Liabilities
    SC: tf.Tensor  # Share Capital
    RR: tf.Tensor  # Restricted Reserves
    URE: tf.Tensor  # Unrestricted Equity

    # Untaxed Reserves
    ASD: tf.Tensor  # Accumulated Supplementary Depreciation
    PFt_5: tf.Tensor
    PFt_4: tf.Tensor
    PFt_3: tf.Tensor
    PFt_2: tf.Tensor
    PFt_1: tf.Tensor
    PFt_0: tf.Tensor  # Periodical Reserves in Current Period t
    OUR: tf.Tensor  # Other Untaxed Reserves
