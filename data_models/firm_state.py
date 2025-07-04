# In data_models/firm_state.py

from dataclasses import dataclass

@dataclass
class FirmState:
    """
    Represents the state of a firm's balance sheet at a single point in time.
    Corresponds to the "Balance Sheet Variables" on page 146 of the paper.
    """
    # === Assets ===
    CA: float = 0.0      # Current Assets
    MA: float = 0.0      # Machinery and Equipment
    BU: float = 0.0      # Buildings
    OFA: float = 0.0     # Other Fixed Assets
    CMA: float = 0.0     # The Taxable Residual Value of Machinery and Equipment

    # === Liabilities & Equity ===
    CL: float = 0.0      # Current Liabilities
    LL: float = 0.0      # Long-Term Liabilities
    
    # Untaxed Reserves (UR is the sum of these)
    ASD: float = 0.0     # Accumulated Supplementary Depreciation
    OUR: float = 0.0     # Other Untaxed Reserves
    PFt_5: float = 0.0   # Remaining Periodical Reserves From t-5
    PFt_4: float = 0.0   # Remaining Periodical Reserves From t-4
    PFt_3: float = 0.0   # Remaining Periodical Reserves From t-3
    PFt_2: float = 0.0   # Remaining Periodical Reserves From t-2
    PFt_1: float = 0.0   # Remaining Periodical Reserves From t-1
    PFt_0: float = 0.0   # Periodical Reserves in Current Period t

    # Equity Capital (EC is the sum of these)
    SC: float = 0.0      # Share Capital
    RR: float = 0.0      # Restricted Reserves
    URE: float = 0.0     # Unrestricted Equity

    # === Other State Variables ===
    OL: float = 0.0      # The Stock of Old Losses
