# In data_models/flow_variables.py

from dataclasses import dataclass


@dataclass
class FlowVariables:
    """
    Represents the financial flows (decisions and events) within a single period.
    Corresponds to the "Flow variables" on page 147 of the paper.
    """

    # From Income Statement
    OIBD: float = 0.0  # Operating Income Before Depreciation
    FI: float = 0.0  # Financial Income
    FE: float = 0.0  # Financial Expenses
    EDEP_MA: float = 0.0  # Economic Depreciation
    TDEP_MA: float = 0.0  # Tax Depreciation of M&E
    TDEP_BU: float = 0.0  # Tax Depreciation of Buildings
    OA: float = 0.0  # Other Allocations
    zpf: float = 0.0  # Change in Periodical Reserves
    p_allo: float = 0.0  # Allocations to Periodical Reserves
    ROT: float = 0.0  # Reduction Of Taxes

    # From Balance Sheet Changes
    I_MA: float = 0.0  # Net Investment in Machinery and Equipment
    I_BU: float = 0.0  # Net Investment in Buildings
    dca: float = 0.0  # Net Change in Current Assets
    dofa: float = 0.0  # Net Change in Other Fixed Assets
    dcl: float = 0.0  # Net Change in Current Liabilities
    dll: float = 0.0  # Net Change in Long-Term Liabilities
    dour: float = 0.0  # Net Change in Other Untaxed Reserves
    dsc: float = 0.0  # Net Change in Share Capital
    drr: float = 0.0  # Net Changes in Restricted Reserves

    # Other Flows
    S_MA: float = 0.0  # Sales of Machinery and Equipment
    DIV: float = 0.0  # Dividends Paid to Shareholders
    GC: float = 0.0  # Net Group Contribution
