from dataclasses import dataclass


@dataclass
class FirmState:
    """
    Represents the state of a firm's balance sheet at a single point in time.
    This version is a more direct replication, including summary variables
    from the paper's Appendix A (page 146) as calculated properties.
    """

    # === Core Stored Attributes (The Fundamental State) ===
    # Assets
    CA: float = 0.0  # Current Assets
    MA: float = 0.0  # Machinery and Equipment
    BU: float = 0.0  # Buildings
    OFA: float = 0.0  # Other Fixed Assets
    CMA: float = 0.0  # The Taxable Residual Value of M&A

    # Liabilities & Equity Components
    CL: float = 0.0  # Current Liabilities
    LL: float = 0.0  # Long-Term Liabilities
    ASD: float = 0.0  # Accumulated Supplementary Depreciation
    OUR: float = 0.0  # Other Untaxed Reserves
    PFt_5: float = 0.0
    PFt_4: float = 0.0
    PFt_3: float = 0.0
    PFt_2: float = 0.0
    PFt_1: float = 0.0
    PFt_0: float = 0.0  # Periodical Reserves in Current Period t
    SC: float = 0.0  # Share Capital
    RR: float = 0.0  # Restricted Reserves
    URE: float = 0.0  # Unrestricted Equity

    # Other State Variables
    OL: float = 0.0  # The Stock of Old Losses

    # === Calculated Properties (Summary Variables from Appendix A) ===
    # These are not stored but are calculated on-the-fly when accessed.

    @property
    def FA(self) -> float:
        """Fixed Assets (FA = MA + BU + OFA)"""
        return self.MA + self.BU + self.OFA

    @property
    def K(self) -> float:
        """Total Assets (K)"""
        return self.CA + self.MA + self.BU + self.OFA

    @property
    def WC(self) -> float:
        """Working Capital (WC = CA - CL)"""
        return self.CA - self.CL

    @property
    def UR(self) -> float:
        """Total Untaxed Reserves (UR)"""
        return (
            self.ASD
            + self.OUR
            + self.PFt_0
            + self.PFt_1
            + self.PFt_2
            + self.PFt_3
            + self.PFt_4
            + self.PFt_5
        )

    @property
    def EC(self) -> float:
        """Total Equity Capital (EC = SC + RR + URE)"""
        return self.SC + self.RR + self.URE

    @property
    def B(self) -> float:
        """Total Liabilities and Equity (B)"""
        return self.CL + self.LL + self.UR + self.EC


if __name__ == "__main__":
    # Example usage
    firm = FirmState(
        CA=1000, MA=500, BU=300, OFA=200, CL=400, LL=600, SC=100, RR=50, URE=150
    )
    assert firm.K == (
        500 + 300 + 200 + 1000
    ), "Total Assets (K) calculation is incorrect"
