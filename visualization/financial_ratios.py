
import tensorflow as tf

class FinancialRatioCalculator:
    """Calculates financial ratios from simulation output."""

    def __init__(self, result, corporate_tax_rate=0.28):
        self.result = result
        self.corporate_tax_rate = corporate_tax_rate
        self.total_assets = self._calculate_total_assets()

    def _calculate_total_assets(self):
        return (
            self.result["CAt"]
            + self.result["MAt"]
            + self.result["BUt"]
            + self.result["OFAt"]
        )

    def _safe_divide(self, numerator, denominator):
        """Performs division, handling potential division by zero."""
        return tf.math.divide_no_nan(numerator, denominator)

    def calculate_all_ratios(self, realr):
        """Calculates and returns all financial ratios."""
        ratios = {
            "CR": self.calculate_cr(),
            "DR": self.calculate_dr(),
            "DER": self.calculate_der(),
            "ECR": self.calculate_ecr(),
            "FQ": self.calculate_fq(),
            "ICR": self.calculate_icr(),
            "ROE": self.calculate_roe(),
            "ROI": self.calculate_roi(),
            "EFFTAX": self.calculate_efftax(),
            "RROI": self.calculate_rroi(realr),
        }
        ratios["ER"] = ratios["ROI"] - ratios["RROI"]
        return ratios

    def calculate_composition(self):
        """Calculates the composition of assets and liabilities."""
        composition = {
            "Asset_CA": self._safe_divide(self.result["CAt"], self.total_assets),
            "Asset_MA": self._safe_divide(self.result["MAt"], self.total_assets),
            "Asset_BU": self._safe_divide(self.result["BUt"], self.total_assets),
            "Asset_OFA": self._safe_divide(self.result["OFAt"], self.total_assets),
            "Liability_CL": self._safe_divide(self.result["CLt"], self.total_assets),
            "Liability_LL": self._safe_divide(self.result["LLt"], self.total_assets),
            "Equity_EC": self._safe_divide(
                self.result["SCt"] + self.result["RRt"] + self.result["UREt"],
                self.total_assets
            ),
            "Untaxed_Reserves_UR": self._safe_divide(
                self.result["ASDt"] + self.result["PFt"] + self.result["OURt"],
                self.total_assets
            ),
        }
        return composition

    def calculate_cr(self):
        """Calculates the Current Ratio (CR)."""
        return self._safe_divide(self.result["CAt"], self.result["CLt"])

    def calculate_dr(self):
        """Calculates the Debt Ratio (DR)."""
        total_debt = (
            self.result["CLt"]
            + self.result["LLt"]
            + self.corporate_tax_rate
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        return self._safe_divide(total_debt, self.total_assets)

    def calculate_der(self):
        """Calculates the Debt to Equity Ratio (DER)."""
        total_debt = (
            self.result["CLt"]
            + self.result["LLt"]
            + self.corporate_tax_rate
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        total_equity = (
            self.result["SCt"]
            + self.result["RRt"]
            + self.result["UREt"]
            + (1 - self.corporate_tax_rate)
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        return self._safe_divide(total_debt, total_equity)

    def calculate_ecr(self):
        """Calculates the Equity to Capital Ratio (ECR)."""
        total_equity = (
            self.result["SCt"]
            + self.result["RRt"]
            + self.result["UREt"]
            + (1 - self.corporate_tax_rate)
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        return self._safe_divide(total_equity, self.total_assets)

    def calculate_fq(self):
        """Calculates the Financial Q (FQ)."""
        numerator = (
            self.result["CAt"]
            - self.result["CLt"]
            - self.result["LLt"]
            - self.corporate_tax_rate
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        return self._safe_divide(numerator, self.total_assets)

    def calculate_icr(self):
        """Calculates the Interest Coverage Ratio (ICR)."""
        numerator = (
            self.result["OIBDt"]
            - self.result["EDEPMAt"]
            - self.result["EDEPBUt"]
            + self.result["FIt"]
        )
        return self._safe_divide(numerator, self.result["FEt"])

    def calculate_roe(self):
        """Calculates the Return on Equity (ROE)."""
        numerator = (
            self.result["OIBDt"]
            - self.result["EDEPMAt"]
            - self.result["EDEPBUt"]
            + self.result["FIt"]
            - self.result["FEt"]
            + self.result["TLt"]
        )
        denominator = (
            self.result["SCt"]
            + self.result["RRt"]
            + self.result["UREt"]
            + (1 - self.corporate_tax_rate)
            * (
                self.result["ASDt"]
                + self.result["PFt"]
                + self.result["OURt"]
            )
        )
        return self._safe_divide(numerator, denominator)

    def calculate_roi(self):
        """Calculates the Return on Investment (ROI)."""
        numerator = (
            self.result["OIBDt"]
            - self.result["EDEPMAt"]
            - self.result["EDEPBUt"]
            + self.result["FIt"]
            + self.result["TLt"]
        )
        return self._safe_divide(numerator, self.total_assets)

    def calculate_efftax(self):
        """Calculates the Effective Tax Rate (EFFTAX)."""
        eba = (
            self.result["OIBDt"]
            - self.result["EDEPMAt"]
            - self.result["EDEPBUt"]
            + self.result["FIt"]
            - self.result["FEt"]
        )
        return self._safe_divide(self.result["FTAXt"], eba)

    def calculate_rroi(self, realr):
        """Calculates the Required Return on Investment (RROI)."""
        eff_tax = self.calculate_efftax()
        return self._safe_divide(realr, (1 - eff_tax))
