"""Configuration for the EDEPMA model.

Source: Section 7.1
"""

EDEPMA_CONFIG = {
    "method": "LLG",
    "steps": [
        {
            # --- Step 1: Probability Model ---
            # This step predicts the probability that a firm will report any EDEPMA > 0.
            "name": "probability_model",
            "type": "Logistic",
            "distribution": "clog-log",
            # The complete list of input variables required for this model.
            # Source: "Parameter" column, Table 1a
            "input_variables": [
                "sumcasht_1",
                "diffcasht_1",
                "TDEPMAt_1",
                "MAt_1",
                "I_MAt_1",
                "I_MAt_12",
                "EDEPBUt_1",
                "EDEPBUt_12",
                "ddmtdmt_1",
                "ddmtdmt_12",
                "dcat_1",
                "ddmpat_1",
                "ddmpat_12",
                "dclt_1",
                "dgnp",
                "FAAB",
                "Public",
                "ruralare",
                "largcity",
                "market",
                "marketw",
            ],
            # The regression coefficients (weights) for each input variable.
            # Source: "Estimate" column, Table 1a
            "coefficients": {
                "Intercept": 0.3749,
                "sumcasht_1": -4.8e-11,
                "diffcasht_1": -2.68e-11,
                "TDEPMAt_1": 7.86e-10,
                "MAt_1": 8.5e-11,
                "I_MAt_1": 1.556e-9,
                "I_MAt_12": -4.8e-19,
                "EDEPBUt_1": 3.043e-8,
                "EDEPBUt_12": -1.16e-16,
                "ddmtdmt_1": 2.52e-12,
                "ddmtdmt_12": -7.82e-21,
                "dcat_1": 2.08e-11,
                "ddmpat_1": 2.71e-11,
                "ddmpat_12": 9.61e-20,
                "dclt_1": -2.3e-12,
                "dgnp": 1.79e-13,
                "FAAB": 0.2807,
                "Public": 0.2681,
                "ruralare": 0.1046,
                "largcity": -0.1321,
                "market": 48.8895,
                "marketw": 8.5185,
            },
        },
        {
            # --- Step 2: Level Model ---
            # This step predicts the actual monetary amount of EDEPMA, but only for firms
            # where the probability from Step 1 is determined to be positive.
            "name": "level_model",
            "type": "Huber-Schweppes",
            "distribution": "Heavy tail",
            # Table 1b, page 172
            "input_variables": [
                "sumcasht_1",
                "diffcasht_1",
                "TDEPMAt_1",
                "MAt_1",
                "I_MAt_1",
                "I_MAt_12",
                "EDEPBUt_1",
                "EDEPBUt_12",
                "ddmtdmt_1",
                "ddmtdmt_12",
                "dcat_1",
                "ddmpat_1",
                "ddmpat_12",
                "dclt_1",
                "dgnp",
                "FAAB",
                "Public",
                "ruralare",
                "largcity",
                "market",
                "marketwsss",
            ],
            "coefficients": {
                "Intercept": 374143,
                "sumcasht_1": 0.000028,
                "diffcasht_1": -0.00003,
                "TDEPMAt_1": 0.5419,
                "MAt_1": 0.0288,
                "I_MAt_1": 0.0563,
                "I_MAt_12": -2.32e-12,
                "EDEPBUt_1": -0.00106,
                "EDEPBUt_12": 1.74e-10,
                "ddmtdmt_1": -0.00065,
                "ddmtdmt_12": -1.29e-12,
                "dcat_1": -0.00005,
                "ddmpat_1": 0.00165,
                "ddmpat_12": -7.48e-13,
                "dclt_1": 0.000035,
                "dgnp": -1.2e-6,
                "FAAB": -241316,
                "Public": 512723,
                "ruralare": -9549.0,
                "largcity": -1144.8,
                "market": 29411097,
                "marketwsss": 1.4677e8,
            },
        },
    ],
}
