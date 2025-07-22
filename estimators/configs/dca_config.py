"""Configuration for the Net Change in Current Assets (dca) model.

Source: Table 7a, page 195
"""

DCA_CONFIG = {
    # The overall method is a single-step Huber-Schweppes regression.
    # The factory can be configured to map "HS" to the HSEstimator.
    "method": "HS",
    # The 'steps' array maintains a consistent structure, even for single-step models.
    "steps": [
        {
            # --- Step 1: Level Model ---
            # The paper uses the Huber-Schweppes robust method directly for this variable
            # as it has very few zero-value observations.
            "name": "level_model",
            "type": "Huber-Schweppes",
            "distribution": "Heavy tail",  # As per the paper's discussion on non-normality
            # The complete list of input variables required for this model.
            # Source: "Parameter" column, Table 7a
            "input_variables": [
                "sumcasht_1",
                "diffcasht_1",
                "EDEPMAt",
                "EDEPMAt2",
                "SMAt",
                "IMAt",
                "EDEPBUt",
                "EDEPBUt2",
                "IBUt",
                "IBUt2",
                "IBUt3",
                "dclt_1",
                "ddmpat_1",
                "ddmpat_12",
                "dgnp",
                "FAAB",
                "Public",
                "ruralare",
                "largcity",
                "market",
                "marketw",
            ],
            # The regression coefficients (weights) for each input variable.
            # Source: "Estimate" column, Table 7a
            "coefficients": {
                "Intercept": 2364307,
                "sumcasht_1": -0.00614,
                "diffcasht_1": 0.000392,
                "EDEPMAt": 0.1248,
                "EDEPMAt2": 8.92e-11,
                "SMAt": -0.5029,
                "IMAt": 0.4168,
                "EDEPBUt": 0.4543,
                "EDEPBUt2": -7.32e-9,
                "IBUt": 0.0192,
                "IBUt2": 2.73e-12,
                "IBUt3": 1.38e-21,
                "dclt_1": -0.00598,
                "ddmpat_1": -0.0176,
                "ddmpat_12": -2.19e-12,
                "dgnp": -2.29e-6,
                "FAAB": -1532929,
                "Public": 10225598,
                "ruralare": -386018,
                "largcity": 715615,
                "market": 1.2121e9,
                "marketw": -1.991e9,
            },
        }
    ],
}
