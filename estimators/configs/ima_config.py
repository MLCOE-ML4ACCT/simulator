"""Configuration for the Investment in Machinery and Equipment (IMA) model.

Source: Table 3
"""

IMA_CONFIG = {
    # The 'method' key tells the EstimatorFactory to use the TobitEstimator.
    "method": "TOBIT",
    # The 'scale' parameter is a key part of the Tobit model's calculation.
    # Source: "Scale" value at the bottom of Table 3a, page 179.
    "scale": 3186957,
    # The 'steps' list for a Tobit model contains a single blueprint for the
    # underlying censored regression.
    "steps": [
        {
            "name": "investment_model",
            "type": "Tobit 1",
            "distribution": "Logistic",
            # Source: Table 3a, page 179
            "input_variables": [
                "sumcasht_1",
                "diffcasht_1",
                "smat",
                "I_BUt_1",
                "EDEPBUt_1",
                "EDEPBUt_12",
                "EDEPMAt",
                "TDEPMAt_1",
                "TDEPMAt_12",
                "ddmtdmt_1",
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
            "coefficients": {
                "Intercept": -1.92e6,
                "sumcasht_1": -0.0003,
                "diffcasht_1": -183e-7,
                "smat": -0.0002,
                "I_BUt_1": 0.00012,
                "EDEPBUt_1": 0.05629,
                "EDEPBUt_12": -68e-11,
                "EDEPMAt": 1.07320,
                "TDEPMAt_1": 0.01086,
                "TDEPMAt_12": -96e-13,
                "ddmtdmt_1": 0.00085,
                "dcat_1": -273e-7,
                "ddmpat_1": 0.00134,
                "ddmpat_12": 374e-15,
                "dclt_1": 0.00002,
                "dgnp": 1.35e-6,
                "FAAB": 381211,
                "Public": 592649,
                "ruralare": 147754,
                "largcity": -352519,
                "market": 1.998e8,
                "marketw": 1616473,
            },
        }
    ],
}
