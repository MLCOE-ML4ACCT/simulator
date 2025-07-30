PALLO_CONFIG = {
    "method": "HS",
    "steps": [
        {
            "name": "level_model",
            "type": "Huber-Schweppes",
            "distribution": "Heavy tail",
            "input_variables": [
                "sumcasht_1",
                "diffcasht_1",
                "ZPFt",
                "dmpat_1",
                "MPAt",
                "realr",
                "FAAB",
                "Public",
                "ruralare",
                "largcity",
                "market",
                "marketw",
            ],
            "coefficients": {
                "Intercept": -13137.3,
                "sumcasht_1": 3.701e-6,
                "diffcasht_1": 1.526e-6,
                "ZPFt": 0.7981,
                "dmpat_1": -0.00263,
                "MPAt": 0.9988,
                "realr": 102203,
                "FAAB": 1203.4,
                "Public": -2897.8,
                "ruralare": 451.6,
                "largcity": -700.3,
                "market": -835002,
                "marketw": -549567,
            },
        }
    ],
}
