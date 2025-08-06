# Financial Simulator

A Python-based financial simulation engine that models firm balance sheets and cash flows based on the theoretical framework from Shahnazarian (2004).

## Overview

This simulator implements a comprehensive financial model that tracks firm state variables and flow variables across time periods. It follows balance sheet identities and accounting relationships to provide realistic financial projections.

## Project Structure

```
simulator/
├── data_models/           # Core data structures
│   ├── firm_state.py     # Balance sheet state representation
│   └── flow_variables.py # Period flow variables
├── estimators/           # Forecasting models and factory
│   ├── factory.py        # EstimatorFactory for creating estimators
│   ├── configs/          # Estimator configurations
│   ├── base_layer/       # Base TensorFlow layers
│   ├── layers/           # TensorFlow layers for estimators
│   └── models/           # Various estimator implementations
├── theoretical/          # Simulation engines
│   └── simple_theoretical.py # Main simulation engine
├── examples/             # Usage examples
│   └── estimators/       # EstimatorFactory examples
├── tests/                # Unit tests
│   └── estimators/       # Tests for estimators
│       └── layers/       # Tests for TensorFlow layers
├── utils/                # Utility functions
└── docs/                 # Documentation
    └── variables.md      # Variable definitions
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simulator

# Install in development mode (Python 3.10+ required)
pip install -e .
pip install -r requirements.txt
```

### Running Tests

```bash
python -m unittest discover 
```

## Examples

The `examples/estimators/` directory contains comprehensive examples of using the EstimatorFactory:

- `basic_usage_example.py` - Basic workflow demonstration
- `config_creation_example.py` - Creating custom configurations
- `debugging_examples.py` - Error handling and debugging

## References

- Shahnazarian, H. (2004). A Dynamic Microeconometric Simulation Model for Incorporated Businesses.

