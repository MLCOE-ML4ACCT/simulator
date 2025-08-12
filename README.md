# Financial Simulator

A Python-based financial simulation engine that models firm balance sheets and cash flows based on the theoretical framework from Shahnazarian (2004).

## Overview

This simulator implements a comprehensive financial model that tracks firm state variables and flow variables across time periods. It follows balance sheet identities and accounting relationships to provide realistic financial projections.

## Project Structure

simulator/
```
simulator/
├── coefficient_comparison.ipynb         # Jupyter notebook for coefficient analysis
├── data/                               # Raw data, processed data, and simulation outputs
├── data_models/                        # Core data structures (firm state, flow variables)
├── docs/                               # Documentation and variable definitions
├── estimators/                         # Forecasting models, layers, configs, and utilities
├── examples/                           # Usage examples and demos
├── synthetic_generator/                # Synthetic data generation scripts
├── tests/                              # Unit tests
├── theoretical/                        # Simulation engines and theoretical models
├── utils/                              # Utility functions
├── visualization/                      # Visualization scripts
├── requirements.txt, setup.py, ...     # Project setup and configuration
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


## References

- Shahnazarian, H. (2004). A Dynamic Microeconometric Simulation Model for Incorporated Businesses.

