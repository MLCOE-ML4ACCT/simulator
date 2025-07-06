# Financial Simulator

A Python-based financial simulation engine that models firm balance sheets and cash flows based on the theoretical framework from Shahnazarian (2004).

## Overview

This simulator implements a comprehensive financial model that tracks firm state variables and flow variables across time periods. It follows the balance sheet identities and accounting relationships described in academic literature to provide realistic financial projections.

## Project Structure

```
simulator/
├── data_models/           # Core data structures
│   ├── firm_state.py     # Balance sheet state representation
│   └── flow_variables.py # Period flow variables
├── estimators/           # Forecasting strategies
│   ├── base_estimator.py # Abstract base class
│   └── dummy_estimator.py # Simple test implementation
├── theoretical/          # Simulation engines
│   ├── base_theoretical.py # Abstract engine interface
│   └── simple_theoretical.py # Main simulation engine
├── tests/               # Unit tests
│   ├── test_balance_sheet_identities.py
│   ├── test_cash_flow.py
│   └── conftest.py      # Test fixtures
├── docs/                # Documentation
│   └── variables.md     # Variable definitions
└── requirements.txt     # Dependencies
```

## Quick Start

### Installation

```bash
# Python 3.10
pip install -r requirements.txt
```

### Basic Usage

```python
python test_simulator.py
```

### Running Tests

```bash
# Run all tests
pytest
```
Note: the test case is incomplete

## References

- Shahnazarian, H. (2004). Corporate Financial Decision Making: A Simulation Model.

