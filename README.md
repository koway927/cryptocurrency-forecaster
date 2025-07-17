# Cryptocurrency Forecaster

A machine learning project for forecasting cryptocurrency prices using LSTM models and market data analysis.

## Project Structure

```
cryptocurrency-forecaster/
├── data/
│   ├── raw/                    # Raw data files (CSV, etc.)
│   ├── processed/              # Cleaned and preprocessed data
│   └── external/               # External data sources
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── src/                        # Source code modules
│   ├── data/                   # Data processing modules
│   ├── models/                 # Model definitions
│   ├── features/               # Feature engineering
│   └── utils/                  # Utility functions
├── models/                     # Trained model artifacts
├── scripts/                    # Execution scripts
├── configs/                    # Configuration files
├── tests/                      # Unit tests
├── docs/                       # Documentation and reports
├── artifacts/                  # MLflow artifacts
├── mlruns/                     # MLflow experiment tracking
└── requirements.txt            # Python dependencies
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

## Data

- Raw data files are stored in `data/raw/`
- The `iex-campus-cluster/` directory contains IEX market data
- Large CSV files are tracked with DVC (Data Version Control)

## Models

This project implements LSTM neural networks for cryptocurrency price forecasting.

## Usage

See individual notebooks in the `notebooks/` directory for analysis and model training examples.
