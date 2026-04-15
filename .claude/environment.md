# Python Environment Setup

**CRITICAL**: All Python code execution, testing, and development MUST use the `fintech` conda environment.

## Environment Location

The conda installation is located at: `C:\Users\qwqw1\anaconda3`

The fintech environment is at: `C:\Users\qwqw1\anaconda3\envs\fintech`

## Activating the Environment

Before running any Python code, tests, or installing packages:

```bash
conda activate fintech
```

## When to Use fintech Environment

- Running backtests (`python backtest_scripts/test_5x15_five_year_analysis.py`)
- Running unit tests (`pytest tests/`)
- Installing Python packages (`pip install ...` or `conda install ...`)
- **Downloading data** (`python scripts/data/download_all_data.py`)
- Running live trading (`python scripts/trading/run_live_paper_trading.py`)
- Any Python script execution in this project

## Verifying Active Environment

Check which environment is active:
```bash
conda info --envs
```

The active environment will have an asterisk (*) next to it.

## Quick Start: Download Trading Data

After activating the environment, download all necessary data:

```bash
conda activate fintech
python scripts/data/download_all_data.py
```

See `docs/guides/DATA_DOWNLOAD_GUIDE.md` for detailed instructions.

## AWS EC2 Environment

On AWS EC2 instances, use the virtual environment instead:

```bash
# Activate venv
source ~/Homeguard/venv/bin/activate

# Download data
python scripts/data/download_all_data.py --yes
```

## Troubleshooting

**Issue**: Commands fail with module import errors
**Solution**: Verify you're in the fintech environment

**Issue**: Wrong Python version
**Solution**: Activate fintech environment before running scripts

**Issue**: Package not found
**Solution**: Install package in fintech environment: `conda activate fintech && pip install <package>`

**Issue**: Data download errors
**Solution**: See `docs/guides/DATA_DOWNLOAD_GUIDE.md` for troubleshooting steps
