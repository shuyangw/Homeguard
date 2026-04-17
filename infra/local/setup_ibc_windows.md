# IBC Setup on Windows (Local Dev Machine)

Run TWS via IBC for local development and paper testing.

## Prerequisites

- Interactive Brokers account (paper trading enabled)
- IB Key mobile app installed (for 2FA)

## Step 1: Install TWS Offline

Download the **offline** installer (IBC does NOT work with the auto-updating version):
https://www.interactivebrokers.com/en/trading/tws-offline-installers.php

Install to the default location (`C:\Jts`).

Note the version number (e.g., 10.30) -- you'll need it for Step 3.

## Step 2: Install IBC

Download `IBCWin-3.23.0.zip` from:
https://github.com/IbcAlpha/IBC/releases/tag/3.23.0

Extract to `C:\IBC`.

**Important**: Before extracting, right-click the ZIP -> Properties -> check "Unblock".

## Step 3: Configure

Run the setup script (from project root):

```
infra\local\setup_ibc_config.bat
```

This renders `config\ibkr\ibc-config.ini.template` into `%USERPROFILE%\Documents\IBC\config.ini`
using your `.env` credentials, and creates a TWS launch script.

## Step 4: Launch

```
infra\local\start_tws.bat
```

First launch will prompt for 2FA on IB Key mobile app. After approval, TWS opens
and the API listens on port 4002 (paper) or 4001 (live).

## Step 5: Verify Homeguard can connect

```
python scripts/trading/check_broker_switch.py --strategy omr --to ibkr
```

## Switching between paper and live

Edit `.env`:
- Paper: `IBKR_TRADING_MODE=paper`, `IBKR_GATEWAY_PORT=4002`
- Live: `IBKR_TRADING_MODE=live`, `IBKR_GATEWAY_PORT=4001`

Then re-run `setup_ibc_config.bat` and restart TWS.
