# Setup Instructions

## Initial Configuration

### 1. Environment Variables

Copy the example file and add your API credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your Alpaca API credentials:

```ini
API_KEY=your_alpaca_api_key_here
API_SECRET=your_alpaca_api_secret_here
```

**WARNING: Never commit `.env` to Git!** It's already in `.gitignore`.

### 2. Settings Configuration

Copy the settings template:

```bash
cp settings.ini.example settings.ini
```

Edit `settings.ini` and update paths for your system:

**Windows:**
```ini
[windows]
local_storage_dir = C:\Your\Preferred\Path\Stock_Data
api_threads = 16
```

**macOS:**
```ini
[macos]
local_storage_dir = /Users/your_username/data/stock_data
api_threads = 6
```

**Linux:**
```ini
[linux]
local_storage_dir = /home/your_username/stock_data
api_threads = 8
```

**WARNING: Never commit `settings.ini` to Git!** It contains personal paths and is in `.gitignore`.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Security Notes

- `.env` - Contains API secrets, never commit
- `settings.ini` - Contains personal paths, never commit
- `.env.example` - Safe template, can commit
- `settings.ini.example` - Safe template, can commit

---

## Deployment Options

After initial setup, you can run Homeguard in two ways:

### Option 1: Local Execution (Default)

Run backtests and trading directly on your local machine.

**Best for**:
- Development and testing
- Running backtests
- Learning and experimentation

**Setup**: Complete steps 1-3 above, then follow:
- [Backtesting Guide](docs/guides/BACKTESTING_GUIDE.md) - For backtesting
- [Quick Start Trading](docs/guides/QUICK_START_TRADING.md) - For live paper trading

### Option 2: Cloud Deployment (Production)

Deploy trading bot to AWS EC2 with automated scheduling and monitoring.

**Best for**:
- 24/7 automated trading
- Production trading bot
- Remote monitoring and management

**Key Features**:
- ✅ Automated start/stop scheduling (9 AM - 4:30 PM ET Mon-Fri)
- ✅ Systemd service with auto-restart on failure
- ✅ SSH management scripts for easy monitoring
- ✅ ~$7/month cost (46% savings vs 24/7 operation)

**Setup Guides**:
- **[Quick Start Deployment](docs/guides/QUICK_START_DEPLOYMENT.md)** - ⭐ Fast 5-minute cloud deployment
- **[Complete Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md)** - Comprehensive Windows/Mac/Linux setup
- **[Infrastructure Overview](docs/INFRASTRUCTURE_OVERVIEW.md)** - Complete AWS architecture and cost breakdown

**Management**:
- **[SSH Scripts Documentation](scripts/ec2/SSH_SCRIPTS_README.md)** - Quick-access management scripts (10 scripts)
- **[Health Check Cheatsheet](docs/HEALTH_CHECK_CHEATSHEET.md)** - Monitoring and troubleshooting guide

**Current Deployment Info**:
- Instance IP: See `.env` file (`EC2_IP` variable)
- Instance ID: See `.env` file (`EC2_INSTANCE_ID` variable)
- Region: us-east-1 (N. Virginia)
- Service: homeguard-trading.service (systemd)

---

## Next Steps

**For Local Development**:
1. Complete initial configuration above
2. Follow [Backtesting Guide](docs/guides/BACKTESTING_GUIDE.md)
3. Explore [Live Paper Trading Guide](docs/guides/LIVE_PAPER_TRADING.md)

**For Cloud Deployment**:
1. Complete initial configuration above
2. Follow [Quick Start Deployment](docs/guides/QUICK_START_DEPLOYMENT.md)
3. Use [SSH Scripts](scripts/ec2/) for daily monitoring
