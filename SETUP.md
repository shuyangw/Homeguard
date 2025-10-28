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
