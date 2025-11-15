# EC2 Scripts - Fix Notes

## Issue Fixed: `ssh.bat` Infinite Loop

**Problem**: The script `ssh.bat` was looping endlessly when executed.

**Root Cause**: Windows batch file name conflict. When a script named `ssh.bat` calls the `ssh` command, Windows searches for executables in this order:
1. Current directory
2. PATH directories

Since `ssh.bat` was in the current directory, Windows found the batch file itself instead of the actual `ssh.exe` executable, causing infinite recursion.

**Solution**: Renamed the SSH connection scripts to avoid the conflict:
- `ssh.bat` → `connect.bat`
- `ssh.sh` → `connect.sh`

---

## Updated Files

### Scripts Renamed:
- ✓ `scripts/ec2/ssh.bat` → `scripts/ec2/connect.bat`
- ✓ `scripts/ec2/ssh.sh` → `scripts/ec2/connect.sh`

### Documentation Updated:
- ✓ `scripts/ec2/SSH_SCRIPTS_README.md` - All references updated

---

## Current Scripts (All Working)

| Script | Purpose | Status |
|--------|---------|--------|
| **connect.bat** | SSH to instance (Windows) | ✓ Fixed |
| **connect.sh** | SSH to instance (Linux/Mac) | ✓ Working |
| **check_bot.bat** | Check status (Windows) | ✓ Working |
| **check_bot.sh** | Check status (Linux/Mac) | ✓ Working |
| **view_logs.bat** | View logs (Windows) | ✓ Working |
| **view_logs.sh** | View logs (Linux/Mac) | ✓ Working |
| **restart_bot.bat** | Restart bot (Windows) | ✓ Working |
| **restart_bot.sh** | Restart bot (Linux/Mac) | ✓ Working |
| **daily_health_check.bat** | Health check (Windows) | ✓ Working |
| **daily_health_check.sh** | Health check (Linux/Mac) | ✓ Working |

---

## Usage

### Connect to EC2 Instance

**Windows**:
```powershell
scripts\ec2\connect.bat
```

**Linux/Mac**:
```bash
scripts/ec2/connect.sh
```

---

## Prevention

To avoid similar issues in the future:
- ✗ **Don't name scripts** the same as system commands (`ssh`, `git`, `python`, etc.)
- ✓ **Use descriptive names** (`connect`, `check_bot`, `view_logs`, etc.)
- ✓ **Test on Windows** where this issue is most common

---

**Fixed**: November 15, 2025
**Issue**: Batch file recursion
**Resolution**: Script renamed
