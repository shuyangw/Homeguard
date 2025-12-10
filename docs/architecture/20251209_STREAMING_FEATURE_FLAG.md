# Streaming Feature Flag Guide

**Date**: 2025-12-09
**Feature**: `USE_STREAMING` environment variable

---

## Quick Start

### Enable Streaming

Add to your `.env` file:
```bash
USE_STREAMING="true"
STREAMING_FEED="iex"  # or "sip" for paid feed
```

Restart the bot:
```bash
sudo systemctl restart homeguard-omr
sudo systemctl restart homeguard-ramp
```

### Disable Streaming (Default)

Set in `.env`:
```bash
USE_STREAMING="false"
```

Or simply omit the variable (defaults to false).

---

## Feature Flag Behavior

### When `USE_STREAMING=false` (Default)

**Data Provider**: `CompositeDataProvider` (Alpaca → yfinance fallback)

**Boot Sequence**:
```
Initializing Alpaca broker...
Connected to Alpaca Paper Trading
Creating data provider (Alpaca -> yfinance fallback)...
Note: Streaming is disabled. Set USE_STREAMING=true in .env to enable.
Data provider ready: composite
Creating omr adapter...
```

**OMR Execution (3:50 PM)**:
```
[OMR] No data provider, fetching from broker...
  → broker.get_historical_bars() × 15 symbols
  → 7.5 seconds
```

**RAMP Execution (3:55 PM)**:
```
[RAMP] No LiveDataProvider, fetching from broker API...
  → broker.get_historical_bars() × 500 symbols
  → 150 seconds
```

---

### When `USE_STREAMING=true`

**Data Provider**: `LiveDataProvider` (WebSocket streaming)

**Boot Sequence**:
```
Initializing Alpaca broker...
Connected to Alpaca Paper Trading

================================================================================
STREAMING DATA ENABLED
================================================================================
Creating LiveDataProvider with IEX feed...
  OMR symbols: 15
  RAMP symbols: 503
Total unique symbols: 515
Starting WebSocket connection...
Streaming enabled: 515 symbols
Feed: IEX
================================================================================

Creating omr adapter...
```

**OMR Execution (3:50 PM)**:
```
[OMR] Fetching intraday data from LiveDataProvider (streaming)...
[OMR] Retrieved 15 symbols from streaming buffer
  → provider.get_bars() × 15 symbols
  → 150 milliseconds
```

**RAMP Execution (3:55 PM)**:
```
[RAMP] Using LiveDataProvider streaming buffer for today's closes...
[RAMP] Retrieved 503/503 symbols from streaming buffer (0 failed)
  → provider.get_bars() × 503 symbols
  → 500 milliseconds
```

---

## Environment Variables

### `USE_STREAMING`

**Type**: Boolean string
**Default**: `"false"`
**Valid Values**:
- `"true"`, `"1"`, `"yes"` → Enable streaming
- `"false"`, `"0"`, `"no"`, (empty) → Disable streaming

**Example**:
```bash
USE_STREAMING="true"
```

### `STREAMING_FEED`

**Type**: String
**Default**: `"iex"`
**Valid Values**:
- `"iex"` → IEX feed (free, ~2-10% of trades)
- `"sip"` → SIP feed (paid subscription, 100% of trades)

**Only used when** `USE_STREAMING=true`

**Recommendation**:
- Paper Trading: Use `"iex"` (free)
- Real Money: Use `"sip"` (paid, required for production)

**Example**:
```bash
STREAMING_FEED="iex"
```

---

## Deployment Scenarios

### Scenario 1: Gradual Rollout (Recommended)

**Week 1**: Deploy code with `USE_STREAMING=false`
- ✅ Code deployed (streaming infrastructure dormant)
- ✅ Backward compatible (polling continues)
- ✅ Zero risk

**Week 2**: Enable for OMR only
```bash
# On EC2: /home/ec2-user/Homeguard/.env
USE_STREAMING="true"
STREAMING_FEED="iex"
```

```bash
sudo systemctl restart homeguard-omr
# Leave RAMP on polling for now
```

**Monitor**: Check logs for WebSocket stability

**Week 3**: Enable for RAMP if OMR stable
```bash
sudo systemctl restart homeguard-ramp
```

**Monitor**: Verify 32x performance improvement

---

### Scenario 2: Instant Rollout

**Deploy and enable immediately**:

1. Deploy code to EC2
2. Update `.env`:
   ```bash
   USE_STREAMING="true"
   STREAMING_FEED="iex"
   ```
3. Restart both services:
   ```bash
   sudo systemctl restart homeguard-omr
   sudo systemctl restart homeguard-ramp
   ```

**Risk**: Both strategies switch to streaming simultaneously

---

### Scenario 3: A/B Testing

**Run both in parallel** (requires two EC2 instances):

**Instance 1** (control):
```bash
USE_STREAMING="false"
```

**Instance 2** (test):
```bash
USE_STREAMING="true"
```

Compare performance and stability over 1-2 weeks.

---

## Verification

### Check Current Mode

**Via Logs** (startup output):
```bash
journalctl -u homeguard-omr -n 50 | grep -i streaming
```

**Streaming Disabled**:
```
Note: Streaming is disabled. Set USE_STREAMING=true in .env to enable.
Data provider ready: composite
```

**Streaming Enabled**:
```
STREAMING DATA ENABLED
Creating LiveDataProvider with IEX feed...
Streaming enabled: 515 symbols
Feed: IEX
```

---

### Check Execution Performance

**OMR at 3:50 PM** (streaming):
```bash
journalctl -u homeguard-omr -S "15:50" -U "15:51" | grep "Retrieved.*symbols"
```

Expected output:
```
[OMR] Retrieved 15 symbols from streaming buffer
```

**RAMP at 3:55 PM** (streaming):
```bash
journalctl -u homeguard-ramp -S "15:55" -U "15:56" | grep "Retrieved.*symbols"
```

Expected output:
```
[RAMP] Retrieved 503/503 symbols from streaming buffer (0 failed)
```

---

## Fallback Behavior

### Smart Fallback for Mid-Day Restarts (NEW)

**Problem**: If bot restarts at 2 PM, buffer only has bars from 2:01 PM forward. OMR needs 390 bars from 9:30 AM to calculate intraday moves.

**Solution**: Hub automatically detects insufficient data (<90% of requested bars) and fetches complete data from 9:30 AM via REST API.

**Logs**:
```
[WARNING] Buffer for TQQQ has 100/390 bars (25.6%). Falling back to REST API for complete data from market open.
[INFO] Fetching TQQQ bars from market open (09:30) to now (14:30) via REST API
[WARNING] [OMR] TQQQ has 100/390 bars (25.6%). Streaming buffer may be incomplete (recent restart?).
```

**Impact**:
- First execution after restart is slower (~5-10s instead of <1s)
- Subsequent executions use fully populated buffer (fast)
- Strategy gets correct data regardless of restart time

**Threshold**: Buffer considered "sufficient" if ≥90% complete (e.g., 351/390 bars)

---

### WebSocket Disconnects

**Automatic**: Streaming provider falls back to REST API polling

**Logs**:
```
Buffer empty for TQQQ, falling back to REST API
```

**Impact**: Strategy continues (slower, but functional)

---

### Buffer Not Populated (9:30-9:35 AM)

**First 5 minutes after market open**, buffer may be sparse.

**Automatic**: Falls back to REST API for missing data

**Impact**: Minimal (OMR/RAMP don't execute during this window)

---

## Performance Monitoring

### Expected Timings (with Streaming)

| Strategy | Time | Event | Duration |
|----------|------|-------|----------|
| **OMR** | 3:50 PM | Data fetch (15 symbols) | <200ms |
| **OMR** | 3:50 PM | Signal generation | ~500ms |
| **OMR** | 3:50 PM | Order execution | ~1s |
| **OMR** | 3:50 PM | **Total** | **~1.7s** |
| **RAMP** | 3:55 PM | Data fetch (500 symbols) | <1s |
| **RAMP** | 3:55 PM | Signal generation | ~1s |
| **RAMP** | 3:55 PM | Order execution | ~2s |
| **RAMP** | 3:55 PM | **Total** | **~4s** |

---

## Troubleshooting

### Streaming Not Starting

**Check 1**: Verify `.env` has correct values
```bash
cat /home/ec2-user/Homeguard/.env | grep STREAMING
```

**Check 2**: Verify credentials are valid
```bash
cat /home/ec2-user/Homeguard/.env | grep ALPACA_PAPER
```

**Check 3**: Check logs for errors
```bash
journalctl -u homeguard-omr -n 100 | grep -i "websocket\|streaming\|error"
```

---

### WebSocket Connection Failures

**Symptom**: Logs show "WebSocket error" or "Connection failed"

**Cause 1**: Invalid API credentials
- **Fix**: Update `ALPACA_PAPER_KEY_ID` and `ALPACA_PAPER_SECRET_KEY` in `.env`

**Cause 2**: Network issues
- **Fix**: Check EC2 security groups allow outbound HTTPS (port 443)

**Cause 3**: Alpaca API down
- **Fix**: Wait for Alpaca to restore service (fallback to polling automatically)

---

### Performance Not Improving

**Check**: Verify streaming is actually enabled

```bash
journalctl -u homeguard-omr -n 100 | grep "STREAMING DATA ENABLED"
```

**If not found**: Streaming is disabled - check `.env` configuration

**If found but slow**: Check fallback logs

```bash
journalctl -u homeguard-omr | grep "falling back to REST API"
```

**If many fallbacks**: WebSocket connection unstable - investigate network

---

## Rollback Procedure

### Emergency Rollback (Immediate)

**Disable streaming** without redeploying code:

```bash
# SSH to EC2
ssh -i ~/.ssh/homeguard-trading.pem ec2-user@<EC2_IP>

# Update .env
cd /home/ec2-user/Homeguard
sed -i 's/USE_STREAMING="true"/USE_STREAMING="false"/' .env

# Restart services
sudo systemctl restart homeguard-omr
sudo systemctl restart homeguard-ramp

# Verify
journalctl -u homeguard-omr -n 20 | grep "Data provider ready"
```

Expected output:
```
Data provider ready: composite
```

**Total time**: <1 minute

---

## Production Checklist

Before enabling `USE_STREAMING=true` in production:

- [ ] Tested on paper trading for 1+ week
- [ ] WebSocket connection stable (no frequent disconnects)
- [ ] Performance improvement verified (32x faster)
- [ ] Logs clean (no errors during market hours)
- [ ] Upgrade to SIP feed if using real money (`STREAMING_FEED="sip"`)
- [ ] Monitor memory usage (should be <250MB per strategy)
- [ ] Verify fallback works (test by killing WebSocket connection)
- [ ] Document rollback procedure for team

---

## Summary

| Aspect | Polling (`USE_STREAMING=false`) | Streaming (`USE_STREAMING=true`) |
|--------|--------------------------------|----------------------------------|
| **Default** | Yes | No (opt-in) |
| **Performance** | 157s execution | 5s execution |
| **Latency** | 300-500ms per symbol | <10ms per symbol |
| **Network** | 515 API calls per execution | 1 WebSocket connection |
| **Memory** | ~200MB | ~250MB |
| **Complexity** | Simple (proven) | More complex (monitoring required) |
| **Risk** | None (current system) | Low (auto-fallback) |
| **Recommendation** | Safe default | Enable after testing |

---

## Quick Reference

**Enable Streaming**:
```bash
echo 'USE_STREAMING="true"' >> .env
echo 'STREAMING_FEED="iex"' >> .env
sudo systemctl restart homeguard-omr homeguard-ramp
```

**Disable Streaming**:
```bash
sed -i 's/USE_STREAMING="true"/USE_STREAMING="false"/' .env
sudo systemctl restart homeguard-omr homeguard-ramp
```

**Check Status**:
```bash
journalctl -u homeguard-omr -n 30 | grep -i "streaming\|data provider"
```
