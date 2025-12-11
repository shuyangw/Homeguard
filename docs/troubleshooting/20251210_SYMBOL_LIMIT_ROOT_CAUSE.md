# Symbol Limit Root Cause Analysis - December 10, 2025

## Executive Summary

**Problem**: OMR strategy generated 0 signals on Dec 10, 2025 at 3:50 PM execution.

**Root Cause**: Alpaca IEX WebSocket has a 405 symbol limit. Multi-strategy deployment attempted to subscribe to 523 symbols (503 RAMP + 19 OMR + 1 SPY), causing silent subscription failure.

**Impact**:
- WebSocket buffer remained empty for all 19 OMR symbols
- Fallback REST API failed due to SIP permission issue (fixed in commit 4ba7407)
- OMR unable to generate signals since Dec 9 deployment

---

## Timeline

### Dec 9, 2025 - 9:12 PM ET (Deployment)
**Commit**: `e941ee6` - Add multi-strategy runner with shared IEX WebSocket connection

**What happened**:
```
Multi-strategy service started
Subscribed to 523 symbols (19 OMR + 503 RAMP + 1 SPY)
error: symbol limit exceeded (405)
```

**Result**: WebSocket subscription silently failed, buffer stayed empty

---

### Dec 10, 2025 - 3:50 PM ET (OMR Execution)
**What happened**:
1. OMR tried to fetch bars from streaming buffer → All 19 symbols empty
2. Fallback triggered → REST API calls for each symbol
3. REST API failed → `subscription does not permit querying recent SIP data`
4. 0 symbols retrieved → 0 signals generated → No trades

**Why REST API failed**:
- FallbackPoller was using SIP feed (default) instead of IEX
- Fixed in commit `4ba7407` by passing `feed` parameter

---

## Error Evidence

**Startup error (logged 4 times)**:
```
Dec 10 02:12:24 UTC: error: symbol limit exceeded (405)
Dec 10 22:58:28 UTC: error: symbol limit exceeded (405)
Dec 11 00:34:14 UTC: error: symbol limit exceeded (405)
```

**OMR execution errors (logged 19 times at 3:50 PM)**:
```
Buffer empty for FAZ, falling back to REST API
Failed to fetch bars for FAZ: {"message":"subscription does not permit querying recent SIP data"}
[OMR] No bars in buffer for FAZ
... (repeated for all 19 symbols)
```

**Result**:
```
[OMR] Retrieved 0 symbols from streaming buffer
Signal evaluation: 0 symbols checked, 0 passed filters
Generated 0 signals for overnight holding
```

---

## Why This Wasn't Caught

1. **Silent failure**: Alpaca SDK logs `error: symbol limit exceeded (405)` but doesn't throw exception
2. **Buried in logs**: Single line among hundreds of startup logs, no ERROR/CRITICAL prefix
3. **Service appeared healthy**: RAMP worked fine using fallback, OMR just silently skipped
4. **No monitoring**: No alerts for "0 signals generated" condition

---

## Symbol Count Breakdown

| Source | Count |
|--------|-------|
| RAMP (S&P 500) | 503 |
| OMR (Leveraged ETFs) | 19 |
| SPY (Market data) | 1 |
| **Total** | **523** |
| **Alpaca IEX Limit** | **405** |
| **Exceeded by** | **118** |

---

## Solution Options

### Option 1: Prioritize Symbols (Quick Fix)
Subscribe only to:
- OMR's 19 symbols (always needed)
- RAMP's current top 20 positions (dynamic)
- SPY
- **Total: ~40 symbols** (well under 405 limit)

**Pros**: Simple, fast to implement
**Cons**: RAMP won't have full S&P 500 streaming (will use fallback for rest)

### Option 2: Separate WebSocket Connections
Run OMR and RAMP in separate processes with separate WebSocket connections

**Pros**: Each strategy gets full streaming
**Cons**: Violates IEX 1 connection limit (may need to upgrade to paid tier)

### Option 3: OMR Uses REST API Only
Remove OMR symbols from WebSocket subscription, use fallback exclusively

**Pros**: RAMP gets full streaming, OMR uses tested REST path
**Cons**: OMR slower (but only 19 symbols, acceptable)

**Recommended**: Option 3 - Use REST API for OMR (with IEX feed fix already applied)

---

## Fixes Applied

### Commit `4ba7407` - Add streaming diagnostics and fix FallbackPoller feed bug

**What was fixed**:
1. FallbackPoller now uses IEX feed instead of defaulting to SIP
2. Added diagnostic logging to show:
   - When bars are received and stored in buffer
   - WebSocket connection status and subscription counts
   - Fallback REST API calls with feed type

**What remains**:
- Symbol limit issue still exists (will hit again on next restart)
- Need to implement one of the solution options above

---

## Monitoring Improvements

**Added logging** (commit 4ba7407):
```
[Buffer] Created buffer for TQQQ
[Buffer] TQQQ: 1 bars stored (latest: 09:31:00 @ $45.23)
[WebSocket] Subscribed to bars for 523 symbols (sample: ['TQQQ', 'SOXL', ...])
[WebSocket] Starting connection with IEX feed
[Fallback] Fetched 350 bars for TQQQ via IEX
```

**Recommended alerts**:
1. Alert if "symbol limit exceeded" appears in logs
2. Alert if OMR generates 0 signals during market hours
3. Alert if WebSocket buffer is empty for >10 minutes after market open

---

## Lessons Learned

1. **Test subscription limits**: Always verify API limits before deployment
2. **Monitor silent failures**: Errors without exceptions can go unnoticed
3. **Alert on zero signals**: Strategies generating 0 signals may indicate data issues
4. **Log buffer status**: Track which symbols are actually receiving streaming data

---

## Next Steps

1. Choose solution option (recommend Option 3)
2. Implement chosen fix
3. Deploy to EC2 with monitoring
4. Verify OMR generates signals at next 3:50 PM execution
5. Add automated alerts for zero-signal conditions
