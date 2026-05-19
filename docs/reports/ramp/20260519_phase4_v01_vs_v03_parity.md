# Phase 4 V01 vs V03 Parity Finding

## Question

Does applying crash exposure correctly (V03) improve net Sharpe over the
fresh-portfolio baseline (V01) that ignores crash exposure?

## Side-by-side at 5.0 bps per side

| Metric | V01 | V03 | Delta (V03 - V01) |
|---|---:|---:|---:|
| Sharpe | 0.282 | 0.077 | -0.204 |
| CAGR | 3.74% | -0.84% | -4.58% |
| Max DD | -79.88% | -66.76% | 13.12% |
| Avg turnover | 90.64% | 72.67% | -17.98% |
| Cost drag | 75.28% | 110.88% | 35.60% |

## Conclusion

Pick ONE based on the metrics:

1. **V03 wins net.** Advance to Wave 1 turnover-control on V03 base.
2. **V03 wins gross but loses net to cost.** Phase 3A generalized; turnover-control needed before V03 is viable.
3. **No material difference.** Investigate signal/regime overlay/sector concentration in Phase C.

## Next step

Documented in docs/progress/<this-session>.md at completion.