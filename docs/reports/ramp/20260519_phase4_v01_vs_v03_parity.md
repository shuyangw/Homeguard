# Phase 4 V01 vs V03 Parity Finding

## Question

Does applying crash exposure correctly (V03) improve net Sharpe over the
fresh-portfolio baseline (V01) that ignores crash exposure?

## Side-by-side at 5.0 bps per side

| Metric | V01 | V03 | Delta (V03 - V01) |
|---|---:|---:|---:|
| Sharpe | 0.554 | 0.620 | 0.066 |
| CAGR | 129.22% | 95.04% | -34.18% |
| Max DD | -66.84% | -44.43% | 22.40% |
| Avg turnover | 91.39% | 72.53% | -18.85% |
| Cost drag | 33.17% | 30.61% | -2.56% |

## Conclusion

Pick ONE based on the metrics:

1. **V03 wins net.** Advance to Wave 1 turnover-control on V03 base.
2. **V03 wins gross but loses net to cost.** Phase 3A generalized; turnover-control needed before V03 is viable.
3. **No material difference.** Investigate signal/regime overlay/sector concentration in Phase C.

## Next step

Documented in docs/progress/<this-session>.md at completion.