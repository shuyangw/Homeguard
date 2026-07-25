# Futures Sharpe-Uplift - Next Steps TODO (UNTRACKED - do not git-commit)

Created 2026-07-04. Working list; not tracked in git per user request.
Context: campaign closed, carry_idm 0.76 best, crypto a real uncorrelated pillar (0.61, rho -0.065)
but unharvestable under pre-committed IDM sizing. Full narrative:
docs/progress/20260704_SHARPE_UPLIFT_PHASE1_SUMMARY.md.

## A. Infrastructure fixes (unblock efficient runs) -- STARTING HERE
- [x] A1. **Daily-panel cache (OOM fix, keystone). DONE + MERGED + PUSHED (acd0db7).** Daily-raw +
      roll-volume disk caches. RESULT: per-window RSS 5.6GB->0.33GB, 8-way 45GB->2.6GB, walk-forward
      47min->17s (~165x), carry_idm 0.7646 + crypto combo 0.4217 BYTE-IDENTICAL. Both OOM + CLI-kill
      bugs (memory-driven) resolved. NOTE: also fixed a __main__-guard spawn-bomb in the throwaway
      verify scripts (not shipped code).
- [x] A2. **CLI ~13-16min kill -- RESOLVED by A1** (was memory-driven; 35-root crypto CLI now 18s).
      (superseded) A2 original text: Likely same marginal-memory (A1 may fix it) or a
      report-phase issue. Add completion sentinel; trim per-run standard-report work. Effort: small-med.
- [ ] A3. **Stopgap: lower --jobs to 4 for large-universe/crypto runs** until A1 lands. Effort: trivial.

## B. Crypto-harvesting research (the real >1 path)
- [x] B4. **Small-sleeve combiner DONE + MERGED (a88d5ba, not pushed).** carry+crypto@15% = 0.81/PBO0.177,
      beats carry 0.76/0.19 on both axes. Non-fitting (15% pre-registered). 1.5x-cost gate not yet on blend.
  (orig) B4. **Proper portfolio small-sleeve combiner.** Combine carry + crypto as
      two return streams with a PRE-REGISTERED small crypto weight (risk-parity by instrument or fixed
      satellite %), NOT IDM's full 1/8 cluster. The legitimate non-fitting way to harvest rho -0.065.
      Effort: medium. EV: the one path that could still reach ~0.9-1.0.
- [ ] B5. **Acquire perp funding-rate data.** External exchange-funding pipeline -> stronger crypto carry
      than CME calendar; could push crypto standalone well above 0.61. Effort: large. EV: high if lands.
- [ ] B6. **Crypto robustness gate.** 0.61/PBO0.24 low-confidence (2 roots, 7 windows, regime-heavy):
      subperiod stability [DONE: robust -- positive in 2020-21 AND 2023-26, blend adds across regimes,
      2022 crash drag contained by 15% weight], best-of-N deflation [TODO], crypto capacity [TODO]. Do BEFORE deploy.

## C. Incremental carry levers (toward ~0.85, not >1)
- [ ] C7. **Multi-horizon carry** (needs B4 combiner): combine 2-3 carry horizons via FDM. ~+0.02-0.08.
- [ ] C8. **Buffering:** position buffer to cut turnover -> net Sharpe. ~+0.02-0.05, reliable.
- [x] (IDM per-instrument cap: TESTED -> hurts carry alone 0.71 < 0.76. Closed.)

## D. Decisions / bank-what-we-have
- [x] D9. **Campaign + A1 MERGED + PUSHED to origin/main (acd0db7, 2026-07-05).**
- [ ] D10. **Deploy carry_idm (0.76)?** Honest best futures book. Portfolio integration vs OMR/RAMP/CSCM,
      capacity, then IBKR paper. Decide.

## Recommended sequence
A1 (cache) -> B4 (small-sleeve combiner) gated by B6 (robustness) -> B5 (funding, big swing) ->
C7/C8 (polish). D9/D10 (bank carry) can run in parallel.
