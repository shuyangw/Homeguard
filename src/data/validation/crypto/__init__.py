"""Crypto domain validation -- placeholder.

To implement:
1. Mirror src/data/validation/futures/ structure.
2. Density expectations differ (24/7 trading, but volume varies sharply by hour
   and by exchange).
3. Currently in store: BTC/ETH/major-alts in pair format (e.g. BTC_USD).
4. Cross-source checks: Coinbase spot vs CME crypto futures basis is the
   natural sanity check.
5. Note: crypto datasets currently use [ns, UTC] dtype (off-spec); validation
   should flag this as documented dtype drift.

Reference doc: docs/superpowers/specs/2026-05-09-data-validation-framework-and-additional-pull-design.md
"""
