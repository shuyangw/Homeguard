"""FX domain validation -- placeholder.

To implement:
1. Mirror src/data/validation/futures/ structure.
2. Density: ~24h trading on majors but very thin overnight on minors.
3. Currently in store: ~9k FX symbols across major and exotic crosses.
4. Note: fx_1min currently uses [ns, UTC] dtype (off-spec).
5. Cross-source checks: agreement between cross-rates (e.g. EURUSD x USDJPY ~ EURJPY)
   is the natural internal consistency check.

Reference doc: docs/superpowers/specs/2026-05-09-data-validation-framework-and-additional-pull-design.md
"""
