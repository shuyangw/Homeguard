"""Equities domain validation -- placeholder.

To implement:
1. Mirror src/data/validation/futures/ structure under this directory:
   - expectations.py (per-symbol density, ranges, listing dates, known events)
   - checks/structural.py / statistical.py / cross_source.py / external.py
2. Define expected_density per the equities universe (Russell 3000 has different
   density than futures -- RTH only, ~390 bars/day for liquid stocks).
3. Cross-source checks differ from futures: equities have splits/dividends/
   corporate actions instead of contract rolls.
4. Add tests under tests/data/validation/equities/.
5. Wire into scripts/data/run_validation.py CLI by adding "equities" to the
   set of allowed --domain values.

Reference doc: docs/superpowers/specs/2026-05-09-data-validation-framework-and-additional-pull-design.md
"""
