"""Phase 2a of the equity-options harness: the shared primitives and the two
harness modules the whole slate is built on.

Contents (registered in `2026-07-25_options_slate_cc_handoff_spec_v2.md`):

    snapshot.py     the ONE snapshot-symmetry guard (Section 1.3 rule 1)
    marks.py        M2 -- the mark convention, as amended by A3
    cost_model.py   M3 -- `cost_model_v2`, parameterized from `spread_census`
    primitives.py   P1-P10

DELIBERATELY NOT BUILT HERE (Phase 2b): M1 early assignment, M4 P&L
decomposition, M5 regime attribution, M6 CPCV wrapper, the walk-forward runner.
Nothing in this package computes P&L, holds a position, or produces a strategy
verdict. Those sit on top; there are no stubs for them here.
"""
