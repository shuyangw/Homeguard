"""Scheduled decision times (ET). The adapters hardcode these; test_decision_times pins them."""

DECISION_TIMES: dict[str, tuple[str, ...]] = {
    "ramp": ("15:55",),
    "omr": ("09:31", "15:50"),
}
