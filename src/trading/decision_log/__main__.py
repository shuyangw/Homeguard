"""Allow running as: python -m src.trading.decision_log <cmd>."""
import sys
from src.trading.decision_log.cli import main

sys.exit(main())
