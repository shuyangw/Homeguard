from __future__ import annotations
import argparse
from datetime import date, datetime
from src.data import fx_pipeline
from src.utils import logger


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser(prog="fx_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build")
    b.add_argument("names", nargs="+")
    b.add_argument("--start", type=_d, default=date(2011, 1, 1))
    b.add_argument("--end", type=_d, default=date.today())
    args = ap.parse_args()
    if args.cmd == "list":
        for c in fx_pipeline.list_components():
            key = c["requires_key"] or "-"
            logger.info(f"{c['name']:22} key={key:12} up_to_date={c['up_to_date']}")
    elif args.cmd == "build":
        fx_pipeline.build(args.names, args.start, args.end)


if __name__ == "__main__":
    main()
