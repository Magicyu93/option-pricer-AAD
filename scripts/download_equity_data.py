from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.common import DEFAULT_DATA_DIR
from data_pipeline.equity import MassiveHistoricalDataLoader

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Massive option and underlying minute data."
    )
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--match", choices=("exact", "backward"), default="exact")
    parser.add_argument("--tolerance", default="1min")
    parser.add_argument("--include-contract-metadata", action="store_true")
    args = parser.parse_args()

    frame = MassiveHistoricalDataLoader(data_dir=args.data_dir).load_market(
        dates=args.dates,
        tickers=args.tickers,
        match=args.match,
        tolerance=args.tolerance,
        include_contract_metadata=args.include_contract_metadata,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False, compression="zstd")
    print(f"Downloaded {len(frame):,} rows.")


if __name__ == "__main__":
    main()
