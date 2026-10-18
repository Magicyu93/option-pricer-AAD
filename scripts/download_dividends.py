from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.common import DEFAULT_DATA_DIR
from data_pipeline.dividends import DividendsDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw announced stock/ETF dividend events."
    )
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    frame = DividendsDataLoader(data_dir=args.data_dir).load(
        args.tickers,
        args.start,
        args.end,
        refresh=args.refresh,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False, compression="zstd")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
