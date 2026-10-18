from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.common import DEFAULT_DATA_DIR
from data_pipeline.rates import TREASURY_CMT_SERIES, RatesDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw rate observations (no curve construction). The default "
            "is Treasury constant-maturity yields from Massive; --provider fred "
            "loads the same series from FRED, or any FRED series via --series."
        )
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--provider", choices=("massive", "fred"), default="massive")
    parser.add_argument(
        "--series",
        nargs="+",
        help=f"series IDs; defaults to the Treasury curve {' '.join(TREASURY_CMT_SERIES)}",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    series = [s.upper() for s in args.series or TREASURY_CMT_SERIES]
    frame = RatesDataLoader(args.provider, data_dir=args.data_dir).load(
        series, args.start, args.end, refresh=args.refresh
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False, compression="zstd")

    # One row per date, one column per series, in the order requested.
    table = frame.pivot(index="observation_date", columns="series_id", values="value")
    table = table.reindex(columns=[s for s in series if s in table.columns])
    print(table.to_string())


if __name__ == "__main__":
    main()
