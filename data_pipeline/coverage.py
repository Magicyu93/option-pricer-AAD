"""What is in the cache: every file, with its dataset, partitions and size.

Reads the directory layout that :func:`~data_pipeline.common.storage.partitioned_path`
writes -- ``<root>/<layer>/<dataset>/<key>=<value>/.../<kind>_<provider>.<ext>`` --
and touches no file contents, so it is cheap enough to call on every page load.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common.storage import DEFAULT_DATA_DIR, parse_partitioned_path

COVERAGE_COLUMNS = ["layer", "dataset", "date", "year", "ticker", "series", "kind", "provider", "bytes", "path"]


def coverage(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per cached file under ``raw/`` and ``parquet/``.

    ``date`` is set for daily partitions (equity), ``year`` for yearly ones
    (rates, dividends); ``ticker`` and ``series`` for whichever partition names
    them. Paths are read with :func:`~data_pipeline.common.storage.parse_partitioned_path`,
    the inverse of the function that wrote them; anything else is skipped.
    """
    root = Path(data_dir)
    rows = []
    for layer in ("raw", "parquet"):
        base = root / layer
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            parsed = parse_partitioned_path(base, path) if path.is_file() else None
            if parsed is None:
                continue
            partitions = parsed["partitions"]
            rows.append({
                "layer": layer,
                "dataset": parsed["domain"],
                "date": partitions.get("date"),
                "year": partitions.get("year"),
                "ticker": partitions.get("ticker"),
                "series": partitions.get("dataset"),
                "kind": parsed["kind"],
                "provider": parsed["provider"],
                "bytes": path.stat().st_size,
                "path": str(path),
            })
    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    return frame.sort_values(["layer", "dataset", "date", "year", "ticker"], na_position="first").reset_index(drop=True)


def cached_days(dataset: str, ticker: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> list[str]:
    """Days with a processed (parquet) file for ``ticker`` in ``dataset``, oldest first."""
    frame = coverage(data_dir)
    hit = frame[(frame["layer"] == "parquet") & (frame["dataset"] == dataset) & (frame["ticker"] == ticker)]
    return sorted(hit["date"].dropna().unique().tolist())


def read_cached(path: str | Path) -> pd.DataFrame:
    """A cached parquet file as a frame, read from disk only.

    Unlike the loaders, this never downloads: a loader asked for the current
    year's rates refetches them once a day, while this shows exactly what is
    on disk. Raw downloads (``.csv.gz`` flat files, ``.json`` API responses)
    are refused: they are provider files, and a day's options flat file holds
    the whole market.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError(f"{path.name}: only cached .parquet files can be read this way")
    import pyarrow.parquet as pq

    # Columns as stored: the key=value folders are for whole-dataset reads.
    return pq.read_table(path, partitioning=None).to_pandas()


def frame_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, how much is missing, how varied, and its range.

    ``min``, ``max`` and ``example`` are text: one summary mixes numbers,
    timestamps and strings, which no single column type can hold.
    """
    def text(value) -> str | None:
        return None if value is None else str(value)

    rows = []
    for name, column in frame.items():
        ordered = pd.api.types.is_numeric_dtype(column) or pd.api.types.is_datetime64_any_dtype(column)
        values = column.dropna()
        rows.append({
            "column": name,
            "dtype": str(column.dtype),
            "non_null": int(column.notna().sum()),
            "missing": int(column.isna().sum()),
            "unique": int(column.nunique(dropna=True)),
            "min": text(values.min()) if ordered and len(values) else None,
            "max": text(values.max()) if ordered and len(values) else None,
            "example": text(values.iloc[0]) if len(values) else None,
        })
    return pd.DataFrame(rows)
