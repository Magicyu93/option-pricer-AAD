"""The cache inventory, and the credential check that runs before a download."""

from data_pipeline.common.storage import parse_partitioned_path, partitioned_path
from data_pipeline.coverage import cached_days, coverage
from data_pipeline.download import DownloadResult, credentials_status


def _touch(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_coverage_reads_the_partitioned_layout_and_skips_everything_else(tmp_path):
    pq = tmp_path / "parquet"
    _touch(partitioned_path(pq, "options", {"date": "2026-09-08", "ticker": "SPX"},
                            kind="minute", provider="massive", suffix=".parquet"), 100)
    _touch(partitioned_path(pq, "rates", {"year": 2026, "dataset": "treasury_yields"},
                            kind="daily", provider="massive", suffix=".parquet"))
    _touch(partitioned_path(tmp_path / "raw", "options", {"date": "2026-09-08"},
                            kind="minute", provider="massive", suffix=".csv.gz"))
    _touch(pq / "options" / "date=2026-09-09" / "ticker=SPX" / "minute_massive.empty")
    _touch(pq / "options" / "stray.txt")

    frame = coverage(tmp_path)
    assert len(frame) == 3
    spx = frame[frame["ticker"] == "SPX"].iloc[0]
    assert (spx["layer"], spx["dataset"], spx["date"], spx["kind"], spx["provider"], spx["bytes"]) == (
        "parquet", "options", "2026-09-08", "minute", "massive", 100
    )
    rates = frame[frame["dataset"] == "rates"].iloc[0]
    assert (rates["year"], rates["series"]) == ("2026", "treasury_yields")
    assert cached_days("options", "SPX", tmp_path) == ["2026-09-08"]


def test_an_empty_cache_is_an_empty_inventory(tmp_path):
    assert coverage(tmp_path / "nothing").empty


def test_credentials_follow_the_loaders_rules(monkeypatch):
    for name in ("MASSIVE_S3_ACCESS_KEY", "MASSIVE_S3_SECRET_KEY", "MASSIVE_API_ID",
                 "MASSIVE_API_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert not any(v["ok"] for v in credentials_status().values())

    monkeypatch.setenv("MASSIVE_API_KEY", "k")
    status = {name.split(" (")[0]: v["ok"] for name, v in credentials_status().items()}
    assert status == {"Massive flat files": False, "Massive REST": True, "FRED": False}

    monkeypatch.setenv("MASSIVE_API_ID", "id")  # the legacy pair also unlocks flat files
    assert all(v["ok"] for k, v in credentials_status().items() if k.startswith("Massive"))


def test_download_result_reports_ok_unless_it_failed():
    assert DownloadResult("equity", "2026-09-08", rows=5).ok
    assert DownloadResult("dividends", "SPX", note="index").ok
    assert not DownloadResult("equity", "2026-09-08", error="boom").ok


def test_parse_partitioned_path_inverts_partitioned_path(tmp_path):
    path = partitioned_path(tmp_path, "rates", {"year": 2025, "dataset": "SOFR"},
                            kind="daily", provider="fred", suffix=".parquet")
    assert parse_partitioned_path(tmp_path, path) == {
        "domain": "rates", "partitions": {"year": "2025", "dataset": "SOFR"},
        "kind": "daily", "provider": "fred", "suffix": ".parquet",
    }
    for stray in ("rates/x.parquet", "rates/year=2025/daily_fred.tmp.parquet",
                  "options/date=2026-09-08/ticker=SPX/minute_massive.empty", "notes.txt"):
        assert parse_partitioned_path(tmp_path, tmp_path / stray) is None, stray


def test_read_cached_reads_parquet_as_stored_and_refuses_raw_files(tmp_path):
    import pandas as pd
    import pytest

    from data_pipeline.coverage import frame_summary, read_cached

    path = partitioned_path(tmp_path / "parquet", "options", {"date": "2026-09-08", "ticker": "SPX"},
                            kind="minute", provider="massive", suffix=".parquet")
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"ticker": ["SPX", "SPX"], "price": [1.5, None],
                          "ts": pd.to_datetime(["2026-09-08 15:45", "2026-09-08 15:46"])})
    frame.to_parquet(path, index=False)
    back = read_cached(path)
    assert list(back.columns) == ["ticker", "price", "ts"]  # no columns added from folder names
    with pytest.raises(ValueError):
        read_cached(path.with_suffix(".csv.gz"))

    summary = frame_summary(back).set_index("column")
    assert summary.loc["price", "missing"] == 1 and summary.loc["ticker", "unique"] == 1
    assert summary.loc["ts", "min"] == "2026-09-08 15:45:00"
    assert pd.isna(summary.loc["ticker", "min"])  # strings have no range
