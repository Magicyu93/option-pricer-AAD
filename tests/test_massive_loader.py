"""The Massive loader, offline: raw files are written into the cache directory
so nothing reaches S3 or the REST API."""

import json
import os

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from data_pipeline.equity import MassiveHistoricalDataLoader

DATE = "2026-09-08"


def _ns(hhmm: str) -> int:
    return pd.Timestamp(f"{DATE} {hhmm}", tz="America/New_York").value


def _bar(ticker: str, hhmm: str, close: float) -> dict:
    return {
        "ticker": ticker, "volume": 10, "open": close, "close": close,
        "high": close, "low": close, "window_start": _ns(hhmm), "transactions": 3,
    }


OPTION_BARS = [
    _bar("O:SPY260918C00500000", "09:30", 5.0),
    _bar("O:SPY260918C00500000", "09:31", 5.1),
    _bar("O:SPY260918C00500000", "16:05", 5.2),
    _bar("O:SPYG260918C00050000", "09:30", 1.0),
    _bar("O:SPX261218C05000000", "09:30", 90.0),
    _bar("O:SPXW260918P05000000", "09:30", 12.0),
    _bar("O:AAPL260918C00150000", "09:30", 3.0),
    _bar("O:AAPL1260918C00150000", "09:30", 2.0),
]

STOCK_BARS = [
    _bar("SPY", "04:00", 499.0),
    # SIP volume can be fractional, and is written that way in the real files.
    {**_bar("SPY", "09:30", 500.0), "volume": "120.500000"},
    _bar("SPY", "09:32", 501.0),
    _bar("SPYG", "09:30", 50.0),
    _bar("AAPL", "09:30", 150.0),
]


def _raw_path(data_dir, asset: str):
    return data_dir / "raw" / asset / f"date={DATE}" / "minute_massive.csv.gz"


def _write_raw(data_dir, asset: str, rows: list[dict]) -> None:
    path = _raw_path(data_dir, asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


@pytest.fixture
def loader(tmp_path) -> MassiveHistoricalDataLoader:
    _write_raw(tmp_path, "options", OPTION_BARS)
    _write_raw(tmp_path, "stocks", STOCK_BARS)
    # Asset classes already known, so routing never reaches the REST API.
    reference = tmp_path / "reference" / "underlying_asset_types.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"SPY": "stocks", "AAPL": "stocks", "SPX": "indices"}))
    return MassiveHistoricalDataLoader(data_dir=tmp_path, api_key="test", verbose=False)


def test_parses_option_ticker_and_excludes_prefix_collisions(loader):
    df = loader.load_options(DATE, "SPY")

    assert len(df) == 3
    assert set(df["option_ticker"]) == {"O:SPY260918C00500000"}  # not SPYG
    row = df.iloc[0]
    assert row["strike"] == 500.0
    assert row["option_type"] == "call"
    assert row["expiration"] == pd.Timestamp("2026-09-18")
    assert row["underlying"] == "SPY"


def test_column_types(loader):
    df = loader.load_options(DATE, ["SPY", "AAPL"])

    assert df["expiration"].dtype == "datetime64[us]"
    assert pd.api.types.is_float_dtype(df["strike"])
    assert str(df["timestamp"].dt.tz) == "America/New_York"
    assert set(df["session_date"]) == {pd.Timestamp(DATE).date()}


def test_every_bar_is_kept_and_the_caller_picks_the_window(loader):
    options = loader.load_options(DATE, "SPY")
    stocks = loader.load_stocks(DATE, "SPY")

    assert options["timestamp"].dt.strftime("%H:%M").tolist() == ["09:30", "09:31", "16:05"]
    assert stocks["timestamp"].dt.strftime("%H:%M").tolist() == ["04:00", "09:30", "09:32"]


def test_index_underlying_includes_weekly_root(loader):
    df = loader.load_options(DATE, "SPX")

    assert sorted(df["option_root"]) == ["SPX", "SPXW"]
    assert set(df["underlying"]) == {"SPX"}


def test_adjusted_roots_are_not_their_underlying(loader):
    # AAPL1 is a corporate-action adjusted root: a different contract, so not AAPL.
    assert loader.load_options(DATE, "AAPL")["option_root"].tolist() == ["AAPL"]


def test_tickers_are_served_from_cache_once_converted(loader, tmp_path):
    assert len(loader.load_options(DATE, "SPY")) == 3
    assert loader.load_options(DATE, "NOPE").empty  # remembered as empty

    # Neither request may touch the raw file again.
    _raw_path(tmp_path, "options").unlink()
    assert len(loader.load_options(DATE, "SPY")) == 3
    assert loader.load_options(DATE, "NOPE").empty


def test_uncached_loader_reads_raw_and_writes_nothing(tmp_path):
    _write_raw(tmp_path, "stocks", STOCK_BARS)
    loader = MassiveHistoricalDataLoader(data_dir=tmp_path, cache_parquet=False, verbose=False)

    assert len(loader.load_stocks(DATE, "SPY")) == 3
    assert not (tmp_path / "parquet").exists()


def test_cache_is_independent_of_timezone(loader, tmp_path):
    ny = loader.load_stocks(DATE, "SPY")
    utc = MassiveHistoricalDataLoader(data_dir=tmp_path, timezone="UTC", verbose=False).load_stocks(DATE, "SPY")

    assert str(utc["timestamp"].dt.tz) == "UTC"
    assert (ny["timestamp"].values == utc["timestamp"].values).all()


def test_combine_exact_and_backward(loader):
    exact = loader.load_market(DATE, ["SPY", "AAPL"])
    spot = exact.set_index(["underlying", exact["timestamp"].dt.strftime("%H:%M")])["spot"]
    assert spot[("SPY", "09:30")] == 500.0
    assert pd.isna(spot[("SPY", "09:31")])  # no stock bar that minute
    assert spot[("AAPL", "09:30")] == 150.0

    back = loader.load_market(DATE, ["SPY", "AAPL"], match="backward", tolerance="1min")
    spot = back.set_index(["underlying", back["timestamp"].dt.strftime("%H:%M")])["spot"]
    assert spot[("SPY", "09:31")] == 500.0
    assert spot[("AAPL", "09:30")] == 150.0  # never borrows another ticker's bar


def test_combine_warns_when_underlying_has_no_bars(loader, tmp_path):
    _write_raw(tmp_path, "indices", [_bar("I:NDX", "09:30", 20000.0)])  # no SPX in the day's file
    with pytest.warns(RuntimeWarning, match="SPX"):
        df = loader.load_market(DATE, "SPX")
    assert df["spot"].isna().all()


class _FakeS3:
    def __init__(self):
        self.calls = 0

    def download_file(self, bucket, key, path):
        self.calls += 1
        raise ClientError({"Error": {"Code": "404"}}, "GetObject")


def test_missing_day_is_skipped_without_a_partial_download(tmp_path):
    loader = MassiveHistoricalDataLoader(data_dir=tmp_path, verbose=False)
    s3 = loader._s3 = _FakeS3()

    assert loader.load_stocks("2020-01-01", "SPY").empty
    assert s3.calls == 1
    assert not any(p.is_file() for p in (tmp_path / "raw").rglob("*"))


class _InterruptedFrame:
    def to_parquet(self, path, **kwargs):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("interrupted")


def test_cache_write_leaves_nothing_behind_on_failure(tmp_path):
    target = tmp_path / "x.parquet"

    with pytest.raises(RuntimeError):
        MassiveHistoricalDataLoader._write_parquet(_InterruptedFrame(), target)
    assert list(tmp_path.iterdir()) == []


def test_cache_files_get_umask_permissions(loader, tmp_path):
    loader.load_stocks(DATE, "SPY")

    umask = os.umask(0)
    os.umask(umask)
    mode = loader._cache_path("stocks", DATE, "SPY").stat().st_mode & 0o777
    assert mode == 0o666 & ~umask


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_contract_metadata_queries_both_expiry_states_and_caches_empty(loader, monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        results = []
        if params["underlying_ticker"] == "SPY" and params["expired"] == "true":
            results = [{
                "ticker": "O:SPY260918C00500000", "underlying_ticker": "SPY",
                "exercise_style": "american", "shares_per_contract": 100,
            }]
        return _FakeResponse({"results": results})

    monkeypatch.setattr(loader._http, "get", fake_get)

    df = loader.load_options(DATE, "SPY", include_contract_metadata=True)
    assert set(df["exercise_style"]) == {"american"}
    assert {c["expired"] for c in calls} == {"false", "true"}
    assert all(c["as_of"] == DATE for c in calls)

    assert loader.load_contract_metadata(DATE, "NOPE").empty
    n = len(calls)
    assert loader.load_contract_metadata(DATE, "NOPE").empty
    loader.load_options(DATE, "SPY", include_contract_metadata=True)
    assert len(calls) == n  # both served from cache, empty one included


def test_rest_session_retries_throttling(loader):
    retry = loader._http.get_adapter("https://api.massive.com").max_retries
    assert 429 in retry.status_forcelist
    assert retry.respect_retry_after_header


def test_cache_paths_are_partitioned_and_name_the_source(tmp_path):
    loader = MassiveHistoricalDataLoader(data_dir=tmp_path, api_key="test")
    assert loader._raw_path("options", DATE) == (
        tmp_path / "raw/options" / f"date={DATE}" / "minute_massive.csv.gz"
    )
    assert loader._cache_path("stocks", DATE, "SPY") == (
        tmp_path / "parquet/stocks" / f"date={DATE}" / "ticker=SPY/minute_massive.parquet"
    )
    assert loader._contract_cache_path(DATE, "SPY") == (
        tmp_path / "parquet/contracts" / f"date={DATE}" / "ticker=SPY/contracts_massive.parquet"
    )
