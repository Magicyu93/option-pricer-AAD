"""The rates loader, offline: fake HTTP sessions stand in for Massive and FRED,
so nothing leaves the machine."""

import datetime as dt
import json

import pandas as pd
import pytest

import data_pipeline.rates.loader as loader_module
from data_pipeline.rates import (
    RATE_COLUMNS,
    FredRatesSource,
    MassiveTreasuryYieldsSource,
    RatesDataLoader,
)

# The shape Massive actually returns: tenors it does not publish are absent,
# not null, and 1M only exists from 2001-07-31.
TREASURY_RECORDS = [
    {"date": "2001-07-27", "yield_3_month": 3.55, "yield_1_year": 3.57, "yield_10_year": 5.13},
    {"date": "2001-07-30", "yield_3_month": 3.57, "yield_1_year": 3.56, "yield_10_year": 5.11},
    {"date": "2001-07-31", "yield_1_month": 3.67, "yield_3_month": 3.54,
     "yield_1_year": 3.53, "yield_10_year": 5.07},
    {"date": "2002-01-02", "yield_1_month": 1.73, "yield_3_month": 1.74,
     "yield_1_year": 2.20, "yield_10_year": 5.16},
]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class ScriptedSession:
    """Replies with the given payloads in order."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return _Response(self.payloads.pop(0))


class MassiveSession:
    """Serves TREASURY_RECORDS filtered by the requested date range."""

    def __init__(self, records=TREASURY_RECORDS):
        self.records = records
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        lo, hi = params["date.gte"], params["date.lte"]
        return _Response(
            {"status": "OK", "results": [r for r in self.records if lo <= r["date"] <= hi]}
        )


class FredSession:
    """Serves one FRED series' metadata and observations."""

    def __init__(self, observations):
        self.observations = observations
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("/series"):
            return _Response({"seriess": [{"id": params["series_id"], "title": "SOFR",
                                           "units": "Percent", "frequency": "Daily"}]})
        lo, hi = params["observation_start"], params["observation_end"]
        rows = [o for o in self.observations if lo <= o["date"] <= hi]
        return _Response({"count": len(rows), "observations": rows})


@pytest.fixture(autouse=True)
def today(monkeypatch):
    """Pin the loader's clock well after the test data."""
    current = {"date": dt.date(2026, 9, 22)}
    monkeypatch.setattr(loader_module, "_today", lambda: current["date"])
    return current


@pytest.fixture
def massive():
    return MassiveSession()


@pytest.fixture
def loader(tmp_path, massive):
    return RatesDataLoader(
        source=MassiveTreasuryYieldsSource(api_key="k", session=massive),
        data_dir=tmp_path,
    )


def _dates(frame):
    return frame["observation_date"].dt.strftime("%Y-%m-%d")


# -- Massive source ---------------------------------------------------------

def test_source_follows_pages_and_redacts_key():
    session = ScriptedSession(
        {"status": "OK", "results": TREASURY_RECORDS[:1],
         "next_url": "https://api.massive.com/fed/v1/treasury-yields?cursor=abc&apiKey=secret"},
        {"status": "OK", "results": TREASURY_RECORDS[1:2]},
    )
    source = MassiveTreasuryYieldsSource(api_key="secret", session=session)
    payload = source.fetch("2001-07-27", "2001-07-30")

    assert [r["date"] for r in payload["records"]] == ["2001-07-27", "2001-07-30"]
    url, params = session.calls[0]
    assert url.endswith("/fed/v1/treasury-yields")
    assert params["date.gte"] == "2001-07-27" and params["date.lte"] == "2001-07-30"
    # next_url already carried the key, so it is not sent twice ...
    assert session.calls[1][1] is None
    # ... and it never reaches the stored raw payload.
    assert "secret" not in json.dumps(payload["pages"])


def test_source_raises_on_error_status():
    source = MassiveTreasuryYieldsSource(
        api_key="k", session=ScriptedSession({"status": "ERROR", "error": "bad key"})
    )
    with pytest.raises(RuntimeError, match="bad key"):
        source.fetch("2025-01-01", "2025-01-31")


# -- load -------------------------------------------------------------------

def test_load_unpivots_to_fred_series_by_date_then_maturity(loader):
    frame = loader.load(["DGS10", "DGS1MO", "DGS3MO", "DGS1"], "2001-07-30", "2001-07-31")

    assert list(frame.columns) == RATE_COLUMNS
    assert (frame["provider"] == "Massive").all()
    assert (frame["units"] == "Percent").all()
    # No 1M row on the day before it was published.
    assert list(zip(_dates(frame), frame["series_id"])) == [
        ("2001-07-30", "DGS3MO"),
        ("2001-07-30", "DGS1"),
        ("2001-07-30", "DGS10"),
        ("2001-07-31", "DGS1MO"),
        ("2001-07-31", "DGS3MO"),
        ("2001-07-31", "DGS1"),
        ("2001-07-31", "DGS10"),
    ]
    assert frame["value"].dtype == "float64"
    assert frame.loc[3, "value"] == pytest.approx(3.67)


def test_any_range_within_a_year_is_served_from_one_file(tmp_path, loader, massive):
    loader.load("DGS10", "2001-07-30", "2001-07-31")
    assert len(massive.calls) == 1
    # The whole year was fetched once ...
    assert massive.calls[0][1]["date.gte"] == "2001-01-01"
    partition = "rates/year=2001/dataset=treasury_yields"
    assert (tmp_path / "parquet" / partition / "daily_massive.parquet").exists()
    assert (tmp_path / "raw" / partition / "daily_massive.json").exists()

    # ... so overlapping and wider ranges, and other tenors, need no new request.
    frame = loader.load(["DGS3MO", "DGS10"], "2001-01-01", "2001-12-31")
    assert len(massive.calls) == 1
    assert _dates(frame).unique().tolist() == ["2001-07-27", "2001-07-30", "2001-07-31"]


def test_a_range_spanning_years_fetches_each_year_once(loader, massive):
    frame = loader.load("DGS1", "2001-07-31", "2002-01-02")
    assert [p["date.gte"] for _, p in massive.calls] == ["2001-01-01", "2002-01-01"]
    assert _dates(frame).tolist() == ["2001-07-31", "2002-01-02"]


def test_current_year_is_refetched_on_a_new_day_only(loader, massive, today):
    records = [{"date": "2026-09-21", "yield_1_year": 4.45}]
    massive.records = records
    loader.load("DGS1", "2026-09-01", "2026-09-30")
    loader.load("DGS1", "2026-09-01", "2026-09-30")
    assert len(massive.calls) == 1  # same day: cached

    records.append({"date": "2026-09-22", "yield_1_year": 4.50})
    today["date"] = dt.date(2026, 9, 23)
    frame = loader.load("DGS1", "2026-09-01", "2026-09-30")
    assert len(massive.calls) == 2  # new day: refetched, picks up the new print
    assert _dates(frame).tolist() == ["2026-09-21", "2026-09-22"]


def test_a_past_year_is_never_refetched(loader, massive, today):
    loader.load("DGS1", "2001-07-30", "2001-07-31")
    today["date"] = dt.date(2027, 1, 5)
    loader.load("DGS1", "2001-07-30", "2001-07-31")
    assert len(massive.calls) == 1


def test_massive_rejects_non_treasury_series(loader):
    with pytest.raises(ValueError, match="SOFR"):
        loader.load("SOFR", "2025-01-01", "2025-01-31")


def test_rejects_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        RatesDataLoader("bloomberg", data_dir=tmp_path)


def test_empty_range_returns_typed_empty_frame(loader):
    frame = loader.load("DGS10", "2001-08-04", "2001-08-05")
    assert frame.empty and list(frame.columns) == RATE_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(frame["observation_date"])


# -- FRED -------------------------------------------------------------------

def test_fred_provider_fetches_metadata_once_per_series(tmp_path):
    session = FredSession([
        {"date": "2024-12-31", "value": "4.49", "realtime_start": "2026-09-22"},
        {"date": "2025-01-02", "value": "4.40"},
        {"date": "2025-01-03", "value": "."},
    ])
    fred = RatesDataLoader(
        "fred", source=FredRatesSource(api_key="k", session=session), data_dir=tmp_path
    )
    frame = fred.load("sofr", "2024-12-31", "2025-01-03")

    urls = [url.rsplit("/fred/", 1)[1] for url, _ in session.calls]
    assert urls == ["series", "series/observations", "series/observations"]
    assert (tmp_path / "parquet/rates/year=2025/dataset=SOFR/daily_fred.parquet").exists()
    assert frame["series_id"].unique().tolist() == ["SOFR"]
    # FRED's "." is kept as a row with a missing value, not dropped.
    assert frame["value"].isna().tolist() == [False, False, True]
    assert frame["value_raw"].tolist() == ["4.49", "4.40", "."]
    assert (frame["units"] == "Percent").all()
    # The vintage is not a column but survives in the provider's record.
    assert json.loads(frame.loc[0, "raw_record"])["realtime_start"] == "2026-09-22"


# -- yields_as_of -----------------------------------------------------------

def test_yields_as_of_uses_that_days_publication(loader):
    snap = loader.yields_as_of("2001-07-31")
    assert snap.observation_date == dt.date(2001, 7, 31)
    assert snap.series_ids == ("DGS1MO", "DGS3MO", "DGS1", "DGS10")
    assert snap.tenors.tolist() == pytest.approx([1 / 12, 0.25, 1.0, 10.0])
    assert snap.yields_pct.tolist() == [3.67, 3.54, 3.53, 5.07]


def test_yields_as_of_falls_back_over_weekends_and_holidays(loader):
    # 2001-07-29 is a Sunday.
    assert loader.yields_as_of("2001-07-29").observation_date == dt.date(2001, 7, 27)


def test_yields_as_of_lag_avoids_same_day_look_ahead(loader):
    snap = loader.yields_as_of("2001-07-31", lag_days=1)
    assert snap.as_of == dt.date(2001, 7, 31)
    assert snap.observation_date == dt.date(2001, 7, 30)
    assert "DGS1MO" not in snap.series_ids  # not yet published then


def test_yields_as_of_crosses_year_boundaries(loader):
    # 2002-01-01 is a holiday; the previous print is in the prior year's file.
    snap = loader.yields_as_of("2002-01-01", lookback_days=200)
    assert snap.observation_date == dt.date(2001, 7, 31)


def test_yields_as_of_raises_when_nothing_is_recent(loader):
    with pytest.raises(LookupError):
        loader.yields_as_of("2001-12-31")


def test_yields_as_of_rejects_non_treasury_series(loader):
    with pytest.raises(ValueError):
        loader.yields_as_of("2001-07-31", series_ids=["SOFR"])
