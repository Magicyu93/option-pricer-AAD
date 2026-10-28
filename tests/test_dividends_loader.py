"""The dividends loader, offline: a fake source stands in for Massive."""

import datetime as dt

import pandas as pd
import pytest

import data_pipeline.dividends.loader as loader_module
from data_pipeline.dividends import DividendsDataLoader

EVENTS = [
    {"id": "a", "ticker": "SPY", "ex_dividend_date": "2024-12-20", "cash_amount": 1.97},
    {"id": "b", "ticker": "SPY", "ex_dividend_date": "2025-03-21", "cash_amount": 1.70},
    {"id": "c", "ticker": "SPY", "ex_dividend_date": "2025-06-20", "cash_amount": 1.76},
]


class FakeSource:
    """Serves EVENTS filtered by ticker and ex-dividend date, like the endpoint."""

    def __init__(self):
        self.calls = []

    def fetch(self, ticker, start_date, end_date):
        self.calls.append((ticker, start_date, end_date))
        records = [
            e for e in EVENTS
            if e["ticker"] == ticker and start_date <= e["ex_dividend_date"] <= end_date
        ]
        return {"records": records, "pages": [{"results": records}]}


@pytest.fixture
def source():
    return FakeSource()


@pytest.fixture
def loader(tmp_path, source):
    return DividendsDataLoader(source=source, data_dir=tmp_path)


@pytest.fixture
def today(monkeypatch):
    current = {"date": dt.date(2026, 9, 29)}
    monkeypatch.setattr(loader_module, "_today", lambda: current["date"])
    return current


def test_one_file_per_year_and_ticker(tmp_path, loader, source, today):
    loader.load("SPY", "2025-03-01", "2025-03-31")
    assert source.calls == [("SPY", "2025-01-01", "2025-12-31")]
    partition = "dividends/year=2025/ticker=SPY"
    assert (tmp_path / "parquet" / partition / "dividends_massive.parquet").exists()
    assert (tmp_path / "raw" / partition / "dividends_massive.json").exists()


def test_any_range_within_cached_years_needs_no_request(loader, source, today):
    frame = loader.load("SPY", "2025-03-01", "2025-03-31")
    assert frame["dividend_id"].tolist() == ["b"]

    frame = loader.load("SPY", "2025-01-01", "2025-12-31")
    assert len(source.calls) == 1
    assert frame["dividend_id"].tolist() == ["b", "c"]


def test_a_range_spanning_years_fetches_each_year_once(loader, source, today):
    frame = loader.load("SPY", "2024-12-01", "2025-04-01")
    assert [c[1] for c in source.calls] == ["2024-01-01", "2025-01-01"]
    assert frame["dividend_id"].tolist() == ["a", "b"]


def test_current_year_is_refetched_on_a_new_day_only(loader, source, today):
    today["date"] = dt.date(2025, 7, 1)
    loader.load("SPY", "2025-01-01", "2025-06-30")
    loader.load("SPY", "2025-01-01", "2025-06-30")
    assert len(source.calls) == 1

    today["date"] = dt.date(2025, 7, 2)
    loader.load("SPY", "2025-01-01", "2025-06-30")
    assert len(source.calls) == 2


def test_a_whole_domain_reads_as_one_dataset(tmp_path, loader, today):
    loader.load("SPY", "2024-01-01", "2025-12-31")
    frame = pd.read_parquet(tmp_path / "parquet/dividends")
    assert sorted(frame["dividend_id"]) == ["a", "b", "c"]
    assert set(frame["ticker"].astype(str)) == {"SPY"}
