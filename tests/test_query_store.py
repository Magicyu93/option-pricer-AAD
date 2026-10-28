"""Query cache: readable file names, and a stale query never served."""

import json

import pandas as pd
import pytest

from data_pipeline.common import QueryResult, QueryStore

QUERY = {"schema": "dividends-v1", "ticker": "SPY", "start_date": "2025-01-01"}
KEY = {"year": 2025, "ticker": "SPY"}


def _fetcher(calls, value=1.0):
    def fetch():
        calls.append(1)
        return QueryResult(frame=pd.DataFrame({"x": [value]}), raw_payload={"v": value})

    return fetch


def test_files_are_named_for_their_contents(tmp_path):
    store = QueryStore(tmp_path, domain="dividends", provider="massive", kind="dividends")
    store.get_or_create(QUERY, _fetcher([]), partitions=KEY)

    raw = tmp_path / "raw/dividends/year=2025/ticker=SPY/dividends_massive.json"
    parquet = tmp_path / "parquet/dividends/year=2025/ticker=SPY/dividends_massive.parquet"
    assert raw.exists() and parquet.exists()
    assert json.loads(raw.read_text())["query"] == QUERY


def test_same_query_is_served_from_cache(tmp_path):
    store = QueryStore(tmp_path, domain="dividends", provider="massive", kind="dividends")
    calls = []
    store.get_or_create(QUERY, _fetcher(calls), partitions=KEY)
    frame = store.get_or_create(QUERY, _fetcher(calls, 2.0), partitions=KEY)
    assert len(calls) == 1 and frame["x"].tolist() == [1.0]


def test_changed_query_at_same_path_refetches(tmp_path):
    store = QueryStore(tmp_path, domain="dividends", provider="massive", kind="dividends")
    calls = []
    store.get_or_create(QUERY, _fetcher(calls), partitions=KEY)
    bumped = {**QUERY, "schema": "dividends-v2"}
    frame = store.get_or_create(bumped, _fetcher(calls, 2.0), partitions=KEY)
    assert len(calls) == 2 and frame["x"].tolist() == [2.0]
    # ... and the file now belongs to the new query.
    assert store.read(bumped, KEY)["x"].tolist() == [2.0]
    assert store.read(QUERY, KEY) is None


@pytest.mark.parametrize("bad", ["", "..", "."])
def test_rejects_path_escapes(tmp_path, bad):
    store = QueryStore(tmp_path, domain="rates", provider="fred", kind="daily")
    with pytest.raises(ValueError):
        store.paths({"dataset": bad})


def test_unsafe_characters_are_replaced(tmp_path):
    store = QueryStore(tmp_path, domain="rates", provider="fred", kind="daily")
    raw, _ = store.paths({"year": 2025, "dataset": "a/b"})
    assert raw == tmp_path / "raw/rates/year=2025/dataset=a_b/daily_fred.json"


def test_providers_for_the_same_partition_do_not_collide(tmp_path):
    massive = QueryStore(tmp_path, domain="rates", provider="massive", kind="daily")
    fred = QueryStore(tmp_path, domain="rates", provider="fred", kind="daily")
    partition = {"year": 2025, "dataset": "DGS10"}
    massive_raw, massive_parquet = massive.paths(partition)
    fred_raw, fred_parquet = fred.paths(partition)
    assert massive_parquet.parent == fred_parquet.parent
    assert (massive_parquet.name, fred_parquet.name) == ("daily_massive.parquet", "daily_fred.parquet")
