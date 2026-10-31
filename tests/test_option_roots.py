"""Option roots written on another root's underlying: SPXW weeklies are SPX options."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.equity import MassiveHistoricalDataLoader
from data_pipeline.equity.massive.schema import OPTION_COLUMNS, process_options

DATE = "2026-09-08"


def _bars(*tickers):
    ns = pd.Timestamp(f"{DATE} 15:50", tz="America/New_York").value
    return pd.DataFrame(
        {"ticker": list(tickers), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
         "volume": 1, "transactions": 1, "window_start": ns}
    )


def test_weekly_roots_map_to_their_index_and_keep_the_root():
    frame = process_options(
        _bars("O:SPX260918C05000000", "O:SPXW260911P05000000", "O:XSP260918C00500000"),
        DATE,
        timezone="America/New_York",
    )
    by_root = dict(zip(frame["option_root"], frame["underlying"]))
    assert by_root == {"SPX": "SPX", "SPXW": "SPX", "XSP": "XSP"}
    assert list(frame.columns) == list(OPTION_COLUMNS)


def test_requesting_spx_scans_in_spxw_but_not_lookalike_etfs(tmp_path):
    loader = MassiveHistoricalDataLoader(data_dir=tmp_path, api_key="test", verbose=False)
    raw = loader._raw_path("options", DATE)
    raw.parent.mkdir(parents=True)
    _bars(
        "O:SPX260918C05000000", "O:SPXW260911P05000000", "O:SPXL260918C00100000"
    ).to_csv(raw, index=False, compression="gzip")

    frame = loader.load_options(DATE, "SPX")
    assert sorted(frame["option_root"]) == ["SPX", "SPXW"]
    assert set(frame["underlying"]) == {"SPX"}


def test_cache_written_before_option_root_existed_is_rebuilt(tmp_path):
    loader = MassiveHistoricalDataLoader(data_dir=tmp_path, api_key="test", verbose=False)
    stale = loader._cache_path("options", DATE, "SPX")
    stale.parent.mkdir(parents=True)
    old = pd.DataFrame({"underlying": ["SPX"], "option_ticker": ["O:SPX260918C05000000"]})
    pq.write_table(pa.Table.from_pandas(old), stale)
    assert not loader._cache_is_current("options", stale)

    raw = loader._raw_path("options", DATE)
    raw.parent.mkdir(parents=True)
    _bars("O:SPX260918C05000000", "O:SPXW260911P05000000").to_csv(
        raw, index=False, compression="gzip"
    )
    assert len(loader.load_options(DATE, "SPX")) == 2
    assert loader._cache_is_current("options", stale)


def test_listed_products_name_the_same_roots_as_the_pipeline_aliases():
    from data_pipeline.equity.massive.schema import OPTION_ROOT_ALIASES
    from torch_pricer.instruments.listed import LISTED_PRODUCTS

    for underlying, product in LISTED_PRODUCTS.items():
        aliased = {root for root, target in OPTION_ROOT_ALIASES.items() if target == underlying}
        assert product.root_names == {underlying} | aliased, underlying


def test_a_root_the_product_does_not_list_is_refused_not_guessed():
    import pytest

    from torch_pricer.errors import ValidationError
    from torch_pricer.instruments.listed import listed_product
    from torch_pricer.market.market_data import window_bars

    frame = process_options(_bars("O:SPXQ260918C05000000"), DATE, timezone="America/New_York")
    frame = frame.assign(option_timestamp=frame["timestamp"], matched_underlying=True, spot=6500.0)
    with pytest.raises(ValidationError, match="SPXQ"):
        window_bars(frame, pd.Timestamp(DATE).date(), listed_product("SPX"))


# -- the legacy standalone loader applies the same mapping -------------------------

def test_legacy_loader_maps_weekly_roots_to_their_index(tmp_path):
    from data_pipeline.option_equity_historical_data_loader import (
        MassiveHistoricalDataLoader as LegacyLoader,
    )

    raw = tmp_path / "raw" / "options" / f"{DATE}.csv.gz"  # its own (older) raw layout
    raw.parent.mkdir(parents=True)
    _bars("O:SPX260918C05000000", "O:SPXW260911P05000000", "O:SPXL260918C00100000").to_csv(
        raw, index=False, compression="gzip"
    )
    loader = LegacyLoader(data_dir=tmp_path, api_key="test", verbose=False)
    frame = loader.load_options(DATE, "SPX")
    assert sorted(frame["option_root"]) == ["SPX", "SPXW"]
    assert set(frame["underlying"]) == {"SPX"}


def test_legacy_loader_rebuilds_a_cache_written_before_the_mapping(tmp_path):
    from data_pipeline.option_equity_historical_data_loader import (
        MassiveHistoricalDataLoader as LegacyLoader,
    )

    loader = LegacyLoader(data_dir=tmp_path, api_key="test", verbose=False)
    stale = loader._cache_path("options", DATE, "SPX")
    stale.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"underlying": ["SPX"]})), stale)
    assert not loader._cache_is_current("options", stale)

    raw = tmp_path / "raw" / "options" / f"{DATE}.csv.gz"
    raw.parent.mkdir(parents=True)
    _bars("O:SPX260918C05000000", "O:SPXW260911P05000000").to_csv(raw, index=False, compression="gzip")
    assert len(loader.load_options(DATE, "SPX")) == 2
    assert loader._cache_is_current("options", stale)
