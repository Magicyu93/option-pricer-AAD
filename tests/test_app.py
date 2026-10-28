"""The research app's pages run end to end (headless), for each cached ticker and day picked.

Skipped when Streamlit is not installed or no options are cached, since the app only
shows what the library computes from cached data. Widget values are seeded through
``session_state`` because AppTest does not apply widget changes under ``st.navigation``.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from data_pipeline.coverage import cached_days  # noqa: E402
from torch_pricer.instruments.listed import LISTED_PRODUCTS  # noqa: E402

TICKERS = [t for t in LISTED_PRODUCTS if cached_days("options", t)]
pytestmark = pytest.mark.skipif(not TICKERS, reason="no options cached")
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGES = ["day_overview", "expiry_drilldown", "surface_models"]


def _errors(at):
    return [e.value for e in at.exception] + [e.value for e in at.error]


def _app(ticker: str, day: date | None = None):
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=300)
    at.session_state["ticker"] = ticker
    if day is not None:
        at.session_state[f"day_{ticker}"] = day
    return at.run()


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("ticker", TICKERS)
def test_page_renders(ticker, page):
    at = _app(ticker)
    assert not _errors(at)
    at.switch_page(f"pages/{page}.py").run()
    assert not _errors(at)
    assert at.title[0].value
    assert at.caption[0].value.startswith(f"{ticker} ")


@pytest.mark.parametrize("ticker", TICKERS)
def test_picked_day_is_the_one_shown(ticker):
    day = cached_days("options", ticker)[0]
    at = _app(ticker, date.fromisoformat(day))
    assert not _errors(at)
    assert f" on {day} " in at.caption[0].value


def test_uncached_day_says_which_days_are_cached():
    ticker = TICKERS[0]
    days = [date.fromisoformat(d) for d in cached_days("options", ticker)]
    gaps = [d + timedelta(1) for d in days[:-1] if d + timedelta(1) not in days]
    if not gaps:
        pytest.skip("every calendar day in the range is cached")
    at = _app(ticker, gaps[0])
    assert any("No " in e and "Nearest cached" in e for e in _errors(at))


def test_drilldown_opens_the_expiry_chosen_on_the_overview():
    at = _app(TICKERS[0])
    assert not _errors(at)
    at.switch_page("pages/expiry_drilldown.py").run()
    assert not _errors(at)
    assert at.session_state["drill_expiry"] == at.selectbox[0].value
