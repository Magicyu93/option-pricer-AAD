"""Download everything the pricer needs for a set of underlyings and dates.

Edit the settings below, then run ``python scripts/download_market_data.py``.

* **equity** -- option minute bars joined to the underlying's minute bars, one
  trading day at a time (stocks/ETFs or indices, routed automatically);
* **rates** -- Treasury yields from ``RATES_LOOKBACK_DAYS`` before ``START``, so
  the first day always has a publication to fall back on;
* **dividends** -- ex-dates from ``START`` to ``DIVIDEND_HORIZON_DAYS`` after
  ``END``, since an option needs the dividends paid before it expires.  Index
  underlyings are skipped.

Everything is cached under ``DATA_DIR``, so rerunning after an interruption only
fetches what is missing.  Credentials come from the environment:
``MASSIVE_S3_ACCESS_KEY`` / ``MASSIVE_S3_SECRET_KEY`` for Flat Files,
``MASSIVE_API_KEY`` for the REST API, and ``FRED_API_KEY`` for FRED.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the repo root importable when run as a file, without installing it.
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.common import DEFAULT_DATA_DIR
from data_pipeline.download import download_dividends, download_equity, download_rates

# -- settings ------------------------------------------------------------------
TICKERS = ["SPY", "SPX"]
START = "2026-09-08"
END = "2026-09-11"
DATA_DIR = DEFAULT_DATA_DIR  # <repo>/market_data, whichever folder you run from

DOWNLOAD_EQUITY = True
DOWNLOAD_RATES = True
DOWNLOAD_DIVIDENDS = True

RATES_PROVIDER = "massive"  # or "fred"
RATES_LOOKBACK_DAYS = 14
DIVIDEND_HORIZON_DAYS = 365
REFRESH = False  # refetch rates and dividends even if cached
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    if DOWNLOAD_EQUITY:
        results = download_equity(TICKERS, START, END, data_dir=DATA_DIR, on_progress=print)
        failed = [r.label for r in results if not r.ok]
        if failed:
            print(f"equity failed for {failed}; rerun to retry them")
    if DOWNLOAD_RATES:
        download_rates(START, END, provider=RATES_PROVIDER, lookback_days=RATES_LOOKBACK_DAYS,
                       refresh=REFRESH, data_dir=DATA_DIR, on_progress=print)
    if DOWNLOAD_DIVIDENDS:
        download_dividends(TICKERS, START, END, horizon_days=DIVIDEND_HORIZON_DAYS,
                           refresh=REFRESH, data_dir=DATA_DIR, on_progress=print)
