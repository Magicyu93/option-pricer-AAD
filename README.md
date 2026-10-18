# torch_pricer

Differentiable Monte Carlo option pricing on PyTorch, with a market-data pipeline and a
Streamlit app for studying the implied-vol surface on real option prices.

The repo has three parts that build on each other:

| Part | What it does |
|---|---|
| `data_pipeline/` | Downloads and caches minute bars for options and their underlyings, Treasury yields and dividends (Massive, FRED) |
| `torch_pricer/` | The library: instruments, curves, vol surfaces, models, the Monte Carlo engine, greeks by autograd |
| `app/` | A Streamlit app that visualises one day's vol surface, put-call parity, rates and carry |

## Status

| Area | State |
|---|---|
| Monte Carlo engine with Black-Scholes (GBM) | Works. Price, delta, vega, theta and rho from one backward pass; gamma by differencing the pathwise delta |
| Treasury par yields → zero curve | Works (bootstrap of the CMT curve) |
| Put-call parity forwards and implied carry | Works |
| Implied vols per minute bar, SSVI surface fit | Works. The fitted surface still needs checking against the data |
| Data pipeline (SPX, SPY; options, stocks, indices, rates, dividends) | Works |
| Vol surface research app | Works, for cached days of SPX and SPY |
| Raw SVI, Heston, local vol, Brownian bridge, `BlackScholesModel.calibrate` | Not implemented yet (stubs) |

Open questions from the research so far, in `scripts/vol_surface_visual.ipynb`:
- Implied vols computed from out-of-the-money and in-the-money options disagree.
- Put-call parity is not exact on the trades.
- Some prices are out of the no-arbitrage bounds, so their implied vol is NaN.

The suspects are the inputs: minute-aggregated trades rather than quotes, and a Treasury
curve with a flat 1.1% dividend yield for SPX. The app is built to test this. On SPX
2026-09-10 the forward implied by parity sits about 0.5–4 index points above the forward the
implied vols use, which matches the sign of the call − put iv gaps.

## Setup

Python 3.10 or newer.

```bash
git clone <this repo> && cd torch_pricer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"
```

`dev` adds pytest and `app` adds Streamlit and Plotly. QuantLib is used only for
calendars and day counts.

The editable install is what makes `data_pipeline` and `torch_pricer` importable from
`scripts/`. Without it, run commands from the repo root with `PYTHONPATH=.` in front.

### Credentials

The download scripts read API keys from the environment:

| Provider | Used for | Variables |
|---|---|---|
| Massive flat files | Option, stock and index minute bars | `MASSIVE_S3_ACCESS_KEY` + `MASSIVE_S3_SECRET_KEY`, or `MASSIVE_API_ID` + `MASSIVE_API_KEY` |
| Massive REST | Treasury yields, dividends, reference data | `MASSIVE_API_KEY` |
| FRED | Treasury yields (alternative provider) | `FRED_API_KEY` |

The pricer, the examples and the tests on synthetic data need no keys. The app needs
downloaded data.

## Quick start

### 1. Download market data

Edit the settings at the top of `scripts/download_market_data.py` (`TICKERS`, `START`,
`END`, provider), then:

```bash
python scripts/download_market_data.py
```

It fetches:
- option minute bars joined to the underlying's bars, one trading day at a time;
- Treasury yields, starting 14 days before `START` so the first day has a print;
- dividends up to a year past `END`. Indices are skipped.

Everything is cached under `market_data/` (git-ignored; one day of SPX options is large).
Reruns fetch only what is missing. To fetch one dataset at a time, set the others'
`DOWNLOAD_EQUITY` / `DOWNLOAD_RATES` / `DOWNLOAD_DIVIDENDS` flags to `False`.

### 2. Run the app

```bash
streamlit run app/streamlit_app.py
```

Then open the URL it prints (usually http://localhost:8501). See [The app](#the-app) below.

### 3. Price an option

```bash
python -m examples.european_options          # Monte Carlo vs closed form, price and greeks
python -m examples.european_bs_convergence   # sampling error vs number of paths
```

### 4. Run the tests

```bash
python -m pytest -q
```

Tests that need cached market data skip themselves when it is missing.

## The app

The app shows one day of cached option trades, using the analysis from
`scripts/vol_surface_visual.ipynb`. It has two pages, and each puts the three things that
should agree side by side: the rates, the forwards, and the call vs put implied vols.

**Sidebar** (applies to both pages):
- the underlying and the day, a calendar limited to the cached days;
- the option roots, e.g. SPXW;
- the minute window, from 15:30 to before 16:00 ET by default. The end is exclusive: the
  16:00 bar holds trades after the index's closing print;
- the Treasury provider and lag, and the flat dividend yield;
- the moneyness band for the parity regression;
- plot filters: days to expiry, minimum price and max |k|/√T.

**Day overview**, all expiries at once:
- **Rates and carry:** the Treasury zero, forward and par curves, with the parity
  regression's r per expiry overlaid. Next to it, the flat q against the regression's q.
- **Does parity hold?:** forward error by expiry, call − put iv gap against moneyness, and
  pairs per expiry.
- **The surface:** smiles, an implied-vol 3D plot with the call-put pairs overlaid, and price
  and iv 3D plots side by side.
- **Per-expiry table:** every number about each expiry. Select a row to open it in the
  drill-down.

**Expiry drill-down**, one expiry:
- **Metrics:** r, q and the forward from the regression against what the implied vols use,
  and the gap between the two forwards.
- **Parity on the trades:** C − P against strike with two lines, the regression's fit and the
  Treasury + flat q line, plus the residuals from each.
- **The vols:** the smile with calls and puts marked differently and pairs joined, and the
  call − put iv gap coloured by minute.

The app only reads cached data. To see a new day, download it first (step 1).

### How the app code is organised

```
app/
  streamlit_app.py     entrypoint: navigation and the shared sidebar
  cache.py             sidebar settings, and every computation wrapped in st.cache_data
  plots.py             Plotly figures; they only draw the frames they are given
  pages/
    day_overview.py
    expiry_drilldown.py
  analysis/            research computations taken from the notebook
    implied_vols.py    curve forwards, implied vol per bar, filters, call-put iv pairs
    rate_curves.py     zero / forward / par curve tables
    compare.py         the per-expiry table, and both parity lines for one expiry
```

The window, the time to settlement and the parity regression come from the library
(`torch_pricer.market.market_data.window_bars` and `torch_pricer.market.parity`), so the app
looks at the same bars a snapshot is calibrated from.

**Where code goes:**
- Exploratory analysis used only by the app belongs in `app/analysis/`, not in the library.
- `torch_pricer` and `data_pipeline` change only for reusable infrastructure or bug fixes.
- Pages and plots don't compute anything.

## Library tour

### `data_pipeline/`

| Module | Role |
|---|---|
| `equity/` | `MassiveHistoricalDataLoader.load_market(dates, tickers)`: option minute bars joined to the underlying's bars. Stocks/ETFs and indices are routed automatically |
| `rates/` | `RatesDataLoader(provider).yields_as_of(day, lag_days)`: Treasury constant-maturity par yields (Massive or FRED) |
| `dividends/` | `DividendsDataLoader`: declared cash dividends |
| `download.py` | `download_equity`, `download_rates`, `download_dividends`, `credentials_status`: the functions behind the scripts |
| `coverage.py` | `cached_days`, `coverage`: what is in the cache, without reading file contents |
| `common/` | HTTP, Massive pagination, and partitioned storage (`<root>/<layer>/<dataset>/<key>=<value>/...`) |

### `torch_pricer/`

| Module | Role |
|---|---|
| `instruments/spec.py`, `payoff.py` | Instrument specs as frozen dataclasses, and differentiable payoffs over simulated paths |
| `instruments/listed.py` | `LISTED_PRODUCTS`: contract terms per underlying, i.e. option roots, AM/PM settlement and exercise style (SPX, SPY) |
| `conventions.py` | Calendars and day counts. The only module that touches QuantLib |
| `market/market_data.py` | What the market publishes: spot, option and rate quotes, kept apart from fitted objects. `window_bars`: the observation window, with each bar's time to settlement (`T`, from its own minute; AM roots settle 6.5h early) |
| `market/parity.py` | Put-call parity: same-minute pairs near the money, the minute-demeaned regression for r and q (`regress_parity`, `parity_fit`), and the forward per expiry (`parity_forwards`) |
| `market/curve/` | The `Curve` interface; `treasury_zero_curve` (par → zero bootstrap); implied carry from the parity forwards |
| `market/surface/` | The `VolSurface` interface in total variance `w(k, T)`; flat and SSVI surfaces; per-bar implied vols and `fit_surface` |
| `market/build.py` | `build_snapshot`: one day of data → calibrated `MarketSnapshot`, with a record of every step |
| `market/snapshot.py` | `MarketSnapshot`: the calibrated curves and surface, the only input to pricing |
| `black_formula.py` | Black-76 price, greeks and implied vol |
| `simulator/` | SDE path generation (GBM in log space, arithmetic BM) and seeded normal draws |
| `models/` | `Model` interface and `BlackScholesModel`. Heston and local vol are placeholders |
| `pricer/engine.py` | `price(...)` with `MCConfig`: Monte Carlo price and greeks by autograd (pathwise) |

Design choices that run through the library:
- **Time is a year fraction from the snapshot's `as_of`.** There is no global evaluation
  date, so QuantLib's process-wide date singleton is never used.
- **Curves and surfaces are torch tensors.** That way rho and vega come out of the same
  backward pass as delta.
- **Quotes and fitted objects are kept separate.** You can always tell whether a number came
  from the market or from a fit.

## Other files

| Path | What |
|---|---|
| `scripts/vol_surface_visual.ipynb` | The research notebook the app is built from |
| `examples/` | Monte Carlo vs closed form for European options, and convergence rates |
| `tests/` | Unit and regression tests. `test_app*.py` cover the app, and run only when data is cached |

## Known issues

- The app's analysis assumes European options and a flat dividend yield. For SPY (American,
  discrete dividends), the parity r and q and the call vs put gaps also reflect early
  exercise and dividends.
- The app cannot download data itself, and only SPX and SPY are listed products. Adding a
  ticker means adding it to `LISTED_PRODUCTS`.
