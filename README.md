# torch_pricer

Differentiable Monte Carlo option pricing on PyTorch, with a market-data pipeline and a
Streamlit app for studying the implied-vol surface on real option prices.

The repo has three parts that build on each other:

| Part | What it does |
|---|---|
| `data_pipeline/` | Downloads and caches minute bars for options and their underlyings, Treasury yields and dividends (Massive, FRED) |
| `torch_pricer/` | The library: instruments, curves, vol surfaces, models, the Monte Carlo engine, greeks by autograd |
| `app/` | A Streamlit app that visualises one day's vol surface, put-call parity, rates and carry, and compares the fitted surface models |

## Status

| Area | State |
|---|---|
| Monte Carlo engine with Black-Scholes (GBM) | Works. Price, delta, vega, theta and rho from one backward pass; gamma by differencing the pathwise delta |
| Treasury par yields → zero curve | Works (bootstrap of the CMT curve) |
| Forwards from put-call parity: each pair at the Treasury discount, one forward per expiry, a non-decreasing implied-dividend curve | Works. Every filter is a `ParityConfig` field and a sidebar control in the app |
| Put-call parity regression for r and q | Works, as a diagnostic only: too noisy to use as an input |
| Implied vols per minute bar; SSVI, eSSVI and raw SVI surface fits | Works. eSSVI fits best of the arbitrage-free models; compare them on the app's Surface models page |
| Data pipeline (SPX, SPY; options, stocks, indices, rates, dividends) | Works |
| Vol surface research app | Works, for cached days of SPX and SPY |
| Heston, local vol and LSV models, Brownian bridge, `BlackScholesModel.calibrate` | Not implemented yet (stubs). Dupire's local variance of any surface already works |

Open questions from the research so far, in `scripts/vol_surface_visual.ipynb`:
- Implied vols computed from out-of-the-money and in-the-money options disagree.
- Put-call parity is not exact on the trades.
- Some prices are out of the no-arbitrage bounds, so their implied vol is NaN.

The suspects were the inputs: minute-aggregated trades rather than quotes, and a Treasury
curve with a flat 1.1% dividend yield for SPX. The flat q put the forward 0.5–28 index points
below what parity implies (more at longer expiries), matching the sign of the call − put iv
gaps. The forwards now come from parity at the Treasury discount, which centres the
near-the-money call − put gap on zero in a held-out test over 19 days.

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
`scripts/vol_surface_visual.ipynb`. Two pages put the three things that should agree side
by side: the rates, the forwards, and the call vs put implied vols. A third fits the
library's surface models and compares them with each other and with the quotes.

**Sidebar** (applies to every page):
- the underlying and the day, a calendar limited to the cached days;
- the option roots, e.g. SPXW;
- the minute window, from 15:30 to before 16:00 ET by default. The end is exclusive: the
  16:00 bar holds trades after the index's closing print;
- the Treasury provider and lag;
- **Forwards: put-call parity**, every filter on the pairs behind the forwards
  (`ParityConfig`): the close window and the earlier-in-the-day fill, the moneyness band
  `max(band × √T, floor)`, the estimator, and under *More pair filters* leg volume and price,
  strikes per minute, close minutes, box prints and the standard-error floor and drift;
- the moneyness band for the parity regression, a diagnostic only;
- plot filters: days to expiry, minimum price and max |k|/√T.

**Day overview**, all expiries at once:
- **Forwards from put-call parity:** the active pair filters in words; implied dividends
  `S − D·F` per expiry (close tier filled, day tier hollow) with the monotone curve through
  them and the expiries read off it; each expiry's own forward minus the curve's; and how
  many pairs each filter drops.
- **Rates and carry:** the Treasury zero, forward and par curves, with the parity
  regression's r per expiry overlaid. Next to it, the dividend curve's q against the
  regression's q.
- **Does parity hold?:** the curve's forward minus each iv pair's own forward (both at the
  Treasury discount factor) by expiry, call − put iv gap against moneyness, and pairs per
  expiry.
- **The surface:** smiles, an implied-vol 3D plot with the call-put pairs overlaid, and price
  and iv 3D plots side by side.
- **Per-expiry table:** every number about each expiry. Select a row to open it in the
  drill-down.

**Expiry drill-down**, one expiry:
- **Metrics:** r and q from the regression against what the implied vols use; the expiry's
  own parity forward (and its tier) against the curve's; the regression's forward gap.
- **The forward through the day:** every pair's forward minus the curve's, minute by minute
  over the whole parity window, dropped pairs faint, with why each was dropped.
- **Parity on the trades:** C − P against strike with two lines, the regression's fit and the
  forward curve's, plus the residuals from each.
- **The vols:** the smile with calls and puts marked differently and pairs joined, and the
  call − put iv gap coloured by minute.

**Surface models**, the library's fits side by side:
- **What to fit**, in a form at the top: the models (SSVI, eSSVI, raw SVI) and the quotes'
  floors (days to expiry, price, max |k|/√T). The forwards are the Treasury curve and the
  parity dividend curve, as on the other pages. Fits are cached; SVI takes about 15 s.
- **Fit quality:** per model, the error, convergence, fit time and the three arbitrage
  checks (butterfly, calendar, local variance). Next to it, the error and bias per expiry;
  below, every quote's error against k/√T, one panel per model.
- **The surface against the quotes:** a model's fitted surface as a 3D mesh, the quotes as
  points coloured by their error, and a segment from each quote to the surface.
- **Smiles:** a grid of expiries with the quotes and every model's smile.
- **Term structure and local vol:** each model's ATM vol, between the expiries too, against
  the quotes' ATM vol. Next to it, Dupire's local vol from a chosen model, with cells
  where the local variance is negative marked.
- **Details:** the fitted parameters per expiry, the constraints, the forwards, and the
  quotes dropped.

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
    surface_models.py
  analysis/            research computations taken from the notebook
    implied_vols.py    curve forwards, implied vol per bar, filters, call-put iv pairs
    rate_curves.py     zero / forward / par curve tables
    compare.py         the per-expiry table, and both parity lines for one expiry
    surface_models.py  the library's surface fits, and the frames and grids the page draws
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
| `market/parity.py` | Put-call parity. The forwards: `ParityConfig` (every pair filter, by name), `forward_pairs` (each same-minute pair's forward at the Treasury D, labelled with the filter that drops it) and `parity_forwards` (one forward per expiry, close or day tier, with a standard error). The minute-demeaned regression for r and q (`regress_parity`, `parity_fit`) is a diagnostic only |
| `market/curve/` | The `Curve` interface and `RateCurve`; `treasury_zero_curve` (par → zero bootstrap); `implied_dividend_curve`: the dividend curve through the parity forwards, implied dividends made non-decreasing (weighted isotonic regression); `curve_forwards`: each expiry's forward `S·D_q(T)/D_r(T)` from the two curves |
| `market/surface/` | The `VolSurface` interface in total variance `w(k, T)`, with butterfly density and Dupire local variance by autograd; flat, SSVI, eSSVI and raw SVI surfaces; per-bar implied vols, `fit_surface` and the arbitrage checks |
| `market/build.py` | `build_snapshot(frame, rate_quotes, product, settings)`: one day of data → calibrated `MarketSnapshot`, with a record of every step. The dividend curve is implied by put-call parity (`settings.parity_config`); forwards come from it and the Treasury curve |
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
