"""What the market actually publishes.

These are the inputs to the curve and surface calibrations
(:mod:`torch_pricer.market.curve.calibration`,
:mod:`torch_pricer.market.surface.calibration`), kept strictly separate from
their outputs. A quote is an observation with a bid, an ask and a timestamp; a
curve or a vol model is a fitted object with parameters. Collapsing the two --
which is what a ``MarketSnapshot`` holding raw quotes would do -- makes it
impossible to say whether a number came from the market or from a fit.

Deliberately dumb: no interpolation, no arbitrage checks, no torch, no pricing
model. Validation that needs a model belongs in the calibrator that consumes
these.

The same holds for :func:`window_bars`, which turns the data pipeline's minute
trade bars into the observations a snapshot at the close is built from. There
is no bid or ask in that data -- only trades -- so the window does what a
spread would otherwise do: a narrow slot before the close, a minimum premium,
one maturity per expiry date. Dates go through :mod:`torch_pricer.conventions`
like everywhere else.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch_pricer.conventions import DEFAULT_DAY_COUNT, year_fraction
from torch_pricer.errors import ValidationError
from torch_pricer.instruments.listed import OptionProduct, Settlement
from torch_pricer.instruments.spec import Right


def _mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    """Mid where there is a two-sided market, else the last trade, else the one
    side that was quoted.

    A one-sided market is still information -- a bid with no offer is a floor on
    the premium -- and dropping it silently is how a quote that exists gets
    reported as missing.
    """
    if bid is not None and ask is not None:
        return 0.5 * (bid + ask)
    if last is not None:
        return last
    return bid if bid is not None else ask


@dataclass(frozen=True, slots=True)
class SpotQuote:
    """The underlying's level."""

    ticker: str
    value: float
    as_of: dt.date | None = None


@dataclass(frozen=True, slots=True)
class RateQuote:
    """A point on the funding or dividend curve, quoted by tenor.

    ``rate`` is a decimal (``0.0425``), whatever the source's convention.
    ``kind="par"`` is a bond-equivalent par yield, such as a Treasury
    constant-maturity yield; ``as_of`` is the date it was published, which can
    precede the quote set's own date (weekends, holidays, publication lag).
    """

    tenor: str
    rate: float
    kind: str = "zero"  # zero | deposit | swap | dividend | par
    as_of: dt.date | None = None


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One listed option's market.

    Either a premium (``bid``/``ask``/``last``) or an ``implied_vol`` may be
    absent; a calibrator should take whichever it is given and say so if both
    are missing.
    """

    expiry: dt.date
    strike: float
    right: Right = Right.CALL
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    implied_vol: float | None = None
    #: Contracts traded over the observation window. With trade prices and no
    #: two-sided market, this is the only measure of how much to trust ``last``.
    volume: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "right", Right(self.right))
        if self.strike <= 0:
            raise ValidationError(f"strike must be positive, got {self.strike}")
        if self.price is None and self.implied_vol is None:
            raise ValidationError(
                f"quote {self.expiry} {self.strike:g} carries neither a premium nor a vol"
            )

    @property
    def price(self) -> float | None:
        """Mid where there is a two-sided market, else the last trade."""
        return _mid(self.bid, self.ask, self.last)


@dataclass(frozen=True, slots=True)
class QuoteSet:
    """Everything observed for one underlying at one instant."""

    as_of: dt.date
    spot: SpotQuote
    options: tuple[OptionQuote, ...] = ()
    rates: tuple[RateQuote, ...] = ()
    dividends: tuple[RateQuote, ...] = ()

    def expiries(self) -> tuple[dt.date, ...]:
        return tuple(sorted({q.expiry for q in self.options}))

    def slice(self, expiry: dt.date) -> tuple[OptionQuote, ...]:
        """The quotes for one expiry, in strike order."""
        return tuple(sorted((q for q in self.options if q.expiry == expiry), key=lambda q: q.strike))


# -- minute trade bars ---------------------------------------------------------

TIMEZONE = "America/New_York"

@dataclass(frozen=True)
class QuoteConfig:
    """What counts as a usable observation: the same rules for every underlying.

    What differs between underlyings -- which roots settle at the open, the
    exercise style -- is a contract term, on the
    :class:`~torch_pricer.instruments.listed.OptionProduct`.
    """

    window_start: str = "15:45"  # ET, inclusive
    window_end: str = "16:00"  # ET, exclusive: the index stops printing at 16:00
    min_days: int = 7  # a date-only snapshot cannot resolve shorter expiries
    min_price: float = 0.05  # one tick: below it a trade price is noise
    #: Keep ``|ln(K/F)| <= max_abs_k * sqrt(T)``: a band that widens with maturity.
    max_abs_k: float = 1.0


def window_bars(
    frame: pd.DataFrame,
    as_of: dt.date,
    product: OptionProduct,
    config: QuoteConfig = QuoteConfig(),
    day_count: str = DEFAULT_DAY_COUNT,
) -> tuple[pd.DataFrame, Counter]:
    """The bars a snapshot at ``as_of``'s close is built from.

    ``frame`` is the data pipeline's ``load_market`` output for ``product``'s
    underlying: one row per option per minute with a trade, joined to the
    underlying's bar for that minute. ``day_count`` is the snapshot's, so ``T``
    here is the snapshot's ``time_to``, less the hours an AM-settled option
    gives up. Returns the bars -- ``timestamp, expiry, option_root, settlement,
    right, strike, price, volume, spot, days, T`` -- and the count of bars
    dropped, by reason. ``right`` is +1 for a call and -1 for a put.

    Where an AM-settled root and a PM-settled weekly share an expiry date (the
    third Friday), only the PM series is kept: they are different contracts
    with different exercise times, and one expiry date must mean one maturity.
    A root the product does not list raises: the data and the contract terms
    disagree, and guessing its settlement would be silently wrong.
    """
    drops: Counter = Counter()

    def keep(mask: pd.Series, reason: str, bars: pd.DataFrame) -> pd.DataFrame:
        drops[reason] += int((~mask).sum())
        return bars[mask]

    bars = frame[frame["session_date"] == as_of]
    clock = bars["option_timestamp"].dt.tz_convert(TIMEZONE).dt.strftime("%H:%M")
    bars = keep((clock >= config.window_start) & (clock < config.window_end), "outside window", bars)
    bars = keep(bars["matched_underlying"].astype(bool), "no spot in the same minute", bars)
    bars = keep(bars["option_close"] >= config.min_price, "below minimum price", bars)

    # Hold the plain values: Settlement is a str enum, and pandas 3 stores a mapped str
    # enum as a str dtype, dropping the enum -- convert back only where its method is needed.
    settlement = bars["option_root"].map({r: product.settlement(r).value for r in set(bars["option_root"])})
    am = settlement == Settlement.AM.value
    pm_dates = set(bars.loc[~am, "expiration"])
    shadowed = am & bars["expiration"].isin(pm_dates)
    bars, settlement = keep(~shadowed, "AM series shadowed by PM weekly", bars), settlement[~shadowed]

    expiry = bars["expiration"].dt.date
    years = {d: year_fraction(as_of, d, day_count) for d in set(expiry)}
    one_day = year_fraction(as_of, as_of + dt.timedelta(days=1), day_count)
    early = settlement.map(lambda s: Settlement(s).hours_before_close() / 24.0 * one_day)
    out = pd.DataFrame(
        {
            "timestamp": bars["option_timestamp"],
            "expiry": expiry,
            "option_root": bars["option_root"],
            "settlement": settlement,
            "right": np.where(bars["option_type"] == "call", 1, -1),
            "strike": bars["strike"].astype(float),
            "price": bars["option_close"].astype(float),
            "volume": bars["option_volume"].astype(float),
            "spot": bars["spot"].astype(float),
            "days": (expiry - as_of).map(lambda d: d.days).astype(int),
            "T": expiry.map(years).to_numpy(dtype=float) - early.to_numpy(dtype=float),
        }
    )
    out = keep(out["days"] >= config.min_days, f"under {config.min_days} days to expiry", out)
    return out.reset_index(drop=True), drops


def reference_spot(bars: pd.DataFrame) -> float:
    """The spot at the window's last minute: the snapshot's spot."""
    last = bars["timestamp"].max()
    return float(bars.loc[bars["timestamp"] == last, "spot"].iloc[0])
