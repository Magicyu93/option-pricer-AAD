from __future__ import annotations

import importlib.util
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
from urllib.parse import quote

import boto3
import pandas as pd
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Weekly / PM-settled roots (SPXW, NDXP, ...) are options on another root's
# underlying. One map for the whole pipeline: the current loader uses it too.
from .equity.massive.schema import option_underlying


TickerInput = str | Sequence[str]
DateInput = str | pd.Timestamp | Sequence[str | pd.Timestamp]
MatchMode = Literal["exact", "backward"]
ResolvedUnderlyingAsset = Literal["stocks", "indices"]
UnderlyingAsset = Literal["auto", "stocks", "indices"]
FlatAsset = Literal["options", "stocks", "indices"]


@dataclass(frozen=True)
class UnderlyingSpec:
    """Resolved provider routing information for one option underlying.

    ``canonical_ticker`` is the symbol used internally by the pricing pipeline
    (for example ``SPX``). ``provider_ticker`` is the symbol Massive uses for
    the underlying market-data feed (for example ``I:SPX`` for an index).
    """

    canonical_ticker: str
    asset_type: ResolvedUnderlyingAsset
    provider_ticker: str
    exercise_style: str | None = None
    settlement_method: str | None = None
    shares_per_contract: float | None = None
    source_option_ticker: str | None = None
    metadata_source: str = "contract"


class MassiveHistoricalDataLoader:
    """Historical Massive loader for listed options and their underlyings.

    Time-series data is read from Massive S3 Flat Files:

    * options: ``us_options_opra/minute_aggs_v1``
    * stocks/ETFs: ``us_stocks_sip/minute_aggs_v1``
    * indices: ``us_indices/minute_aggs_v1``

    Option contract reference metadata (for example ``exercise_style``) is
    fetched from Massive's REST reference API when requested.

    Underlying type is inferred primarily from option contract metadata.  In
    particular, the option CFI code identifies whether the contract references
    equity or index data; ticker-reference probing is only a fallback.  A mixed
    request such as ``["SPY", "SPX"]`` therefore routes each underlying to the
    correct Flat File dataset automatically.

    Canonical symbol convention used inside this class:

    * stock/ETF: ``SPY``
    * index: ``SPX`` (Massive provider symbol ``I:SPX`` is normalized)
    * option: ``O:SPX260918C06500000`` -> underlying ``SPX``
    """

    BUCKET = "flatfiles"
    ENDPOINT = "https://files.massive.com"
    REST_BASE = "https://api.massive.com"

    # Bump this when the processed cache schema changes.  Raw downloads remain
    # reusable across versions.
    CACHE_VERSION = ""

    DATASETS: dict[FlatAsset, str] = {
        "options": "us_options_opra/minute_aggs_v1",
        "stocks": "us_stocks_sip/minute_aggs_v1",
        "indices": "us_indices/minute_aggs_v1",
    }

    BASE_RAW_COLUMNS = {
        "ticker",
        "open",
        "close",
        "high",
        "low",
        "window_start",
    }
    OPTIONAL_TRADE_COLUMNS = {"volume", "transactions"}

    # Massive's ticker reference endpoint may classify OTC securities under
    # market="otc".  For this loader they belong to the stock/equity branch.
    MARKET_TO_UNDERLYING_ASSET: Mapping[str, ResolvedUnderlyingAsset] = {
        "stocks": "stocks",
        "otc": "stocks",
        "indices": "indices",
    }

    # OCC / ISO 10962 option CFI positions:
    #   1 category, 2 put/call, 3 exercise style, 4 underlying class,
    #   5 settlement method, 6 standard/non-standard terms.
    CFI_EXERCISE_STYLE: Mapping[str, str] = {
        "A": "american",
        "E": "european",
        "X": "unknown",
    }
    CFI_UNDERLYING_CLASS: Mapping[str, str] = {
        "B": "basket",
        "S": "equity",
        "D": "debt",
        "T": "commodity",
        "C": "currency",
        "I": "index",
        "O": "option",
        "F": "future",
        "W": "swap",
        "M": "other",
        "X": "unknown",
    }
    CFI_TO_LOADER_ASSET: Mapping[str, ResolvedUnderlyingAsset] = {
        "S": "stocks",
        "I": "indices",
    }
    CFI_SETTLEMENT_METHOD: Mapping[str, str] = {
        "P": "physical",
        "C": "cash",
        "X": "unknown",
    }
    CFI_STANDARD_TERMS: Mapping[str, str] = {
        "S": "standard",
        "N": "non_standard",
        "X": "unknown",
    }

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        data_dir: str | Path = "./market_data",
        timezone: str = "America/New_York",
        chunksize: int = 500_000,
        cache_parquet: bool = True,
        api_key: str | None = None,
        request_timeout: float = 30.0,
        verbose: bool = True,
    ) -> None:
        """Create a historical market-data loader.

        Parameters
        ----------
        access_key, secret_key
            Massive S3 Flat File credentials.  If omitted, the loader reads
            ``MASSIVE_S3_ACCESS_KEY`` and ``MASSIVE_S3_SECRET_KEY``.
        data_dir
            Root directory for raw downloads, processed caches, and reference
            metadata.
        timezone
            Time zone used for returned timestamps.
        chunksize
            Number of CSV rows processed at a time when scanning market-wide
            daily files.
        cache_parquet
            Cache filtered per-date/per-underlying data locally as Parquet.
            If no Parquet engine is installed, caching is disabled gracefully.
        api_key
            Massive REST API key.  Used for automatic underlying-type inference
            and optional option-contract metadata enrichment.  If omitted, read
            from ``MASSIVE_API_KEY``.
        request_timeout
            Timeout, in seconds, for REST reference calls.
        verbose
            Print download/cache progress messages.
        """
        self.access_key = access_key or os.getenv("MASSIVE_API_ID")
        self.secret_key = secret_key or os.getenv("MASSIVE_API_KEY")
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY")

        self.data_dir = Path(data_dir)
        self.timezone = timezone
        self.chunksize = int(chunksize)
        self.request_timeout = float(request_timeout)
        self.verbose = bool(verbose)

        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        requested_cache = bool(cache_parquet)
        parquet_available = (
            importlib.util.find_spec("pyarrow") is not None
            or importlib.util.find_spec("fastparquet") is not None
        )
        self.cache_parquet = requested_cache and parquet_available
        if requested_cache and not parquet_available:
            warnings.warn(
                "Parquet caching requested, but neither pyarrow nor fastparquet "
                "is installed. Continuing with cache_parquet=False. Install "
                "pyarrow to enable processed-data caching.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._s3 = None
        self._http = self._build_http_session()
        self._asset_type_cache = self._load_asset_type_cache()

    # ======================================================================
    # Public API: raw asset classes
    # ======================================================================

    def load_options(
        self,
        dates: DateInput,
        tickers: TickerInput,
        *,
        include_contract_metadata: bool = False,
    ) -> pd.DataFrame:
        """Load option minute bars for one or more underlying tickers."""
        underlyings = self._normalize_underlying_tickers(tickers)
        options = self._load_many_flatfile("options", dates, underlyings)
        if include_contract_metadata and not options.empty:
            options = self.add_contract_metadata(options)
        return options

    def load_stocks(
        self,
        dates: DateInput,
        tickers: TickerInput,
    ) -> pd.DataFrame:
        """Load stock/ETF minute bars from Massive stock Flat Files."""
        underlyings = self._normalize_underlying_tickers(tickers)
        return self._load_many_flatfile("stocks", dates, underlyings)

    def load_indices(
        self,
        dates: DateInput,
        tickers: TickerInput,
    ) -> pd.DataFrame:
        """Load index minute bars from Massive index Flat Files.

        Both ``"SPX"`` and ``"I:SPX"`` are accepted.  Returned data uses the
        canonical ``underlying="SPX"`` and the same normalized underlying_*
        columns as the stock loader.
        """
        underlyings = self._normalize_underlying_tickers(tickers)
        return self._load_many_flatfile("indices", dates, underlyings)

    # ======================================================================
    # Public API: automatic underlying classification
    # ======================================================================

    def resolve_underlying_spec(
        self,
        ticker: str,
        *,
        date: str | pd.Timestamp | None = None,
        option_ticker: str | None = None,
        underlying_asset: UnderlyingAsset = "auto",
        refresh: bool = False,
    ) -> UnderlyingSpec:
        """Resolve the underlying from option contract metadata.

        Contract metadata is authoritative.  The option CFI identifies the
        underlying asset class, while Massive's explicit ``exercise_style``
        field is retained when available.  Once the asset class is known,
        Massive provider symbology is deterministic: equities/ETFs use the
        canonical symbol and indices use ``I:<symbol>``.

        Ticker-reference discovery is only a defensive fallback when contract
        metadata is unavailable or incomplete.
        """
        raw = ticker.strip().upper()
        if not raw:
            raise ValueError("ticker must be non-empty")
        canonical = self._canonical_underlying_ticker(raw)
        as_of = pd.Timestamp(date).strftime("%Y-%m-%d") if date is not None else None

        if underlying_asset not in {"auto", "stocks", "indices"}:
            raise ValueError(
                "underlying_asset must be 'auto', 'stocks', or 'indices'"
            )

        metadata: dict | None = None
        if self.api_key:
            if option_ticker:
                metadata = self._fetch_option_contract_overview(
                    option_ticker, as_of
                )
            if metadata is None:
                metadata = self._fetch_representative_contract_metadata(
                    canonical, as_of
                )

        metadata_asset = (
            self._asset_from_contract_metadata(metadata)
            if metadata is not None
            else None
        )

        if underlying_asset in {"stocks", "indices"}:
            asset: ResolvedUnderlyingAsset = underlying_asset
            if metadata_asset is not None and metadata_asset != asset:
                warnings.warn(
                    f"Explicit underlying_asset={asset!r} conflicts with option "
                    f"contract metadata for {canonical!r}, which indicates "
                    f"{metadata_asset!r}. Using the explicit override.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif metadata_asset is not None:
            asset = metadata_asset
        else:
            asset = self._infer_asset_from_reference(
                canonical, date=as_of, refresh=refresh
            )

        provider_ticker = (
            f"I:{canonical}" if asset == "indices" else canonical
        )
        exercise_style = None
        settlement_method = None
        shares_per_contract = None
        metadata_source = "explicit" if underlying_asset != "auto" else "reference"

        if metadata is not None:
            provider_from_metadata = metadata.get(
                "contract_underlying_provider_ticker"
            )
            if provider_from_metadata:
                provider_ticker = str(provider_from_metadata).upper()
            exercise_style = self._none_if_na(metadata.get("exercise_style"))
            settlement_method = self._none_if_na(
                metadata.get("settlement_method")
            )
            shares = metadata.get("shares_per_contract")
            if shares is not None and not pd.isna(shares):
                try:
                    shares_per_contract = float(shares)
                except (TypeError, ValueError):
                    shares_per_contract = None
            metadata_source = "contract"

        self._remember_asset_type(canonical, asset)
        return UnderlyingSpec(
            canonical_ticker=canonical,
            asset_type=asset,
            provider_ticker=provider_ticker,
            exercise_style=exercise_style,
            settlement_method=settlement_method,
            shares_per_contract=shares_per_contract,
            source_option_ticker=option_ticker,
            metadata_source=metadata_source,
        )

    def infer_underlying_asset(
        self,
        ticker: str,
        *,
        date: str | pd.Timestamp | None = None,
        refresh: bool = False,
        option_ticker: str | None = None,
    ) -> ResolvedUnderlyingAsset:
        """Backward-compatible convenience wrapper returning only asset type."""
        return self.resolve_underlying_spec(
            ticker,
            date=date,
            option_ticker=option_ticker,
            refresh=refresh,
        ).asset_type

    def _infer_asset_from_reference(
        self,
        ticker: str,
        *,
        date: str | None,
        refresh: bool = False,
    ) -> ResolvedUnderlyingAsset:
        """Fallback cross-asset lookup used only when option metadata fails."""
        canonical = self._canonical_underlying_ticker(ticker)
        if not refresh and canonical in self._asset_type_cache:
            return self._asset_type_cache[canonical]

        self._require_api_key("underlying-asset fallback inference")
        url = f"{self.REST_BASE}/v3/reference/tickers"
        candidates: list[tuple[str, str, ResolvedUnderlyingAsset]] = [
            (canonical, "stocks", "stocks"),
            (f"I:{canonical}", "indices", "indices"),
        ]

        for provider_ticker, market, asset in candidates:
            params: dict[str, object] = {
                "ticker": provider_ticker,
                "market": market,
                "active": "true",
                "limit": 10,
                "apiKey": self.api_key,
            }
            if date is not None:
                params["date"] = date

            response = self._http.get(
                url, params=params, timeout=self.request_timeout
            )
            if response.status_code in {403, 404}:
                continue
            response.raise_for_status()
            for record in response.json().get("results") or []:
                returned_ticker = str(record.get("ticker") or "").upper()
                returned_market = str(record.get("market") or "").lower()
                if returned_ticker != provider_ticker:
                    continue
                mapped = self.MARKET_TO_UNDERLYING_ASSET.get(returned_market)
                if mapped is not None:
                    self._remember_asset_type(canonical, mapped)
                    return mapped
                self._remember_asset_type(canonical, asset)
                return asset

        raise ValueError(
            f"Could not infer the underlying asset class for {canonical!r} from "
            "option metadata or Massive ticker reference data. Pass "
            "underlying_asset='stocks' or 'indices' to override routing."
        )

    def infer_underlying_assets(
        self,
        tickers: TickerInput,
        *,
        date: str | pd.Timestamp | None = None,
        refresh: bool = False,
    ) -> dict[str, ResolvedUnderlyingAsset]:
        """Infer asset type for multiple tickers."""
        canonical = self._normalize_underlying_tickers(tickers)
        return {
            ticker: self.infer_underlying_asset(
                ticker,
                date=date,
                refresh=refresh,
            )
            for ticker in canonical
        }

    def load_underlying(
        self,
        dates: DateInput,
        tickers: TickerInput,
        *,
        underlying_asset: UnderlyingAsset = "auto",
    ) -> pd.DataFrame:
        """Load normalized underlying bars using metadata-driven routing."""
        dates_ = self._normalize_dates(dates)
        tickers_ = self._normalize_underlying_tickers(tickers)
        specs = self._resolve_underlying_specs(
            tickers_,
            underlying_asset=underlying_asset,
            as_of_date=min(dates_),
            option_rows=None,
        )
        return self._load_underlyings_from_specs(dates_, specs)

    # ======================================================================
    # Public API: option + underlying market view
    # ======================================================================

    def load_market(
        self,
        dates: DateInput,
        tickers: TickerInput,
        *,
        underlying_asset: UnderlyingAsset = "auto",
        match: MatchMode = "exact",
        tolerance: str | pd.Timedelta = "1min",
        spot_column: str | None = None,
        include_contract_metadata: bool = False,
    ) -> pd.DataFrame:
        """Load option bars and attach the corresponding underlying price.

        Routing is metadata-driven.  For each requested underlying, an observed
        option contract is used to resolve its asset class.  Equity/ETF
        underlyings route to stock Flat Files; index underlyings route to index
        Flat Files.  Provider symbols such as ``I:SPX`` never participate in
        the join: both sides are normalized to canonical ``SPX`` first.
        """
        dates_ = self._normalize_dates(dates)
        tickers_ = self._normalize_underlying_tickers(tickers)

        options = self.load_options(
            dates_, tickers_, include_contract_metadata=False
        )

        specs = self._resolve_underlying_specs(
            tickers_,
            underlying_asset=underlying_asset,
            as_of_date=min(dates_),
            option_rows=options,
        )
        underlyings = self._load_underlyings_from_specs(dates_, specs)

        if include_contract_metadata and not options.empty:
            options = self.add_contract_metadata(options)

        return self.combine_market(
            options,
            underlyings,
            match=match,
            tolerance=tolerance,
            spot_column=spot_column or "underlying_close",
        )

    def load_market_range(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        tickers: TickerInput,
        **kwargs,
    ) -> pd.DataFrame:
        """Load a business-day date range and combine options/underlyings.

        Exchange holidays are harmless: if a daily Flat File is absent, that
        date is skipped.
        """
        dates = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        return self.load_market(dates, tickers, **kwargs)

    def combine_market(
        self,
        options: pd.DataFrame,
        underlyings: pd.DataFrame,
        *,
        match: MatchMode = "exact",
        tolerance: str | pd.Timedelta = "1min",
        spot_column: str = "underlying_close",
    ) -> pd.DataFrame:
        """Attach normalized underlying bars to option observations.

        By this stage routing has already been resolved from contract metadata.
        Therefore the merge is deliberately asset-agnostic and uses only:

        ``session_date + canonical underlying + timestamp``.

        Raw provider symbols (for example ``I:SPX``) are normalized before
        this stage and do not appear in the combined market dataframe.
        """
        if options.empty:
            return options.copy()

        required_options = {"session_date", "underlying", "timestamp"}
        missing_options = required_options.difference(options.columns)
        if missing_options:
            raise ValueError(
                "options is missing required columns: "
                + ", ".join(sorted(missing_options))
            )

        left = options.copy()
        left["option_timestamp"] = left["timestamp"]
        left["join_timestamp"] = left["timestamp"].dt.floor("min")

        if underlyings.empty:
            out = left.copy()
            for col in self._unified_underlying_columns():
                if col not in out.columns:
                    out[col] = pd.NA
            out["underlying_timestamp"] = pd.NaT
            out["underlying_age"] = pd.NaT
            out["matched_underlying"] = False
            out["spot"] = pd.NA
            return out

        required_underlyings = {
            "session_date", "underlying", "timestamp", "underlying_close"
        }
        missing_underlyings = required_underlyings.difference(underlyings.columns)
        if missing_underlyings:
            raise ValueError(
                "underlyings is missing required columns: "
                + ", ".join(sorted(missing_underlyings))
            )
        if spot_column not in underlyings.columns:
            raise ValueError(
                f"spot_column={spot_column!r} is not present in underlying data"
            )

        right = underlyings.copy()
        right["underlying_timestamp"] = right["timestamp"]
        right["join_timestamp"] = right["timestamp"].dt.floor("min")

        if match == "exact":
            keys = ["session_date", "underlying", "join_timestamp"]
            right = (
                right.sort_values([*keys, "underlying_timestamp"])
                .drop_duplicates(subset=keys, keep="last")
            )
            payload = right.drop(columns=["timestamp"], errors="ignore")
            out = left.merge(
                payload,
                on=keys,
                how="left",
                validate="many_to_one",
            )
        elif match == "backward":
            out = self._combine_backward(
                left, right, tolerance=pd.Timedelta(tolerance)
            )
        else:
            raise ValueError("match must be 'exact' or 'backward'")

        out["spot"] = out[spot_column]
        out["matched_underlying"] = out["underlying_timestamp"].notna()
        out["underlying_age"] = (
            out["option_timestamp"] - out["underlying_timestamp"]
        )

        # ``join_timestamp`` is an internal minute-bucket key only.  Do not
        # expose it in the final market dataframe.
        out = out.drop(columns=["join_timestamp"], errors="ignore")

        sort_cols = [
            "session_date",
            "underlying",
            "option_timestamp",
            "expiration",
            "option_type",
            "strike",
        ]
        sort_cols = [c for c in sort_cols if c in out.columns]
        return out.sort_values(sort_cols).reset_index(drop=True)

    def _combine_backward(
        self,
        options: pd.DataFrame,
        underlyings: pd.DataFrame,
        *,
        tolerance: pd.Timedelta,
    ) -> pd.DataFrame:
        """Backward as-of join on canonical underlying, never future data."""
        frames: list[pd.DataFrame] = []

        for (date, underlying), opt in options.groupby(
            ["session_date", "underlying"], sort=False
        ):
            base = underlyings[
                (underlyings["session_date"] == date)
                & (underlyings["underlying"] == underlying)
            ].copy()

            if base.empty:
                joined = opt.copy()
                for col in self._unified_underlying_columns():
                    if col not in joined.columns:
                        joined[col] = pd.NA
                joined["underlying_timestamp"] = pd.NaT
                frames.append(joined)
                continue

            left = opt.sort_values("option_timestamp").copy()
            right = (
                base.sort_values("underlying_timestamp")
                .drop_duplicates(subset=["underlying_timestamp"], keep="last")
                .drop(
                    columns=[
                        "session_date",
                        "underlying",
                        "timestamp",
                        "join_timestamp",
                    ],
                    errors="ignore",
                )
            )

            joined = pd.merge_asof(
                left,
                right,
                left_on="option_timestamp",
                right_on="underlying_timestamp",
                direction="backward",
                tolerance=tolerance,
            )
            frames.append(joined)

        return self._concat(frames)

    # Backward-compatible wrappers from earlier versions -------------------

    def combine(
        self,
        options: pd.DataFrame,
        stocks: pd.DataFrame,
        *,
        match: MatchMode = "exact",
        tolerance: str | pd.Timedelta = "1min",
        spot_column: str = "underlying_close",
    ) -> pd.DataFrame:
        """Backward-compatible stock-only wrapper around :meth:`combine_market`."""
        stocks = self._ensure_unified_underlying_schema(stocks, "stocks")
        if spot_column == "stock_close":
            spot_column = "underlying_close"
        return self.combine_market(
            options,
            stocks,
            match=match,
            tolerance=tolerance,
            spot_column=spot_column,
        )

    def combine_indices(
        self,
        options: pd.DataFrame,
        indices: pd.DataFrame,
        *,
        match: MatchMode = "exact",
        tolerance: str | pd.Timedelta = "1min",
        spot_column: str = "underlying_close",
    ) -> pd.DataFrame:
        """Backward-compatible index-only wrapper around :meth:`combine_market`."""
        indices = self._ensure_unified_underlying_schema(indices, "indices")
        if spot_column == "index_close":
            spot_column = "underlying_close"
        return self.combine_market(
            options,
            indices,
            match=match,
            tolerance=tolerance,
            spot_column=spot_column,
        )

    # ======================================================================
    # Public API: option reference metadata
    # ======================================================================

    def load_contract_metadata(
        self,
        date: str | pd.Timestamp,
        tickers: TickerInput,
    ) -> pd.DataFrame:
        """Load option-contract reference metadata as of one historical date."""
        date_ = pd.Timestamp(date).strftime("%Y-%m-%d")
        tickers_ = self._normalize_underlying_tickers(tickers)
        frames = [
            self._load_contract_metadata_one(date_, ticker)
            for ticker in tickers_
        ]
        return self._concat(frames)

    def add_contract_metadata(self, options: pd.DataFrame) -> pd.DataFrame:
        """Attach exercise style and related contract fields to option rows."""
        if options.empty:
            return options.copy()

        required = {"session_date", "underlying", "option_ticker"}
        missing = required.difference(options.columns)
        if missing:
            raise ValueError(
                "options is missing required columns: "
                + ", ".join(sorted(missing))
            )

        keys = (
            options[["session_date", "underlying"]]
            .drop_duplicates()
            .sort_values(["session_date", "underlying"])
        )

        metadata_frames = []
        for row in keys.itertuples(index=False):
            date = pd.Timestamp(row.session_date).strftime("%Y-%m-%d")
            metadata_frames.append(
                self._load_contract_metadata_one(date, str(row.underlying))
            )

        metadata = self._concat(metadata_frames)
        if metadata.empty:
            out = options.copy()
            for col in self._contract_metadata_columns():
                if col not in out.columns:
                    out[col] = pd.NA
            return out

        keep = [
            "session_date",
            "underlying",
            "option_ticker",
            *self._contract_metadata_columns(),
        ]
        metadata = metadata[[c for c in keep if c in metadata.columns]]

        return options.merge(
            metadata,
            on=["session_date", "underlying", "option_ticker"],
            how="left",
            validate="many_to_one",
        )

    # ======================================================================
    # Underlying resolution / normalization
    # ======================================================================

    def _resolve_underlying_specs(
        self,
        tickers: list[str],
        *,
        underlying_asset: UnderlyingAsset,
        as_of_date: str | None,
        option_rows: pd.DataFrame | None,
    ) -> dict[str, UnderlyingSpec]:
        """Resolve one metadata-backed routing specification per underlying."""
        specs: dict[str, UnderlyingSpec] = {}

        for ticker in tickers:
            representative_option = None
            representative_date = as_of_date

            if option_rows is not None and not option_rows.empty:
                subset = option_rows[option_rows["underlying"] == ticker]
                if not subset.empty:
                    row = subset.sort_values(
                        ["session_date", "timestamp", "option_ticker"]
                    ).iloc[0]
                    representative_option = str(row["option_ticker"])
                    representative_date = pd.Timestamp(
                        row["session_date"]
                    ).strftime("%Y-%m-%d")

            specs[ticker] = self.resolve_underlying_spec(
                ticker,
                date=representative_date,
                option_ticker=representative_option,
                underlying_asset=underlying_asset,
            )

        return specs

    def _load_underlyings_from_specs(
        self,
        dates: list[str],
        specs: Mapping[str, UnderlyingSpec],
    ) -> pd.DataFrame:
        """Route each underlying to its metadata-determined Flat File feed."""
        stock_tickers = [
            spec.canonical_ticker
            for spec in specs.values()
            if spec.asset_type == "stocks"
        ]
        index_tickers = [
            spec.canonical_ticker
            for spec in specs.values()
            if spec.asset_type == "indices"
        ]

        frames: list[pd.DataFrame] = []
        if stock_tickers:
            frames.append(self.load_stocks(dates, stock_tickers))
        if index_tickers:
            frames.append(self.load_indices(dates, index_tickers))

        out = self._concat(frames)
        if out.empty:
            return out

        # Asset-specific processors already normalize to the common
        # underlying schema.  ``underlying`` is the canonical pricing key
        # (SPX), while ``underlying_ticker`` preserves Massive's original
        # provider symbol (I:SPX) for auditability.
        asset_map = {
            spec.canonical_ticker: spec.asset_type
            for spec in specs.values()
        }
        out["underlying_asset"] = out["underlying"].map(asset_map)
        return out

    def _ensure_unified_underlying_schema(
        self,
        df: pd.DataFrame,
        asset: ResolvedUnderlyingAsset,
    ) -> pd.DataFrame:
        """Return only the canonical common underlying columns.

        Provider-specific stock/index column names are accepted for backward
        compatibility at this boundary, but they are normalized immediately
        and are not propagated downstream.
        """
        if df.empty:
            return df.copy()

        out = df.copy()
        out["underlying_asset"] = asset

        if "underlying_ticker" not in out.columns:
            if asset == "stocks":
                if "stock_ticker" in out.columns:
                    out["underlying_ticker"] = out["stock_ticker"]
                elif "ticker" in out.columns:
                    out["underlying_ticker"] = out["ticker"]
                else:
                    out["underlying_ticker"] = out.get("underlying", pd.NA)
            else:
                if "index_ticker" in out.columns:
                    out["underlying_ticker"] = out["index_ticker"]
                elif "ticker" in out.columns:
                    out["underlying_ticker"] = out["ticker"]
                elif "underlying" in out.columns:
                    out["underlying_ticker"] = "I:" + out["underlying"].astype("string")
                else:
                    out["underlying_ticker"] = pd.NA

        prefix = "stock" if asset == "stocks" else "index"
        for field in ["open", "high", "low", "close"]:
            target = f"underlying_{field}"
            source = f"{prefix}_{field}"
            if target not in out.columns and source in out.columns:
                out[target] = out[source]

        if "underlying_volume" not in out.columns:
            out["underlying_volume"] = (
                out.get("stock_volume", pd.NA) if asset == "stocks" else pd.NA
            )
        if "underlying_transactions" not in out.columns:
            out["underlying_transactions"] = (
                out.get("stock_transactions", pd.NA)
                if asset == "stocks"
                else pd.NA
            )
        if "underlying_window_start_ns" not in out.columns:
            source = (
                "stock_window_start_ns"
                if asset == "stocks"
                else "index_window_start_ns"
            )
            out["underlying_window_start_ns"] = out.get(source, pd.NA)

        columns = [
            "session_date",
            "timestamp",
            "underlying",
            *self._unified_underlying_columns(),
        ]
        return out[[c for c in columns if c in out.columns]].copy()

    @staticmethod
    def _unified_underlying_columns() -> list[str]:
        return [
            "underlying_ticker",
            "underlying_asset",
            "underlying_open",
            "underlying_high",
            "underlying_low",
            "underlying_close",
            "underlying_volume",
            "underlying_transactions",
            "underlying_window_start_ns",
        ]

    # ======================================================================
    # Unified Flat File loading
    # ======================================================================

    def _load_many_flatfile(
        self,
        asset: FlatAsset,
        dates: DateInput,
        tickers: TickerInput,
    ) -> pd.DataFrame:
        dates_ = self._normalize_dates(dates)
        tickers_ = self._normalize_underlying_tickers(tickers)

        frames = [
            self._load_one_date_flatfile(asset, date, tickers_)
            for date in dates_
        ]
        return self._concat(frames)

    def _load_one_date_flatfile(
        self,
        asset: FlatAsset,
        date: str,
        tickers: list[str],
    ) -> pd.DataFrame:
        cached, missing = self._read_cache(asset, date, tickers)

        fresh = pd.DataFrame()
        if missing:
            raw_path = self._ensure_raw_file(asset, date)
            if raw_path is not None:
                fresh = self._scan_raw(asset, raw_path, date, missing)
                self._write_cache(asset, date, fresh, missing)

        return self._concat([cached, fresh])

    def _scan_raw(
        self,
        asset: FlatAsset,
        path: Path,
        date: str,
        tickers: list[str],
    ) -> pd.DataFrame:
        wanted = set(tickers)
        frames: list[pd.DataFrame] = []
        usecols = self._raw_usecols(path, asset)

        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            chunksize=self.chunksize,
        ):
            symbols = chunk["ticker"].astype("string")

            if asset == "stocks":
                canonical = symbols
                mask = canonical.isin(wanted)

            elif asset == "indices":
                canonical = symbols.str.replace(r"^I:", "", regex=True)
                mask = canonical.isin(wanted)

            else:  # options
                # Massive/OCC option symbol:
                # O:<root><YYMMDD><C/P><8-digit strike>
                # The suffix is fixed at 15 characters.  Parsing from the right
                # avoids making assumptions about valid underlying characters.
                # A root counts when it *maps* to a requested underlying,
                # so "SPX" also scans in its SPXW weeklies.
                roots = symbols.str.slice(2, -15)
                mask = symbols.str.startswith("O:", na=False) & option_underlying(
                    roots
                ).isin(wanted)

            selected = chunk.loc[mask].copy()
            if not selected.empty:
                frames.append(selected)

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        if asset == "options":
            return self._process_options(raw, date)
        if asset == "stocks":
            return self._process_stocks(raw, date)
        return self._process_indices(raw, date)

    def _raw_usecols(self, path: Path, asset: FlatAsset) -> list[str]:
        """Inspect a Flat File header and read only columns this loader needs."""
        header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
        available = set(header)

        missing = self.BASE_RAW_COLUMNS.difference(available)
        if missing:
            raise ValueError(
                f"{asset} flat file {path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        desired = set(self.BASE_RAW_COLUMNS)
        if asset in {"options", "stocks"}:
            desired |= self.OPTIONAL_TRADE_COLUMNS

        return [col for col in header if col in desired]

    # ======================================================================
    # Asset-specific processing
    # ======================================================================

    def _process_options(
        self,
        df: pd.DataFrame,
        session_date: str,
    ) -> pd.DataFrame:
        df = df.copy()
        self._ensure_columns(df, ["volume", "transactions"])

        symbols = df["ticker"].astype("string")
        df["option_root"] = symbols.str.slice(2, -15)
        df["underlying"] = option_underlying(df["option_root"])
        expiration_code = symbols.str.slice(-15, -9)
        type_code = symbols.str.slice(-9, -8)
        strike_code = symbols.str.slice(-8)

        df["expiration"] = pd.to_datetime(
            expiration_code,
            format="%y%m%d",
            errors="coerce",
        )
        df["option_type"] = type_code.map({"C": "call", "P": "put"})
        df["strike"] = pd.to_numeric(strike_code, errors="coerce") / 1000.0
        df["timestamp"] = self._convert_flatfile_timestamp(df["window_start"])
        df["session_date"] = pd.Timestamp(session_date).date()

        df = df.dropna(
            subset=["underlying", "expiration", "option_type", "strike", "timestamp"]
        )

        df = df.rename(
            columns={
                "ticker": "option_ticker",
                "open": "option_open",
                "high": "option_high",
                "low": "option_low",
                "close": "option_close",
                "volume": "option_volume",
                "transactions": "option_transactions",
                "window_start": "option_window_start_ns",
            }
        )

        columns = [
            "session_date",
            "timestamp",
            "underlying",
            "option_root",
            "option_ticker",
            "expiration",
            "option_type",
            "strike",
            "option_open",
            "option_high",
            "option_low",
            "option_close",
            "option_volume",
            "option_transactions",
            "option_window_start_ns",
        ]
        return (
            df[columns]
            .sort_values(
                ["underlying", "timestamp", "expiration", "option_type", "strike"]
            )
            .reset_index(drop=True)
        )

    def _process_stocks(
        self,
        df: pd.DataFrame,
        session_date: str,
    ) -> pd.DataFrame:
        df = df.copy()
        self._ensure_columns(df, ["volume", "transactions"])

        df["timestamp"] = self._convert_flatfile_timestamp(df["window_start"])
        df["session_date"] = pd.Timestamp(session_date).date()
        df["underlying_ticker"] = df["ticker"].astype("string").str.upper()
        df["underlying"] = df["underlying_ticker"]
        df["underlying_asset"] = "stocks"

        df = df.rename(
            columns={
                "open": "underlying_open",
                "high": "underlying_high",
                "low": "underlying_low",
                "close": "underlying_close",
                "volume": "underlying_volume",
                "transactions": "underlying_transactions",
                "window_start": "underlying_window_start_ns",
            }
        )

        columns = [
            "session_date",
            "timestamp",
            "underlying",
            *self._unified_underlying_columns(),
        ]
        return (
            df[columns]
            .dropna(subset=["timestamp", "underlying"])
            .sort_values(["underlying", "timestamp"])
            .reset_index(drop=True)
        )

    def _process_indices(
        self,
        df: pd.DataFrame,
        session_date: str,
    ) -> pd.DataFrame:
        df = df.copy()

        # Massive uses provider symbols such as I:SPX.  Normalize them once
        # here so all downstream code sees the canonical option underlying
        # symbol SPX.
        df["underlying_ticker"] = df["ticker"].astype("string").str.upper()
        df["underlying"] = (
            df["underlying_ticker"].str.replace(r"^I:", "", regex=True)
        )
        df["timestamp"] = self._convert_flatfile_timestamp(df["window_start"])
        df["session_date"] = pd.Timestamp(session_date).date()
        df["underlying_asset"] = "indices"

        df = df.rename(
            columns={
                "open": "underlying_open",
                "high": "underlying_high",
                "low": "underlying_low",
                "close": "underlying_close",
                "window_start": "underlying_window_start_ns",
            }
        )
        # Index aggregates are index values rather than exchange trades.
        df["underlying_volume"] = pd.NA
        df["underlying_transactions"] = pd.NA

        columns = [
            "session_date",
            "timestamp",
            "underlying",
            *self._unified_underlying_columns(),
        ]
        return (
            df[columns]
            .dropna(subset=["timestamp", "underlying"])
            .sort_values(["underlying", "timestamp"])
            .reset_index(drop=True)
        )

    def _convert_flatfile_timestamp(self, values: pd.Series) -> pd.Series:
        return (
            pd.to_datetime(values, unit="ns", utc=True, errors="coerce")
            .dt.tz_convert(self.timezone)
        )

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA

    # ======================================================================
    # Option contract reference metadata
    # ======================================================================

    @staticmethod
    def _contract_metadata_columns() -> list[str]:
        return [
            "exercise_style",
            "contract_underlying_asset",
            "contract_underlying_class",
            "contract_underlying_provider_ticker",
            "settlement_method",
            "standard_terms",
            "shares_per_contract",
            "primary_exchange",
            "cfi",
        ]

    def _enrich_contract_metadata_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive normalized contract attributes from Massive metadata + CFI."""
        if df.empty:
            return df.copy()

        out = df.copy()
        if "cfi" not in out.columns:
            out["cfi"] = pd.NA
        cfi = out["cfi"].astype("string").str.upper()

        cfi_exercise = cfi.str.slice(2, 3).map(self.CFI_EXERCISE_STYLE)
        if "exercise_style" not in out.columns:
            out["exercise_style"] = cfi_exercise
        else:
            explicit = out["exercise_style"].astype("string").str.lower()
            out["exercise_style"] = explicit.where(explicit.notna(), cfi_exercise)

        underlying_code = cfi.str.slice(3, 4)
        out["contract_underlying_class"] = underlying_code.map(
            self.CFI_UNDERLYING_CLASS
        )
        out["contract_underlying_asset"] = underlying_code.map(
            self.CFI_TO_LOADER_ASSET
        )
        out["settlement_method"] = cfi.str.slice(4, 5).map(
            self.CFI_SETTLEMENT_METHOD
        )
        out["standard_terms"] = cfi.str.slice(5, 6).map(
            self.CFI_STANDARD_TERMS
        )

        if "underlying" in out.columns:
            canonical = (
                out["underlying"]
                .astype("string")
                .str.upper()
                .str.replace(r"^I:", "", regex=True)
            )
            provider = canonical.copy()
            is_index = out["contract_underlying_asset"].eq("indices")
            provider = provider.where(~is_index, "I:" + canonical)
            out["contract_underlying_provider_ticker"] = provider

        return out

    def _asset_from_contract_metadata(
        self,
        metadata: Mapping[str, object] | pd.Series,
    ) -> ResolvedUnderlyingAsset | None:
        """Return loader asset class from one Massive option contract record."""
        raw_asset = metadata.get("contract_underlying_asset")
        if raw_asset in {"stocks", "indices"}:
            return raw_asset  # type: ignore[return-value]

        cfi = str(metadata.get("cfi") or "").upper()
        if len(cfi) >= 4:
            return self.CFI_TO_LOADER_ASSET.get(cfi[3])
        return None

    def _fetch_option_contract_overview(
        self,
        option_ticker: str,
        as_of: str | None,
    ) -> dict | None:
        self._require_api_key("option contract metadata")
        encoded = quote(option_ticker, safe="")
        url = f"{self.REST_BASE}/v3/reference/options/contracts/{encoded}"
        params: dict[str, object] = {"apiKey": self.api_key}
        if as_of is not None:
            params["as_of"] = as_of

        response = self._http.get(url, params=params, timeout=self.request_timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        record = response.json().get("results") or {}
        if not record:
            return None

        frame = pd.DataFrame([record]).rename(
            columns={"underlying_ticker": "underlying"}
        )
        if "underlying" in frame.columns:
            frame["underlying"] = (
                frame["underlying"]
                .astype("string")
                .str.replace(r"^I:", "", regex=True)
            )
        frame = self._enrich_contract_metadata_frame(frame)
        return frame.iloc[0].to_dict()

    def _fetch_representative_contract_metadata(
        self,
        underlying: str,
        as_of: str | None,
    ) -> dict | None:
        """Fetch one contract record, enough to classify an underlying."""
        self._require_api_key("option contract metadata")
        url = f"{self.REST_BASE}/v3/reference/options/contracts"

        for expired in (False, True):
            params: dict[str, object] = {
                "underlying_ticker": underlying,
                "expired": str(expired).lower(),
                "limit": 10,
                "sort": "ticker",
                "order": "asc",
                "apiKey": self.api_key,
            }
            if as_of is not None:
                params["as_of"] = as_of

            response = self._http.get(
                url, params=params, timeout=self.request_timeout
            )
            if response.status_code in {403, 404}:
                continue
            response.raise_for_status()
            results = response.json().get("results") or []
            if not results:
                continue

            frame = pd.DataFrame.from_records(results).rename(
                columns={"underlying_ticker": "underlying"}
            )
            if "underlying" in frame.columns:
                frame["underlying"] = (
                    frame["underlying"]
                    .astype("string")
                    .str.replace(r"^I:", "", regex=True)
                )
            frame = self._enrich_contract_metadata_frame(frame)
            for _, row in frame.iterrows():
                if self._asset_from_contract_metadata(row) is not None:
                    return row.to_dict()
        return None

    def _contract_cache_path(self, date: str, underlying: str) -> Path:
        return (
            self.data_dir
            / "parquet"
            / self.CACHE_VERSION
            / "contracts"
            / f"date={date}"
            / f"underlying={underlying}"
            / "contracts.parquet"
        )

    def _load_contract_metadata_one(
        self,
        date: str,
        underlying: str,
    ) -> pd.DataFrame:
        cache_path = self._contract_cache_path(date, underlying)
        if self.cache_parquet and cache_path.exists():
            return pd.read_parquet(cache_path)

        metadata = self._fetch_contract_metadata(date, underlying)

        if self.cache_parquet and not metadata.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            metadata.to_parquet(cache_path, index=False, compression="zstd")

        return metadata

    def _fetch_contract_metadata(
        self,
        date: str,
        underlying: str,
    ) -> pd.DataFrame:
        self._require_api_key("contract metadata")

        records: list[dict] = []

        # Historical datasets often contain contracts that are expired today.
        # Query both sides of the expired filter and deduplicate to avoid silently
        # losing reference metadata for historical contracts.
        for expired in (False, True):
            url: str | None = f"{self.REST_BASE}/v3/reference/options/contracts"
            params: dict[str, object] | None = {
                "underlying_ticker": underlying,
                "as_of": date,
                "expired": str(expired).lower(),
                "limit": 1000,
                "sort": "ticker",
                "order": "asc",
                "apiKey": self.api_key,
            }

            while url:
                response = self._http.get(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                records.extend(payload.get("results", []))

                url = payload.get("next_url")
                params = {"apiKey": self.api_key} if url else None

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        df = df.rename(
            columns={
                "ticker": "option_ticker",
                "underlying_ticker": "underlying",
            }
        )
        if "underlying" in df.columns:
            df["underlying"] = (
                df["underlying"]
                .astype("string")
                .str.replace(r"^I:", "", regex=True)
            )
        df = self._enrich_contract_metadata_frame(df)
        df["session_date"] = pd.Timestamp(date).date()

        columns = [
            "session_date",
            "underlying",
            "option_ticker",
            *self._contract_metadata_columns(),
        ]
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA

        return (
            df[columns]
            .drop_duplicates(
                subset=["session_date", "underlying", "option_ticker"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    # ======================================================================
    # Processed-data cache
    # ======================================================================

    def _cache_path(
        self,
        asset: FlatAsset,
        date: str,
        ticker: str,
    ) -> Path:
        return (
            self.data_dir
            / "parquet"
            / self.CACHE_VERSION
            / asset
            / f"date={date}"
            / f"underlying={ticker}"
            / "minute.parquet"
        )

    def _empty_cache_marker(
        self,
        asset: FlatAsset,
        date: str,
        ticker: str,
    ) -> Path:
        return self._cache_path(asset, date, ticker).with_name(".empty")

    @staticmethod
    def _cache_is_current(asset: FlatAsset, path: Path) -> bool:
        """Whether a cached options file was written with the root mapping.

        A file without ``option_root`` predates it, so it lacks the weekly
        roots (most of SPX volume): a miss to rebuild from the raw file.
        """
        if asset != "options":
            return True
        import pyarrow.parquet as pq

        return "option_root" in pq.read_schema(path).names

    def _read_cache(
        self,
        asset: FlatAsset,
        date: str,
        tickers: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        if not self.cache_parquet:
            return pd.DataFrame(), tickers

        frames: list[pd.DataFrame] = []
        missing: list[str] = []

        for ticker in tickers:
            path = self._cache_path(asset, date, ticker)
            empty_marker = self._empty_cache_marker(asset, date, ticker)

            if path.exists() and self._cache_is_current(asset, path):
                frames.append(pd.read_parquet(path))
            elif empty_marker.exists():
                continue
            else:
                missing.append(ticker)

        return self._concat(frames), missing

    def _write_cache(
        self,
        asset: FlatAsset,
        date: str,
        df: pd.DataFrame,
        tickers: list[str],
    ) -> None:
        if not self.cache_parquet:
            return

        found: set[str] = set()
        if not df.empty and "underlying" in df.columns:
            found = set(df["underlying"].dropna().astype(str).unique())

        for ticker in tickers:
            path = self._cache_path(asset, date, ticker)
            empty_marker = self._empty_cache_marker(asset, date, ticker)
            path.parent.mkdir(parents=True, exist_ok=True)

            part = (
                df[df["underlying"] == ticker]
                if not df.empty and "underlying" in df.columns
                else pd.DataFrame()
            )

            if not part.empty:
                part.to_parquet(path, index=False, compression="zstd")
                if empty_marker.exists():
                    empty_marker.unlink()
            elif ticker not in found:
                empty_marker.touch(exist_ok=True)

    # ======================================================================
    # Massive S3 Flat Files
    # ======================================================================

    def _ensure_raw_file(
        self,
        asset: FlatAsset,
        date: str,
    ) -> Path | None:
        path = self.data_dir / "raw" / asset / f"{date}.csv.gz"
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        key = self._s3_key(self.DATASETS[asset], date)
        temp_path = path.with_suffix(path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink()

        self._log(f"Downloading {asset} data for {date}...")

        try:
            self._get_s3().download_file(
                self.BUCKET,
                key,
                str(temp_path),
            )
            temp_path.replace(path)
        except ClientError as exc:
            if temp_path.exists():
                temp_path.unlink()
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                self._log(f"No {asset} file available for {date}; skipping.")
                return None
            if code in {"403", "AccessDenied"}:
                raise PermissionError(
                    f"Access denied to Massive {asset} Flat File {key}. "
                    "Check that your subscription includes this dataset and "
                    "that the S3 credentials are correct."
                ) from exc
            raise
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return path

    @staticmethod
    def _s3_key(dataset: str, date: str) -> str:
        dt = pd.Timestamp(date)
        return f"{dataset}/{dt:%Y}/{dt:%m}/{dt:%Y-%m-%d}.csv.gz"

    def _get_s3(self):
        if self._s3 is not None:
            return self._s3

        if not self.access_key or not self.secret_key:
            raise RuntimeError(
                "Massive S3 credentials are missing. Set "
                "MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY, "
                "or pass access_key= and secret_key=."
            )

        session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self._s3 = session.client(
            "s3",
            endpoint_url=self.ENDPOINT,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )
        return self._s3

    # ======================================================================
    # Reference cache / HTTP
    # ======================================================================

    def _build_http_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _asset_type_cache_path(self) -> Path:
        return self.data_dir / "reference" / "underlying_asset_types.json"

    def _load_asset_type_cache(self) -> dict[str, ResolvedUnderlyingAsset]:
        path = self._asset_type_cache_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

        out: dict[str, ResolvedUnderlyingAsset] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if value in {"stocks", "indices"}:
                    out[str(key).upper()] = value
        return out

    def _remember_asset_type(
        self,
        ticker: str,
        asset: ResolvedUnderlyingAsset,
    ) -> None:
        ticker = self._canonical_underlying_ticker(ticker)
        if self._asset_type_cache.get(ticker) == asset:
            return

        self._asset_type_cache[ticker] = asset
        path = self._asset_type_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._asset_type_cache, indent=2, sort_keys=True))
        temp.replace(path)

    # ======================================================================
    # Input normalization / helpers
    # ======================================================================

    @staticmethod
    def _normalize_tickers(tickers: TickerInput) -> list[str]:
        values = [tickers] if isinstance(tickers, str) else list(tickers)
        out = [x.strip().upper() for x in values if x and x.strip()]
        if not out:
            raise ValueError("At least one ticker is required.")
        return list(dict.fromkeys(out))

    def _normalize_underlying_tickers(self, tickers: TickerInput) -> list[str]:
        return list(
            dict.fromkeys(
                self._canonical_underlying_ticker(ticker)
                for ticker in self._normalize_tickers(tickers)
            )
        )

    @staticmethod
    def _canonical_underlying_ticker(ticker: str) -> str:
        ticker = ticker.strip().upper()
        return ticker[2:] if ticker.startswith("I:") else ticker

    @staticmethod
    def _normalize_dates(dates: DateInput) -> list[str]:
        if isinstance(dates, (str, pd.Timestamp)):
            values = [dates]
        else:
            values = list(dates)

        out = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in values]
        if not out:
            raise ValueError("At least one date is required.")
        return list(dict.fromkeys(out))

    @staticmethod
    def _none_if_na(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return str(value)

    def _require_api_key(self, purpose: str) -> None:
        if not self.api_key:
            raise RuntimeError(
                f"Massive REST API key is required for {purpose}. "
                "Set MASSIVE_API_KEY or pass api_key= to the loader. "
                "Alternatively pass underlying_asset='stocks' or 'indices' "
                "explicitly when automatic inference is not needed."
            )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _concat(frames) -> pd.DataFrame:
        frames = [x for x in frames if x is not None and not x.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()




if __name__ == "__main__":
    from pathlib import Path

    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    loader = MassiveHistoricalDataLoader(data_dir= current_dir / "market_data")

    market = loader.load_market(
        dates="2026-09-08",
        tickers=["SPX", "SPY"],
    )

    print(market.head())
