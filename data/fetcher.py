"""Historical and realtime XAUUSD price fetching via yfinance and Alpha Vantage."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf

# Yahoo Finance symbol for spot gold vs USD (daily OHLCV).
_XAUUSD_YF_SYMBOL: str = "GC=F"

# Alpha Vantage physical currency pair for gold (XAU) priced in USD.
_AV_BASE_URL: str = "https://www.alphavantage.co/query"


def fetch_xauusd_history(
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    repair: bool = True,
) -> pd.DataFrame:
    """
    Download multi-year daily OHLCV history for XAUUSD using yfinance.

    Args:
        period: yfinance lookback window (default five years).
        interval: Bar size; daily bars by default.
        auto_adjust: Pass-through to yfinance for split/dividend adjustment.
        repair: Pass-through to yfinance to attempt bad row repair.

    Returns:
        DataFrame indexed by timezone-aware DatetimeIndex (UTC) with columns
        ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase, sorted).
    """
    ticker = yf.Ticker(_XAUUSD_YF_SYMBOL)
    raw: pd.DataFrame = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        repair=repair,
    )
    if raw.empty:
        raise RuntimeError(
            f"No data returned for {_XAUUSD_YF_SYMBOL}. "
            "Check connectivity or try symbol GC=F as an alternative."
        )

    frame = raw.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    expected = {"open", "high", "low", "close", "volume"}
    missing = expected - set(frame.columns)
    if missing:
        raise RuntimeError(f"Downloaded data missing columns: {sorted(missing)}")

    frame = frame[sorted(expected)]
    frame = frame.sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "date"
    return frame


def fetch_realtime(api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch the latest XAU/USD exchange rate from Alpha Vantage (free tier).

    Uses ``CURRENCY_EXCHANGE_RATE`` for ``from_currency=XAU``, ``to_currency=USD``.
    The API key is read from the ``api_key`` argument or ``ALPHA_VANTAGE_API_KEY``.

    Args:
        api_key: Optional Alpha Vantage API key. Falls back to env var.

    Returns:
        Single-row DataFrame with UTC DatetimeIndex and columns
        ``open``, ``high``, ``low``, ``close``, ``volume``. Intraday OHLC is
        approximated from the quoted rate when only one price is provided.
    """
    key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise ValueError(
            "Alpha Vantage API key required: pass api_key= or set ALPHA_VANTAGE_API_KEY."
        )

    params: dict[str, str] = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "XAU",
        "to_currency": "USD",
        "apikey": key,
    }
    response: requests.Response = requests.get(_AV_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()

    if "Realtime Currency Exchange Rate" not in payload:
        note = payload.get("Note") or payload.get("Information") or payload
        raise RuntimeError(f"Unexpected Alpha Vantage response: {note}")

    bucket: dict[str, Any] = payload["Realtime Currency Exchange Rate"]
    rate_raw = bucket.get("5. Exchange Rate")
    if rate_raw is None:
        for k, v in bucket.items():
            if "Exchange Rate" in str(k):
                rate_raw = v
                break
    if rate_raw is None:
        raise RuntimeError(f"Could not parse exchange rate from: {bucket}")

    rate = float(rate_raw)
    ts_raw = bucket.get("6. Last Refreshed") or bucket.get("7. Last Refreshed")
    if ts_raw:
        ts = pd.to_datetime(ts_raw, utc=True)
    else:
        ts = pd.Timestamp.now(tz=timezone.utc)

    idx = pd.DatetimeIndex([ts], name="date")

    # Spot quote: synthesize flat OHLC; volume unknown for FX quote.
    row = pd.DataFrame(
        {
            "open": [rate],
            "high": [rate],
            "low": [rate],
            "close": [rate],
            "volume": [0.0],
        },
        index=idx,
    )
    return row


__all__ = ["fetch_realtime", "fetch_xauusd_history"]
