"""Append technical indicators to OHLCV frames using pandas-ta."""

from __future__ import annotations

import pandas as pd

from indicators.ta_compat import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``df`` with RSI, EMAs, SMA, MACD, and Bollinger columns.

    Required input columns: ``open``, ``high``, ``low``, ``close``, ``volume``
    (matched case-insensitively).

    Appended columns (when computable):

    - ``rsi``: RSI(14)
    - ``ema20``, ``ema50``: EMAs
    - ``sma200``: SMA(200)
    - ``macd_line``, ``macd_signal``: MACD(12,26,9) line and signal
    - ``bb_upper``, ``bb_lower``: Bollinger Bands(20,2)

    Args:
        df: OHLCV DataFrame indexed by time (typically DatetimeIndex).

    Returns:
        DataFrame including original columns plus indicator columns.
    """
    cols_lower = {str(c).lower(): c for c in df.columns}
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in cols_lower]
    if missing:
        raise ValueError(
            f"add_all_indicators requires columns {list(required)}; missing {missing}"
        )

    out = df.copy()
    close = out[cols_lower["close"]]

    out["rsi"] = ta.rsi(close, length=14)
    out["ema20"] = ta.ema(close, length=20)
    out["ema50"] = ta.ema(close, length=50)
    out["sma200"] = ta.sma(close, length=200)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        out["macd_line"] = pd.NA
        out["macd_signal"] = pd.NA
    else:
        mcols = list(macd_df.columns)
        line_name = next((c for c in mcols if c.startswith("MACD_") and "MACDs" not in c and "MACDh" not in c), None)
        sig_name = next((c for c in mcols if c.startswith("MACDs_")), None)
        if line_name is None or sig_name is None:
            raise RuntimeError(f"Unexpected MACD columns: {mcols}")
        out["macd_line"] = macd_df[line_name]
        out["macd_signal"] = macd_df[sig_name]

    bb = ta.bbands(close, length=20, std=2)
    if bb is None or bb.empty:
        out["bb_upper"] = pd.NA
        out["bb_lower"] = pd.NA
    else:
        bcols = list(bb.columns)
        upper_name = next((c for c in bcols if c.startswith("BBU_")), None)
        lower_name = next((c for c in bcols if c.startswith("BBL_")), None)
        if upper_name is None or lower_name is None:
            raise RuntimeError(f"Unexpected Bollinger columns: {bcols}")
        out["bb_upper"] = bb[upper_name]
        out["bb_lower"] = bb[lower_name]

    return out


__all__ = ["add_all_indicators"]
