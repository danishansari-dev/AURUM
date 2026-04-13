"""Append technical indicators to OHLCV frames using pandas-ta."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from indicators.ta_compat import ta


def _extract_required_periods(conditions: Optional[List[dict]] = None) -> dict:
    """
    Scan parsed conditions to discover which EMA/SMA periods are needed.

    Without this, only hardcoded EMA(20,50) and SMA(200) would be computed,
    causing conditions like "EMA9 crosses above EMA21" to silently fail
    (BUG-001 / BUG-009).

    Args:
        conditions: List of parsed condition dicts from the StrategyParser.

    Returns:
        Dict with keys 'ema_periods' and 'sma_periods' — sorted sets of ints.
    """
    ema_periods = {20, 50}  # always compute the baseline overlay periods
    sma_periods = {200}

    if not conditions:
        return {"ema_periods": sorted(ema_periods), "sma_periods": sorted(sma_periods)}

    for cond in conditions:
        indicator = str(cond.get("indicator", "")).upper()

        if indicator == "EMA":
            # Crossover conditions carry fast/slow period keys
            if cond.get("fast") is not None:
                ema_periods.add(int(cond["fast"]))
            if cond.get("slow") is not None:
                ema_periods.add(int(cond["slow"]))
            # Price-vs-EMA conditions carry the period in 'value'
            if cond.get("value") is not None and "crossover" not in str(cond.get("operator", "")):
                ema_periods.add(int(cond["value"]))

        elif indicator == "SMA":
            if cond.get("fast") is not None:
                sma_periods.add(int(cond["fast"]))
            if cond.get("slow") is not None:
                sma_periods.add(int(cond["slow"]))
            if cond.get("value") is not None and "crossover" not in str(cond.get("operator", "")):
                sma_periods.add(int(cond["value"]))

    return {"ema_periods": sorted(ema_periods), "sma_periods": sorted(sma_periods)}


def add_all_indicators(
    df: pd.DataFrame,
    conditions: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with RSI, EMAs, SMAs, MACD, and Bollinger columns.

    Required input columns: ``open``, ``high``, ``low``, ``close``, ``volume``
    (matched case-insensitively).

    Now accepts an optional ``conditions`` list so that EMA/SMA periods
    referenced in the user's strategy are computed dynamically — not just
    the hardcoded defaults (BUG-001 / BUG-009 fix).

    Appended columns (when computable):

    - ``rsi``: RSI(14)
    - ``ema{N}``: EMA for each period N found in conditions (always includes 20, 50)
    - ``sma{N}``: SMA for each period N found in conditions (always includes 200)
    - ``macd_line``, ``macd_signal``: MACD(12,26,9) line and signal
    - ``bb_upper``, ``bb_lower``: Bollinger Bands(20,2)

    Args:
        df: OHLCV DataFrame indexed by time (typically DatetimeIndex).
        conditions: Parsed condition dicts from StrategyParser (optional).

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

    # --- RSI (always 14-period) ---
    out["rsi"] = ta.rsi(close, length=14)

    # --- Dynamic EMA/SMA periods based on parsed conditions ---
    periods = _extract_required_periods(conditions)

    for p in periods["ema_periods"]:
        col_name = f"ema{p}"
        out[col_name] = ta.ema(close, length=p)

    for p in periods["sma_periods"]:
        col_name = f"sma{p}"
        out[col_name] = ta.sma(close, length=p)

    # --- MACD (12, 26, 9) ---
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

    # --- Bollinger Bands (20, 2) ---
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
