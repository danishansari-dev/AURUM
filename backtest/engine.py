"""Vectorised backtesting engine for multi-condition strategies on Gold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.ta_compat import ta


class BacktestEngine:
    """
    Simulate a strategy across historical OHLCV data using vectorised signal logic.

    All conditions are combined with AND logic: a trade fires only when **every**
    condition is satisfied on the same bar.  The engine tracks entries, exits,
    per-trade returns, and aggregate performance metrics.

    Attributes:
        INITIAL_CAPITAL: Starting USD balance for portfolio simulation.
        COMMISSION_PCT: Round-trip commission as a fraction (0.001 = 0.1 %).
        FORWARD_HOLD: Bars to hold a position before auto-exit.
    """

    INITIAL_CAPITAL: float = 10_000.0
    COMMISSION_PCT: float = 0.001
    FORWARD_HOLD: int = 5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        conditions: list[dict],
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Execute a full backtest of the given conditions on *df*.

        Args:
            conditions: Parsed condition dicts (each with ``indicator``,
                ``operator``, ``value``/``fast``/``slow``, ``action``).
            df: OHLCV DataFrame with lowercase columns and a DatetimeIndex.

        Returns:
            Dictionary of performance metrics::

                {
                    "total_return_pct": float,
                    "win_rate": float,         # percentage
                    "total_trades": int,
                    "sharpe_ratio": float,
                    "max_drawdown_pct": float,
                    "profit_factor": float,
                    "final_value": float,
                    "equity_curve": dict,      # {iso_date: equity_value}
                }
        """
        if df.empty or not conditions:
            return self._empty_result()

        frame = df.copy()
        close = frame["close"].astype(float)

        # --- Build composite signal mask (AND of all conditions) ---
        composite_mask = pd.Series(True, index=frame.index)
        action = "BUY"  # default direction

        for cond in conditions:
            mask = self._condition_mask(cond, frame)
            composite_mask &= mask
            if cond.get("action", "BUY").upper() == "SELL":
                action = "SELL"

        # --- Simulate trades ---
        trades = self._simulate_trades(composite_mask, close, action)

        if not trades:
            return self._empty_result()

        # --- Compute metrics ---
        returns = np.array([t["return_pct"] for t in trades])
        wins = int((returns > 0).sum())
        losses = int((returns <= 0).sum())
        gross_profit = float(returns[returns > 0].sum()) if wins else 0.0
        gross_loss = float(abs(returns[returns <= 0].sum())) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Build equity curve
        equity = self._build_equity_curve(trades, close)

        # Sharpe (annualised, assuming daily bars)
        trade_returns = pd.Series(returns)
        mean_r = trade_returns.mean()
        std_r = trade_returns.std()
        sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 0 else 0.0

        # Max drawdown from equity curve
        eq_series = pd.Series(equity)
        running_max = eq_series.cummax()
        drawdown = (eq_series - running_max) / running_max
        max_dd = float(drawdown.min() * 100)

        final_val = list(equity.values())[-1] if equity else self.INITIAL_CAPITAL
        total_return = (final_val - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL * 100

        return {
            "total_return_pct": round(total_return, 2),
            "win_rate": round(wins / len(trades) * 100, 1) if trades else 0.0,
            "total_trades": len(trades),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(abs(max_dd), 2),
            "profit_factor": round(profit_factor, 2),
            "final_value": round(final_val, 2),
            "equity_curve": equity,
        }

    # ------------------------------------------------------------------
    # Signal mask builders
    # ------------------------------------------------------------------

    def _condition_mask(self, cond: dict, df: pd.DataFrame) -> pd.Series:
        """
        Build a boolean mask for a single condition across all bars.

        Handles RSI, EMA, SMA, MACD, and BB conditions with their common
        operators and crossover patterns.

        Args:
            cond: Single parsed condition dict.
            df: OHLCV DataFrame.

        Returns:
            Boolean Series aligned with *df* index.
        """
        indicator = str(cond.get("indicator", "")).upper()
        op = str(cond.get("operator", ""))
        close = df["close"].astype(float)
        default = pd.Series(True, index=df.index)

        try:
            if indicator == "RSI":
                rsi = df["rsi"] if "rsi" in df.columns else ta.rsi(close, length=14)
                value = float(cond.get("value", 30))
                if op in ("<", "<="):
                    return rsi < value
                if op in (">", ">="):
                    return rsi > value
                return default

            if indicator == "EMA":
                if "crossover" in op:
                    fast_p = int(cond.get("fast", 20))
                    slow_p = int(cond.get("slow", 50))
                    fast_col = f"ema{fast_p}" if f"ema{fast_p}" in df.columns else None
                    slow_col = f"ema{slow_p}" if f"ema{slow_p}" in df.columns else None
                    fast_ema = df[fast_col] if fast_col else ta.ema(close, length=fast_p)
                    slow_ema = df[slow_col] if slow_col else ta.ema(close, length=slow_p)
                    if "above" in op:
                        return fast_ema > slow_ema
                    return fast_ema < slow_ema
                # Price vs EMA
                period = int(cond.get("value", 20))
                col = f"ema{period}" if f"ema{period}" in df.columns else None
                ema = df[col] if col else ta.ema(close, length=period)
                if op in (">", ">=", "above"):
                    return close > ema
                return close < ema

            if indicator == "SMA":
                period = int(cond.get("value", 50))
                col = f"sma{period}" if f"sma{period}" in df.columns else None
                sma = df[col] if col else ta.sma(close, length=period)
                if op in (">", ">=", "above"):
                    return close > sma
                return close < sma

            if indicator == "MACD":
                if "macd_line" in df.columns:
                    macd_line = df["macd_line"]
                    macd_sig = df["macd_signal"]
                else:
                    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                    cols = list(macd_df.columns)
                    macd_line = macd_df[cols[0]]
                    macd_sig = macd_df[cols[1]]
                if "above" in op:
                    return macd_line > macd_sig
                return macd_line < macd_sig

            if indicator in ("BB", "BOLLINGER"):
                if "bb_upper" in df.columns:
                    bb_upper = df["bb_upper"]
                    bb_lower = df["bb_lower"]
                else:
                    bb = ta.bbands(close, length=20, std=2)
                    bcols = list(bb.columns)
                    bb_lower = bb[bcols[0]]
                    bb_upper = bb[bcols[2]]
                if op in ("<", "<="):
                    return close <= bb_lower
                return close >= bb_upper

        except Exception:
            pass

        return default

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def _simulate_trades(
        self,
        mask: pd.Series,
        close: pd.Series,
        action: str,
    ) -> list[dict]:
        """
        Walk the signal mask and generate individual trade records.

        Positions are held for ``FORWARD_HOLD`` bars and then auto-exited.
        No overlapping trades — a new trade only opens after the current one closes.

        Args:
            mask: Boolean signal mask (True = entry bar).
            close: Close price series.
            action: ``"BUY"`` or ``"SELL"``.

        Returns:
            List of trade dicts with ``entry_idx``, ``exit_idx``, ``return_pct``.
        """
        trades: list[dict] = []
        i = 0
        n = len(close)
        indices = close.index.tolist()

        while i < n - self.FORWARD_HOLD:
            if bool(mask.iloc[i]):
                entry_price = float(close.iloc[i])
                exit_price = float(close.iloc[i + self.FORWARD_HOLD])
                if entry_price == 0:
                    i += 1
                    continue

                if action == "BUY":
                    ret = (exit_price - entry_price) / entry_price - self.COMMISSION_PCT
                else:
                    ret = (entry_price - exit_price) / entry_price - self.COMMISSION_PCT

                trades.append({
                    "entry_idx": indices[i],
                    "exit_idx": indices[i + self.FORWARD_HOLD],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret,
                })
                # Skip ahead past the hold period (no overlapping trades)
                i += self.FORWARD_HOLD
            else:
                i += 1

        return trades

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def _build_equity_curve(
        self,
        trades: list[dict],
        close: pd.Series,
    ) -> dict[str, float]:
        """
        Build a simplified equity curve from trade records.

        Between trades the equity stays flat; each trade adjusts the running
        balance by its percentage return.

        Args:
            trades: Trade records from ``_simulate_trades``.
            close: Close series for date context.

        Returns:
            ``{iso_date_string: equity_value}`` ordered chronologically.
        """
        equity: dict[str, float] = {}
        balance = self.INITIAL_CAPITAL

        # Start point
        first_date = close.index[0]
        equity[str(first_date.date()) if hasattr(first_date, "date") else str(first_date)] = balance

        for trade in trades:
            balance *= (1 + trade["return_pct"])
            dt = trade["exit_idx"]
            key = str(dt.date()) if hasattr(dt, "date") else str(dt)
            equity[key] = round(balance, 2)

        return equity

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return zeroed-out metrics when no trades can be generated."""
        return {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "final_value": 10_000.0,
            "equity_curve": {},
        }


__all__ = ["BacktestEngine"]
