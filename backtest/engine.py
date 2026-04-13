"""
Full-featured backtesting engine for AURUM Gold (XAUUSD) strategy evaluation.

Implements stop-loss / take-profit trade management with overlapping trade
support, vectorised signal generation, and comprehensive performance metrics.
Designed for academic review — no external backtest libraries, pure pandas + Python.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Opt-in to future pandas behaviour to suppress downcasting FutureWarnings
pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gold (XAUUSD): 1 pip = $0.01 — standard forex convention for metals
PIP_VALUE: float = 0.01


# ---------------------------------------------------------------------------
# Trade record dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """
    Immutable record of a single completed trade.

    Stores all entry/exit details, direction, result, and PnL in both
    pips and percentage terms for downstream analytics.
    """

    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    sl_price: float
    target_price: float
    direction: str          # "BUY" or "SELL"
    result: str             # "WIN" or "LOSS"
    pnl_pips: float         # signed profit/loss in pips
    pnl_pct: float          # signed profit/loss as percentage of entry

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON-safe output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Backtesting engine for XAUUSD multi-condition trading strategies.

    Accepts parsed conditions from ``StrategyParser``, applies vectorised signal
    generation, then runs a sequential trade-management loop with SL/TP exits.
    Overlapping trades are fully supported — each signal opens an independent
    position tracked separately.

    Key design decisions:
    - Entry on NEXT candle's open after signal candle closes (no lookahead).
    - Conservative same-candle exit: stop-loss takes priority over target.
    - ATR fallback computed from rolling True Range if column is missing.

    Attributes:
        PIP_VALUE: Dollar value of one pip for XAUUSD ($0.01).
    """

    PIP_VALUE: float = PIP_VALUE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        conditions: List[dict],
        stop_loss_config: dict,
        risk_reward_ratio: Optional[float] = None,
        target_config: Optional[dict] = None,
        timeframe: str = "1h",
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a full backtest over the provided OHLCV + indicators DataFrame.

        Signal generation is vectorised (boolean masks combined with AND logic).
        Trade management iterates row-by-row — unavoidable because each candle
        must check every open position for SL/TP hits.

        Args:
            df: OHLCV DataFrame with indicator columns and a DatetimeIndex.
                Expected columns: open, high, low, close, volume, RSI_14,
                EMA_9, EMA_20, EMA_50, SMA_200, MACD_12_26_9, MACDs_12_26_9,
                MACDh_12_26_9, BBU_20_2.0, BBL_20_2.0, ATRr_14.
            conditions: List of condition dicts from StrategyParser.
                Each must have 'indicator', 'operator', 'action', and
                optionally 'value', 'fast', 'slow'.
            stop_loss_config: Stop-loss specification, one of:
                {"type": "pips",       "value": 150}
                {"type": "percentage", "value": 1.5}
                {"type": "atr",        "multiplier": 1.5}
            risk_reward_ratio: Target = RR × stop distance. Mutually exclusive
                with target_config.
            target_config: Fixed pip target, e.g. {"type": "fixed", "value": 300}.
                Ignored if risk_reward_ratio is set.
            timeframe: Informational label for the data timeframe (default "1h").
            date_range: Optional (start, end) ISO date strings to trim the data.

        Returns:
            Dict with performance metrics, equity curve, and full trade log.

        Raises:
            ValueError: If conditions list is empty or data is insufficient.
        """
        # --- Input validation ---
        if not conditions:
            raise ValueError("No conditions provided")

        frame = df.copy()
        warnings_list: List[str] = []

        # --- Date range filtering ---
        frame, date_warnings = self._apply_date_range(frame, date_range)
        warnings_list.extend(date_warnings)

        # Minimum data guard — need enough candles for reliable indicator values
        if len(frame) < 100:
            raise ValueError(
                f"Insufficient data: only {len(frame)} rows after date filtering "
                f"(minimum 100 required)"
            )

        # --- Ensure ATR column exists (fallback to manual True Range) ---
        frame = self._ensure_atr(frame)

        # --- Determine trade direction from conditions ---
        direction = self._infer_direction(conditions)

        # --- Build composite entry signal (vectorised) ---
        entry_signal = self._build_entry_signal(frame, conditions)

        # --- Shift signal forward: entry on NEXT candle's open ---
        # Signal on candle[i] → eligible to enter at candle[i+1]
        entry_signal_shifted = entry_signal.shift(1).fillna(False).astype(bool)

        # --- Run trade management loop ---
        trades = self._manage_trades(
            frame=frame,
            entry_signal=entry_signal_shifted,
            direction=direction,
            stop_loss_config=stop_loss_config,
            risk_reward_ratio=risk_reward_ratio,
            target_config=target_config,
        )

        # --- Compute and return metrics ---
        result = self._compute_metrics(trades, frame, warnings_list)
        return result

    # ------------------------------------------------------------------
    # Date range filtering
    # ------------------------------------------------------------------

    def _apply_date_range(
        self,
        df: pd.DataFrame,
        date_range: Optional[Tuple[str, str]],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Trim DataFrame to the requested date range, warning if bounds exceed data.

        Args:
            df: Full DataFrame with DatetimeIndex.
            date_range: Optional (start, end) tuple of ISO date strings.

        Returns:
            Tuple of (trimmed DataFrame, list of warning strings).
        """
        warnings: List[str] = []
        if date_range is None:
            return df, warnings

        start_str, end_str = date_range
        df_start = df.index.min()
        df_end = df.index.max()

        req_start = pd.Timestamp(start_str)
        req_end = pd.Timestamp(end_str)

        # Warn if requested range extends beyond available data
        if req_start < df_start:
            warnings.append(
                f"Requested start {start_str} is before data start "
                f"{df_start.strftime('%Y-%m-%d')}; trimmed to available range."
            )
        if req_end > df_end:
            warnings.append(
                f"Requested end {end_str} is after data end "
                f"{df_end.strftime('%Y-%m-%d')}; trimmed to available range."
            )

        filtered = df.loc[start_str:end_str]
        return filtered, warnings

    # ------------------------------------------------------------------
    # ATR fallback
    # ------------------------------------------------------------------

    def _ensure_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee an ATRr_14 column exists, computing from True Range if missing.

        The manual True Range formula avoids a pandas-ta dependency for this
        single indicator, keeping the engine self-contained.

        Args:
            df: Working DataFrame (modified in place and returned).

        Returns:
            DataFrame with ATRr_14 column guaranteed present.
        """
        if "ATRr_14" in df.columns:
            return df

        logger.info("ATRr_14 column missing — computing 14-period ATR from True Range.")
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = df["close"].astype(float).shift(1)

        # True Range: max of (H-L, |H-prevC|, |L-prevC|)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["ATRr_14"] = tr.rolling(window=14, min_periods=1).mean()
        return df

    # ------------------------------------------------------------------
    # Direction inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_direction(conditions: List[dict]) -> str:
        """
        Determine overall trade direction from the conditions list.

        Uses majority vote; ties default to BUY (retail trader convention).

        Args:
            conditions: Parsed condition dicts with 'action' keys.

        Returns:
            "BUY" or "SELL".
        """
        buy_count = sum(1 for c in conditions if c.get("action", "BUY").upper() == "BUY")
        sell_count = len(conditions) - buy_count
        return "SELL" if sell_count > buy_count else "BUY"

    # ------------------------------------------------------------------
    # Vectorised signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_signal_window(mask: pd.Series, window: int = 5) -> pd.Series:
        """
        Extend a point-in-time boolean signal to persist for *window* candles.

        Crossover events fire True on a single candle.  When AND-ed with
        threshold conditions (e.g. RSI < 35) in daily data, both rarely
        coincide on the same bar — producing zero trades.  This method
        keeps crossover signals True for the next *window* candles so
        threshold conditions have a realistic chance to overlap.

        Uses a rolling-max to avoid a Python loop over the entire Series.

        Args:
            mask: Boolean Series where True marks the original event.
            window: Number of candles to persist the signal (default 5).

        Returns:
            Boolean Series with windowed persistence applied.
        """
        # rolling(window).max() propagates any True in the last *window* bars
        numeric = mask.astype(int)
        windowed = numeric.rolling(window=window, min_periods=1).max()
        return windowed.fillna(0).astype(bool)

    @staticmethod
    def _infer_window_size(df: pd.DataFrame) -> int:
        """
        Detect data frequency from index spacing and return an appropriate
        crossover window size.

        Daily data → 15 bars (~3 trading weeks) because EMA crossovers on
        daily charts are macro events that rarely align with threshold
        conditions on the same bar.

        Intraday (≤ 4h) → 5 bars (tight window for faster signals).

        Args:
            df: DataFrame with DatetimeIndex.

        Returns:
            Integer window size (bars).
        """
        if len(df) < 3:
            return 5

        # Median time delta between bars (robust to gaps like weekends)
        deltas = pd.Series(df.index).diff().dropna()
        median_hours = deltas.median().total_seconds() / 3600.0

        if median_hours >= 20:  # ~daily (24h minus weekend compression)
            return 30
        elif median_hours >= 3:  # 4h bars
            return 10
        else:  # 1h or smaller
            return 5

    def _build_entry_signal(
        self,
        df: pd.DataFrame,
        conditions: List[dict],
    ) -> pd.Series:
        """
        Combine all conditions into a single boolean entry signal via AND logic.

        Each condition generates its own boolean Series; the final signal is
        True only where ALL conditions are simultaneously satisfied.
        Crossover conditions are automatically windowed (adaptive to timeframe)
        so they can realistically overlap with threshold conditions.

        Args:
            df: Working DataFrame with indicator columns.
            conditions: Parsed condition dicts.

        Returns:
            Boolean Series aligned with df.index.
        """
        composite = pd.Series(True, index=df.index)
        window_size = self._infer_window_size(df)

        for cond in conditions:
            mask = self._condition_to_mask(cond, df)
            # BUG-002 fix: crossover signals are point-in-time events that
            # almost never coincide with threshold conditions on the same
            # candle.  Persist them using an adaptive window so AND-logic
            # can fire across different timeframes.
            op = str(cond.get("operator", ""))
            if "crossover" in op:
                mask = self._apply_signal_window(mask, window=window_size)
            composite = composite & mask

        return composite

    def _condition_to_mask(self, cond: dict, df: pd.DataFrame) -> pd.Series:
        """
        Convert a single parsed condition dict into a boolean mask.

        Handles RSI, EMA (price-vs and crossover), SMA (price-vs and crossover),
        MACD (line-vs-signal crossover), and Bollinger Bands (price-vs-band).

        Args:
            cond: Single condition dict with indicator, operator, value/fast/slow.
            df: Working DataFrame.

        Returns:
            Boolean Series; defaults to all-True for unrecognised indicators
            so it doesn't accidentally block valid signals.
        """
        indicator = str(cond.get("indicator", "")).upper()
        op = str(cond.get("operator", ""))
        close = df["close"].astype(float)
        default_true = pd.Series(True, index=df.index)

        try:
            # --- RSI ---
            if indicator == "RSI":
                rsi = self._get_column(df, "RSI_14", fallback_name="rsi")
                value = float(cond.get("value", 30))
                if op in ("<", "<="):
                    return rsi < value
                if op in (">", ">="):
                    return rsi > value
                return default_true

            # --- EMA ---
            if indicator == "EMA":
                if "crossover" in op:
                    fast_p = int(cond.get("fast", 20))
                    slow_p = int(cond.get("slow", 50))
                    fast_ema = self._get_column(df, f"EMA_{fast_p}", fallback_name=f"ema{fast_p}")
                    slow_ema = self._get_column(df, f"EMA_{slow_p}", fallback_name=f"ema{slow_p}")

                    if "above" in op:
                        # Crossover: fast was below slow, now above
                        prev_below = fast_ema.shift(1) <= slow_ema.shift(1)
                        curr_above = fast_ema > slow_ema
                        return prev_below & curr_above
                    else:
                        # Crossunder: fast was above slow, now below
                        prev_above = fast_ema.shift(1) >= slow_ema.shift(1)
                        curr_below = fast_ema < slow_ema
                        return prev_above & curr_below

                # Price vs single EMA
                period = int(cond.get("value", 20))
                ema = self._get_column(df, f"EMA_{period}", fallback_name=f"ema{period}")
                if op in (">", ">=", "above"):
                    return close > ema
                return close < ema

            # --- SMA ---
            if indicator == "SMA":
                if "crossover" in op:
                    fast_p = int(cond.get("fast", 50))
                    slow_p = int(cond.get("slow", 200))
                    fast_sma = self._get_column(df, f"SMA_{fast_p}", fallback_name=f"sma{fast_p}")
                    slow_sma = self._get_column(df, f"SMA_{slow_p}", fallback_name=f"sma{slow_p}")

                    if "above" in op:
                        prev_below = fast_sma.shift(1) <= slow_sma.shift(1)
                        curr_above = fast_sma > slow_sma
                        return prev_below & curr_above
                    else:
                        prev_above = fast_sma.shift(1) >= slow_sma.shift(1)
                        curr_below = fast_sma < slow_sma
                        return prev_above & curr_below

                period = int(cond.get("value", 200))
                sma = self._get_column(df, f"SMA_{period}", fallback_name=f"sma{period}")
                if op in (">", ">=", "above"):
                    return close > sma
                return close < sma

            # --- MACD ---
            if indicator == "MACD":
                macd_line = self._get_column(df, "MACD_12_26_9", fallback_name="macd_line")
                macd_signal = self._get_column(df, "MACDs_12_26_9", fallback_name="macd_signal")

                if "above" in op:
                    prev_below = macd_line.shift(1) <= macd_signal.shift(1)
                    curr_above = macd_line > macd_signal
                    return prev_below & curr_above
                else:
                    prev_above = macd_line.shift(1) >= macd_signal.shift(1)
                    curr_below = macd_line < macd_signal
                    return prev_above & curr_below

            # --- Bollinger Bands ---
            if indicator in ("BB", "BOLLINGER"):
                if op in ("<", "<="):
                    bb_lower = self._get_column(df, "BBL_20_2.0", fallback_name="bb_lower")
                    return close <= bb_lower
                else:
                    bb_upper = self._get_column(df, "BBU_20_2.0", fallback_name="bb_upper")
                    return close >= bb_upper

        except KeyError as exc:
            logger.warning("Missing column for condition %s: %s — defaulting to True", cond, exc)

        return default_true

    @staticmethod
    def _get_column(df: pd.DataFrame, primary: str, fallback_name: str = "") -> pd.Series:
        """
        Retrieve a column by primary name, falling back to an alternate name.

        Raises KeyError if neither exists so the caller can handle gracefully.

        Args:
            df: Working DataFrame.
            primary: Preferred column name (pandas-ta convention).
            fallback_name: Alternative column name (calculator.py convention).

        Returns:
            The matched column as a float Series.
        """
        if primary in df.columns:
            return df[primary].astype(float)
        if fallback_name and fallback_name in df.columns:
            return df[fallback_name].astype(float)
        raise KeyError(f"Column '{primary}' (or fallback '{fallback_name}') not found in DataFrame")

    # ------------------------------------------------------------------
    # Trade management loop
    # ------------------------------------------------------------------

    def _manage_trades(
        self,
        frame: pd.DataFrame,
        entry_signal: pd.Series,
        direction: str,
        stop_loss_config: dict,
        risk_reward_ratio: Optional[float],
        target_config: Optional[dict],
    ) -> List[TradeRecord]:
        """
        Iterate candle-by-candle managing open trades and opening new ones.

        This is the core trade simulation loop. It must be sequential because
        each candle's high/low must be checked against every open position's
        SL and TP levels.

        Overlapping trades are allowed — each signal opens an independent trade
        even if previous trades are still open.

        Args:
            frame: Working DataFrame with OHLCV + ATRr_14.
            entry_signal: Shifted boolean signal (True = enter at this candle's open).
            direction: "BUY" or "SELL".
            stop_loss_config: SL specification.
            risk_reward_ratio: Optional RR multiplier for target calculation.
            target_config: Optional fixed-pip target config.

        Returns:
            List of completed TradeRecord objects.
        """
        completed_trades: List[TradeRecord] = []
        open_trades: List[dict] = []  # dicts with entry info + sl/tp prices

        sl_type = stop_loss_config.get("type", "percentage")

        for i in range(len(frame)):
            row = frame.iloc[i]
            candle_high = float(row["high"])
            candle_low = float(row["low"])
            candle_open = float(row["open"])
            candle_time = str(frame.index[i])

            # --- Check existing open trades for SL/TP hits ---
            still_open: List[dict] = []
            for trade in open_trades:
                hit = self._check_exit(
                    trade, candle_high, candle_low, candle_time, direction
                )
                if hit is not None:
                    completed_trades.append(hit)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # --- Check for new entry on this candle ---
            if bool(entry_signal.iloc[i]):
                entry_price = candle_open

                # Avoid degenerate entries on zero-price candles
                if entry_price <= 0:
                    continue

                atr_value = float(row.get("ATRr_14", 0.0))

                sl_price = self._calc_sl(
                    entry_price, direction, sl_type, stop_loss_config, atr_value
                )
                tp_price = self._calc_tp(
                    entry_price, sl_price, direction,
                    risk_reward_ratio, target_config
                )

                open_trades.append({
                    "entry_time": candle_time,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "target_price": tp_price,
                })

        # Trades still open at end of data are NOT force-closed — per spec,
        # only SL/TP exits are valid. They simply don't appear in the log.

        return completed_trades

    def _check_exit(
        self,
        trade: dict,
        candle_high: float,
        candle_low: float,
        candle_time: str,
        direction: str,
    ) -> Optional[TradeRecord]:
        """
        Test whether a candle's high/low hits a trade's SL or TP.

        If both SL and TP are hit on the same candle, the conservative
        assumption is that SL was hit first (worst-case scenario).

        Args:
            trade: Open trade dict with entry info and SL/TP levels.
            candle_high: Current candle's high price.
            candle_low: Current candle's low price.
            candle_time: ISO timestamp of the current candle.
            direction: "BUY" or "SELL".

        Returns:
            Completed TradeRecord if trade exited, None if still open.
        """
        entry_price = trade["entry_price"]
        sl_price = trade["sl_price"]
        tp_price = trade["target_price"]

        if direction == "BUY":
            sl_hit = candle_low <= sl_price
            tp_hit = candle_high >= tp_price
        else:
            # SELL: SL is above entry, TP is below entry
            sl_hit = candle_high >= sl_price
            tp_hit = candle_low <= tp_price

        if not sl_hit and not tp_hit:
            return None

        # Conservative: if both hit on same candle, SL takes priority
        if sl_hit:
            exit_price = sl_price
            result = "LOSS"
        else:
            exit_price = tp_price
            result = "WIN"

        # PnL calculations
        if direction == "BUY":
            pnl_pips = (exit_price - entry_price) / self.PIP_VALUE
        else:
            pnl_pips = (entry_price - exit_price) / self.PIP_VALUE

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        if direction == "SELL":
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        return TradeRecord(
            entry_time=trade["entry_time"],
            exit_time=candle_time,
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=sl_price,
            target_price=tp_price,
            direction=direction,
            result=result,
            pnl_pips=round(pnl_pips, 2),
            pnl_pct=round(pnl_pct, 4),
        )

    # ------------------------------------------------------------------
    # SL / TP calculation
    # ------------------------------------------------------------------

    def _calc_sl(
        self,
        entry_price: float,
        direction: str,
        sl_type: str,
        config: dict,
        atr_value: float,
    ) -> float:
        """
        Compute the stop-loss price based on the chosen SL strategy.

        Args:
            entry_price: Trade entry price.
            direction: "BUY" or "SELL".
            sl_type: One of "pips", "percentage", "atr".
            config: Full SL config dict with type-specific keys.
            atr_value: Current ATRr_14 value at entry candle.

        Returns:
            Absolute stop-loss price level.
        """
        if sl_type == "pips":
            sl_distance = self.PIP_VALUE * float(config["value"])
        elif sl_type == "percentage":
            sl_distance = entry_price * (float(config["value"]) / 100.0)
        elif sl_type == "atr":
            multiplier = float(config.get("multiplier", 1.5))
            sl_distance = atr_value * multiplier
        else:
            # Fallback: 1% of entry as a safe default
            sl_distance = entry_price * 0.01

        if direction == "BUY":
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def _calc_tp(
        self,
        entry_price: float,
        sl_price: float,
        direction: str,
        risk_reward_ratio: Optional[float],
        target_config: Optional[dict],
    ) -> float:
        """
        Compute the take-profit price from RR ratio or fixed-pip config.

        Risk-reward ratio takes precedence over fixed-pip if both are provided.

        Args:
            entry_price: Trade entry price.
            sl_price: Already-computed stop-loss price.
            direction: "BUY" or "SELL".
            risk_reward_ratio: Optional multiplier (target = RR × stop distance).
            target_config: Optional {"type": "fixed", "value": <pips>}.

        Returns:
            Absolute take-profit price level.
        """
        if risk_reward_ratio is not None:
            stop_distance = abs(entry_price - sl_price)
            tp_distance = stop_distance * risk_reward_ratio
        elif target_config is not None and target_config.get("type") == "fixed":
            tp_distance = self.PIP_VALUE * float(target_config["value"])
        else:
            # Fallback: 2:1 RR when nothing else specified
            stop_distance = abs(entry_price - sl_price)
            tp_distance = stop_distance * 2.0

        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        trades: List[TradeRecord],
        frame: pd.DataFrame,
        warnings_list: List[str],
    ) -> Dict[str, Any]:
        """
        Aggregate all individual trade results into a comprehensive metrics dict.

        Computes accuracy, PnL, drawdown, Sharpe ratio, profit factor, streaks,
        and the cumulative equity curve.

        Args:
            trades: List of completed TradeRecord objects.
            frame: Working DataFrame (for backtest period metadata).
            warnings_list: Any warnings accumulated during processing.

        Returns:
            Complete results dict matching the specified output schema.
        """
        total = len(trades)

        # --- Backtest period metadata ---
        period = {
            "start": str(frame.index[0]),
            "end": str(frame.index[-1]),
            "total_candles": len(frame),
        }

        # --- Zero-trade early return ---
        if total == 0:
            result: Dict[str, Any] = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "accuracy": 0.0,
                "net_pnl_pips": 0.0,
                "net_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_win_pips": 0.0,
                "avg_loss_pips": 0.0,
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "current_streak": {"type": "WIN", "count": 0},
                "equity_curve": [],
                "trade_log": [],
                "backtest_period": period,
                "warning": (
                    "No signals generated — conditions may be too strict "
                    "for this data range."
                ),
            }
            if warnings_list:
                result["warnings"] = warnings_list
            return result

        # --- Win/Loss counts ---
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = total - wins
        accuracy = round((wins / total) * 100, 2)

        # --- PnL aggregation ---
        pnl_pips_list = [t.pnl_pips for t in trades]
        pnl_pct_list = [t.pnl_pct for t in trades]
        net_pnl_pips = round(sum(pnl_pips_list), 2)
        net_pnl_pct = round(sum(pnl_pct_list), 4)

        # --- Avg win / avg loss ---
        win_pips = [t.pnl_pips for t in trades if t.result == "WIN"]
        loss_pips = [t.pnl_pips for t in trades if t.result == "LOSS"]
        avg_win = round(sum(win_pips) / len(win_pips), 2) if win_pips else 0.0
        avg_loss = round(sum(loss_pips) / len(loss_pips), 2) if loss_pips else 0.0

        # --- Profit factor: gross_profit / |gross_loss| ---
        gross_profit = sum(p for p in pnl_pips_list if p > 0)
        gross_loss = sum(p for p in pnl_pips_list if p < 0)
        if gross_loss != 0:
            profit_factor = round(gross_profit / abs(gross_loss), 2)
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        # --- Equity curve (cumulative PnL starting from 0) ---
        equity_curve: List[float] = []
        cumulative = 0.0
        for t in trades:
            cumulative += t.pnl_pips
            equity_curve.append(round(cumulative, 2))

        # --- Max drawdown on pip-based equity curve ---
        max_drawdown_pct = self._calc_max_drawdown(equity_curve)

        # --- Sharpe ratio (annualised using daily PnL, risk_free = 0) ---
        sharpe = self._calc_sharpe(trades)

        # --- Streak analysis ---
        max_win_streak, max_loss_streak, current_streak = self._calc_streaks(trades)

        # --- Trade log serialisation ---
        trade_log = [t.to_dict() for t in trades]

        result = {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "accuracy": accuracy,
            "net_pnl_pips": net_pnl_pips,
            "net_pnl_pct": net_pnl_pct,
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "avg_win_pips": avg_win,
            "avg_loss_pips": avg_loss,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_streak": current_streak,
            "equity_curve": equity_curve,
            "trade_log": trade_log,
            "backtest_period": period,
        }

        if warnings_list:
            result["warnings"] = warnings_list

        return result

    @staticmethod
    def _calc_max_drawdown(equity_curve: List[float]) -> float:
        """
        Compute maximum peak-to-trough drawdown on the cumulative PnL curve.

        Uses percentage-based drawdown relative to peak value, treating the
        equity as starting capital of 10,000 + cumulative pip PnL.

        Args:
            equity_curve: List of cumulative PnL values (in pips).

        Returns:
            Maximum drawdown as a positive percentage.
        """
        if not equity_curve:
            return 0.0

        # Convert pip PnL to equity values (starting from a base)
        base = 10000.0
        equity_values = [base + e for e in equity_curve]

        peak = equity_values[0]
        max_dd = 0.0

        for val in equity_values:
            if val > peak:
                peak = val
            if peak > 0:
                dd = (peak - val) / peak * 100
                max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _calc_sharpe(trades: List[TradeRecord]) -> float:
        """
        Annualised Sharpe ratio using per-trade PnL percentage, risk-free = 0.

        Groups trades by calendar day and aggregates daily returns for the
        Sharpe calculation, then annualises with sqrt(252).

        Args:
            trades: Completed trade records.

        Returns:
            Annualised Sharpe ratio, rounded to 2 decimal places.
        """
        if len(trades) < 2:
            return 0.0

        # Group PnL by exit date for daily aggregation
        daily_pnl: Dict[str, float] = {}
        for t in trades:
            # Extract date portion from exit_time
            exit_date = t.exit_time[:10]
            daily_pnl[exit_date] = daily_pnl.get(exit_date, 0.0) + t.pnl_pct

        daily_returns = list(daily_pnl.values())

        if len(daily_returns) < 2:
            return 0.0

        arr = np.array(daily_returns)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))

        if std_r == 0:
            return 0.0

        sharpe = (mean_r / std_r) * math.sqrt(252)
        return round(sharpe, 2)

    @staticmethod
    def _calc_streaks(
        trades: List[TradeRecord],
    ) -> Tuple[int, int, dict]:
        """
        Compute maximum win/loss streaks and the current (latest) streak.

        Args:
            trades: Chronologically ordered trade records.

        Returns:
            Tuple of (max_win_streak, max_loss_streak, current_streak_dict).
        """
        if not trades:
            return 0, 0, {"type": "WIN", "count": 0}

        max_win = 0
        max_loss = 0
        current_type = trades[0].result
        current_count = 0

        for t in trades:
            if t.result == current_type:
                current_count += 1
            else:
                # Finalise the previous streak before switching
                if current_type == "WIN":
                    max_win = max(max_win, current_count)
                else:
                    max_loss = max(max_loss, current_count)
                current_type = t.result
                current_count = 1

        # Finalise the last streak after the loop
        if current_type == "WIN":
            max_win = max(max_win, current_count)
        else:
            max_loss = max(max_loss, current_count)

        return max_win, max_loss, {"type": current_type, "count": current_count}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke test: fetch 2 years of XAUUSD data from yfinance, compute RSI,
    run a simple RSI < 30 BUY backtest with 1.5% SL and 2:1 RR.
    """
    import json
    import yfinance as yf

    print("=" * 60)
    print("AURUM BacktestEngine — Smoke Test")
    print("=" * 60)

    # Fetch ~2 years of hourly Gold data (yfinance caps at 730 days for 1h)
    ticker = yf.Ticker("GC=F")
    raw = ticker.history(period="2y", interval="1h")

    if raw.empty:
        print("ERROR: yfinance returned no data for GC=F. Check network or ticker.")
        exit(1)

    # Normalise column names to lowercase
    raw.columns = [c.lower() for c in raw.columns]
    print(f"Fetched {len(raw)} candles from {raw.index[0]} to {raw.index[-1]}")

    # Compute RSI_14 using pandas rolling (avoid pandas-ta dependency in smoke test)
    delta = raw["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    raw["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    raw["RSI_14"] = raw["RSI_14"].fillna(50.0)

    # Compute ATRr_14 for the SL calculation
    high = raw["high"].astype(float)
    low = raw["low"].astype(float)
    prev_close = raw["close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    raw["ATRr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # Define a simple strategy: BUY when RSI < 30
    conditions = [
        {"indicator": "RSI", "operator": "<", "value": 30, "action": "BUY"},
    ]

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(
        df=raw,
        conditions=conditions,
        stop_loss_config={"type": "percentage", "value": 1.5},
        risk_reward_ratio=2.0,
    )

    # Pretty-print results (exclude trade_log for readability)
    display = {k: v for k, v in results.items() if k != "trade_log"}
    display["trade_log_count"] = len(results.get("trade_log", []))
    print("\n" + json.dumps(display, indent=2, default=str))

    # Show first 3 trades if any
    trade_log = results.get("trade_log", [])
    if trade_log:
        print(f"\nFirst {min(3, len(trade_log))} trades:")
        for t in trade_log[:3]:
            print(f"  {t['entry_time'][:19]} -> {t['exit_time'][:19]}  "
                  f"{t['direction']} {t['result']}  "
                  f"PnL: {t['pnl_pips']:+.2f} pips ({t['pnl_pct']:+.4f}%)")

    print("\n[OK] Smoke test complete.")


__all__ = ["BacktestEngine", "TradeRecord", "PIP_VALUE"]
