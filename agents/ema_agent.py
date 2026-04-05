"""EMA-focused agent with trend, crossover, and slope scoring."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from agents.base_agent import AgentResult, BaseAgent
from indicators.ta_compat import ta


class EMAAgent(BaseAgent):
    """
    Evaluates EMA-based conditions: price-vs-EMA, EMA crossovers, and slope analysis.

    Scoring model (100 points):

    - Up to 40 points from crossover/position rules on the latest bar.
    - Up to 40 points from ``calculate_win_rate(condition, df) * 40``.
    - Up to 20 points from slope-based trend strength assessment.
    """

    _FORWARD_HORIZON: int = 5
    _MOVE_PCT: float = 0.005
    _SLOPE_LOOKBACK: int = 5

    # Well-known EMA crossover combos used for domain-rule scoring
    _CROSSOVER_SCORES: dict[tuple[int, int], float] = {
        (9, 21): 30.0,
        (20, 50): 38.0,
        (50, 200): 40.0,
        (20, 200): 36.0,
    }

    def _ensure_ema(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Guarantee an ``ema_{period}`` column exists in *df*, computing it if absent.

        Args:
            df: Input OHLCV frame.
            period: Lookback length for the EMA.

        Returns:
            Copy of *df* with the required column populated.
        """
        frame = df.copy()
        col = f"ema_{period}"
        if col in frame.columns:
            return frame
        if "close" not in frame.columns:
            raise ValueError("DataFrame must include a 'close' column to compute EMA.")
        frame[col] = ta.ema(frame["close"], length=period)
        return frame

    # ------------------------------------------------------------------
    # Win-rate estimation
    # ------------------------------------------------------------------

    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Fraction of bars matching the EMA condition that saw a 0.5 % forward move.

        Args:
            condition: Rule dict with ``indicator``, ``operator``, ``value``, ``action``.
            df: Historical OHLCV data.

        Returns:
            Success fraction in ``[0.0, 1.0]``, or ``0.0`` when no setups found.
        """
        if str(condition.get("indicator", "")).upper() != "EMA":
            raise ValueError("EMAAgent.calculate_win_rate expects indicator 'EMA'.")

        period = int(condition.get("value", 20))
        frame = self._ensure_ema(df, period)
        col = f"ema_{period}"
        close = frame["close"].astype(float)
        ema = frame[col].astype(float)
        action = str(condition["action"]).upper()
        op = str(condition.get("operator", ">"))

        # Build boolean mask based on operator
        if op in (">", ">=", "above"):
            mask = close > ema
        elif op in ("<", "<=", "below"):
            mask = close < ema
        else:
            mask = close > ema  # default: price above EMA

        successes, total = 0, 0
        horizon = self._FORWARD_HORIZON
        threshold = self._MOVE_PCT

        for i in range(len(frame) - horizon):
            if pd.isna(ema.iloc[i]) or not bool(mask.iloc[i]):
                continue
            c0 = float(close.iloc[i])
            c_h = float(close.iloc[i + horizon])
            if pd.isna(c0) or pd.isna(c_h) or c0 == 0.0:
                continue
            ret = (c_h - c0) / c0
            total += 1
            if action == "BUY" and ret > threshold:
                successes += 1
            elif action == "SELL" and ret < -threshold:
                successes += 1

        return successes / total if total else 0.0

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _rule_points(self, action: str, close_val: float, ema_val: float) -> float:
        """
        Award up to 40 rule-based points depending on price position relative to EMA.

        Args:
            action: ``BUY`` or ``SELL``.
            close_val: Latest close price.
            ema_val: Latest EMA value.

        Returns:
            Points in ``[0, 40]``.
        """
        if pd.isna(close_val) or pd.isna(ema_val) or ema_val == 0.0:
            return 0.0
        pct_diff = (close_val - ema_val) / ema_val
        act = action.upper()

        if act == "BUY":
            # Price above EMA → stronger buy alignment
            if pct_diff > 0.02:
                return 40.0
            if pct_diff > 0.005:
                return 30.0
            if pct_diff > 0.0:
                return 20.0
            return 5.0
        if act == "SELL":
            # Price below EMA → stronger sell alignment
            if pct_diff < -0.02:
                return 40.0
            if pct_diff < -0.005:
                return 30.0
            if pct_diff < 0.0:
                return 20.0
            return 5.0
        return 0.0

    def _slope_points(self, ema_series: pd.Series, action: str) -> float:
        """
        Award up to 20 points based on EMA slope direction and steepness.

        Args:
            ema_series: Full EMA column.
            action: ``BUY`` or ``SELL``.

        Returns:
            Slope component score in ``[0, 20]``.
        """
        tail = ema_series.dropna().tail(self._SLOPE_LOOKBACK)
        if len(tail) < 2:
            return 0.0
        # Average per-bar change as fraction of EMA level
        changes = tail.diff().dropna()
        avg_change = float(changes.mean())
        base = float(tail.iloc[0])
        if base == 0.0:
            return 0.0
        slope_pct = avg_change / base

        act = action.upper()
        if act == "BUY" and slope_pct > 0:
            return min(20.0, abs(slope_pct) * 4000)
        if act == "SELL" and slope_pct < 0:
            return min(20.0, abs(slope_pct) * 4000)
        return 0.0

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def generate_suggestions(self, score: float) -> list[str]:
        """
        Produce actionable follow-up suggestions based on the composite score.

        Args:
            score: Final composite score in ``[0, 100]``.

        Returns:
            Ordered list of recommendation strings.
        """
        s = max(0.0, min(100.0, float(score) if not pd.isna(score) else 0.0))

        if s >= 85.0:
            return [
                "EMA trend is strongly in your favour; consider full position sizing.",
                "Lock partials at the next significant EMA level.",
            ]
        if s >= 70.0:
            return [
                "Trend alignment is solid; confirm with volume or MACD.",
                "Place stop just past the EMA band for invalidation.",
            ]
        if s >= 55.0:
            return [
                "Moderate EMA signal; reduce size and wait for slope confirmation.",
                "Watch if price retests the EMA before adding exposure.",
            ]
        if s >= 40.0:
            return [
                "Weak EMA reading; seek confluence from RSI or Bollinger Bands.",
                "Avoid initiating a new position on EMA alone.",
            ]
        return [
            "No meaningful EMA edge detected; stay flat.",
            "Re-evaluate once EMA slope changes direction.",
        ]

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Produce a composite ``AgentResult`` for an EMA rule on current history.

        Args:
            condition: ``{"indicator": "EMA", "operator": ">", "value": 20, "action": "BUY"}``.
            df: OHLCV history; EMA is computed if absent.

        Returns:
            Populated ``AgentResult`` with score, win rate, and narrative feedback.
        """
        if str(condition.get("indicator", "")).upper() != "EMA":
            raise ValueError("EMAAgent.evaluate_condition expects indicator 'EMA'.")

        period = int(condition.get("value", 20))
        frame = self._ensure_ema(df, period)
        col = f"ema_{period}"

        if frame.empty:
            return AgentResult(
                agent_name="EMA",
                score=0.0,
                win_rate=0.0,
                feedback=["No rows available for EMA evaluation."],
                suggestions=self.generate_suggestions(0.0),
                action_alignment="NEUTRAL",
            )

        latest_close = float(frame["close"].iloc[-1])
        latest_ema = float(frame[col].iloc[-1])
        action = str(condition["action"]).upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("condition['action'] must be 'BUY' or 'SELL'.")

        rule = self._rule_points(action, latest_close, latest_ema)
        win_rate = float(self.calculate_win_rate(condition, frame))
        ml_component = 40.0 * win_rate
        slope = self._slope_points(frame[col], action)

        raw_total = rule + ml_component + slope
        score = self._safe_score(raw_total)

        if score >= 55.0:
            alignment: Literal["BUY", "SELL", "NEUTRAL"] = action  # type: ignore[assignment]
        else:
            alignment = "NEUTRAL"

        pct_diff = ((latest_close - latest_ema) / latest_ema * 100) if latest_ema else 0.0
        feedback: list[str] = [
            f"EMA({period}) = {latest_ema:.2f}, Close = {latest_close:.2f} ({pct_diff:+.2f}%).",
            f"Rule-based credit {rule:.1f}/40.",
            f"Historical win-rate estimate = {win_rate:.2%} contributing {ml_component:.1f}/40.",
            f"Slope component = {slope:.1f}/20.",
        ]

        return AgentResult(
            agent_name="EMA",
            score=score,
            win_rate=win_rate,
            feedback=feedback,
            suggestions=self.generate_suggestions(score),
            action_alignment=alignment,
        )


__all__ = ["EMAAgent"]
