"""SMA-focused agent with trend context, crossover, and mean-reversion scoring."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from agents.base_agent import AgentResult, BaseAgent
from indicators.ta_compat import ta


class SMAAgent(BaseAgent):
    """
    Evaluates SMA-based conditions: price-vs-SMA position, crossovers, and mean reversion.

    Scoring model (100 points):

    - Up to 40 points from position rules (price above/below SMA).
    - Up to 40 points from ``calculate_win_rate(condition, df) * 40``.
    - Up to 20 points from mean-reversion proximity assessment.
    """

    _FORWARD_HORIZON: int = 5
    _MOVE_PCT: float = 0.005

    def _ensure_sma(self, df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """
        Guarantee an ``sma_{period}`` column exists in *df*, computing it if absent.

        Args:
            df: Input OHLCV frame.
            period: Lookback length for the SMA.

        Returns:
            Copy of *df* with the required column populated.
        """
        frame = df.copy()
        col = f"sma_{period}"
        if col in frame.columns:
            return frame
        if "close" not in frame.columns:
            raise ValueError("DataFrame must include a 'close' column to compute SMA.")
        frame[col] = ta.sma(frame["close"], length=period)
        return frame

    # ------------------------------------------------------------------
    # Win-rate estimation
    # ------------------------------------------------------------------

    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Fraction of bars matching the SMA condition that saw a 0.5 % forward move.

        Args:
            condition: Rule dict with ``indicator``, ``operator``, ``value``, ``action``.
            df: Historical OHLCV data.

        Returns:
            Success fraction in ``[0.0, 1.0]``, or ``0.0`` when no setups found.
        """
        if str(condition.get("indicator", "")).upper() != "SMA":
            raise ValueError("SMAAgent.calculate_win_rate expects indicator 'SMA'.")

        period = int(condition.get("value", 50))
        frame = self._ensure_sma(df, period)
        col = f"sma_{period}"
        close = frame["close"].astype(float)
        sma = frame[col].astype(float)
        action = str(condition["action"]).upper()
        op = str(condition.get("operator", ">"))

        if op in (">", ">=", "above"):
            mask = close > sma
        elif op in ("<", "<=", "below"):
            mask = close < sma
        else:
            mask = close > sma

        successes, total = 0, 0
        horizon = self._FORWARD_HORIZON
        threshold = self._MOVE_PCT

        for i in range(len(frame) - horizon):
            if pd.isna(sma.iloc[i]) or not bool(mask.iloc[i]):
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

    def _rule_points(self, action: str, close_val: float, sma_val: float) -> float:
        """
        Award up to 40 rule-based points based on price position relative to SMA.

        Args:
            action: ``BUY`` or ``SELL``.
            close_val: Latest close price.
            sma_val: Latest SMA value.

        Returns:
            Points in ``[0, 40]``.
        """
        if pd.isna(close_val) or pd.isna(sma_val) or sma_val == 0.0:
            return 0.0
        pct_diff = (close_val - sma_val) / sma_val
        act = action.upper()

        if act == "BUY":
            if pct_diff > 0.03:
                return 40.0
            if pct_diff > 0.01:
                return 30.0
            if pct_diff > 0.0:
                return 20.0
            return 5.0
        if act == "SELL":
            if pct_diff < -0.03:
                return 40.0
            if pct_diff < -0.01:
                return 30.0
            if pct_diff < 0.0:
                return 20.0
            return 5.0
        return 0.0

    def _reversion_points(self, close_val: float, sma_val: float, action: str) -> float:
        """
        Award up to 20 points for mean-reversion proximity.

        When price is stretched far from SMA, a snap-back becomes likely — this
        rewards conditions that align with the expected reversion direction.

        Args:
            close_val: Latest close price.
            sma_val: Latest SMA value.
            action: ``BUY`` or ``SELL``.

        Returns:
            Reversion component score in ``[0, 20]``.
        """
        if pd.isna(close_val) or pd.isna(sma_val) or sma_val == 0.0:
            return 0.0
        deviation = abs(close_val - sma_val) / sma_val
        act = action.upper()

        # Overextended below SMA → bullish reversion expected
        if act == "BUY" and close_val < sma_val and deviation > 0.02:
            return min(20.0, deviation * 500)
        # Overextended above SMA → bearish reversion expected
        if act == "SELL" and close_val > sma_val and deviation > 0.02:
            return min(20.0, deviation * 500)
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
                "SMA trend context is exceptionally strong; maintain position.",
                "Use SMA as a trailing stop reference.",
            ]
        if s >= 70.0:
            return [
                "Solid SMA alignment; confirm with EMA for timing precision.",
                "Set alerts at the SMA level for pullback entries.",
            ]
        if s >= 55.0:
            return [
                "Moderate SMA reading; consider combining with momentum indicators.",
                "Wait for price to close decisively above/below the SMA.",
            ]
        if s >= 40.0:
            return [
                "Weak SMA signal; use as context only, not primary trigger.",
                "Look for SMA crossover confirmation before acting.",
            ]
        return [
            "No meaningful SMA edge; the market lacks directional context.",
            "Re-evaluate on a longer SMA period for macro trend clarity.",
        ]

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Produce a composite ``AgentResult`` for an SMA rule on current history.

        Args:
            condition: ``{"indicator": "SMA", "operator": ">", "value": 50, "action": "BUY"}``.
            df: OHLCV history; SMA is computed if absent.

        Returns:
            Populated ``AgentResult`` with score, win rate, and narrative feedback.
        """
        if str(condition.get("indicator", "")).upper() != "SMA":
            raise ValueError("SMAAgent.evaluate_condition expects indicator 'SMA'.")

        period = int(condition.get("value", 50))
        frame = self._ensure_sma(df, period)
        col = f"sma_{period}"

        if frame.empty:
            return AgentResult(
                agent_name="SMA",
                score=0.0,
                win_rate=0.0,
                feedback=["No rows available for SMA evaluation."],
                suggestions=self.generate_suggestions(0.0),
                action_alignment="NEUTRAL",
            )

        latest_close = float(frame["close"].iloc[-1])
        latest_sma = float(frame[col].iloc[-1])
        action = str(condition["action"]).upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("condition['action'] must be 'BUY' or 'SELL'.")

        rule = self._rule_points(action, latest_close, latest_sma)
        win_rate = float(self.calculate_win_rate(condition, frame))
        ml_component = 40.0 * win_rate
        reversion = self._reversion_points(latest_close, latest_sma, action)

        raw_total = rule + ml_component + reversion
        score = self._safe_score(raw_total)

        if score >= 55.0:
            alignment: Literal["BUY", "SELL", "NEUTRAL"] = action  # type: ignore[assignment]
        else:
            alignment = "NEUTRAL"

        pct_diff = ((latest_close - latest_sma) / latest_sma * 100) if latest_sma else 0.0
        feedback: list[str] = [
            f"SMA({period}) = {latest_sma:.2f}, Close = {latest_close:.2f} ({pct_diff:+.2f}%).",
            f"Rule-based credit {rule:.1f}/40.",
            f"Historical win-rate estimate = {win_rate:.2%} contributing {ml_component:.1f}/40.",
            f"Mean-reversion component = {reversion:.1f}/20.",
        ]

        return AgentResult(
            agent_name="SMA",
            score=score,
            win_rate=win_rate,
            feedback=feedback,
            suggestions=self.generate_suggestions(score),
            action_alignment=alignment,
        )


__all__ = ["SMAAgent"]
