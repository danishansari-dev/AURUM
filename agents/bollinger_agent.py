"""Bollinger Bands agent with band-position, squeeze, and bandwidth scoring."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from agents.base_agent import AgentResult, BaseAgent
from indicators.ta_compat import ta


class BollingerAgent(BaseAgent):
    """
    Evaluates Bollinger-Band conditions: price position within bands, squeeze detection, and bandwidth.

    Scoring model (100 points):

    - Up to 40 points from band-position rules (price near upper/lower band).
    - Up to 40 points from ``calculate_win_rate(condition, df) * 40``.
    - Up to 20 points from Bollinger-squeeze detection (low bandwidth → breakout imminent).
    """

    _FORWARD_HORIZON: int = 5
    _MOVE_PCT: float = 0.005
    _BB_PERIOD: int = 20
    _BB_STD: float = 2.0

    def _ensure_bbands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee ``bb_upper``, ``bb_middle``, ``bb_lower``, and ``bb_bandwidth`` columns exist.

        Args:
            df: Input OHLCV frame.

        Returns:
            Copy of *df* with Bollinger Bands columns populated.
        """
        frame = df.copy()
        if {"bb_upper", "bb_middle", "bb_lower"}.issubset(frame.columns):
            return frame
        if "close" not in frame.columns:
            raise ValueError("DataFrame must include a 'close' column to compute Bollinger Bands.")
        bb = ta.bbands(frame["close"], length=self._BB_PERIOD, std=self._BB_STD)
        cols = list(bb.columns)
        # pandas_ta bbands returns: BBL, BBM, BBU, BBB, BBP (order may vary)
        frame["bb_lower"] = bb[cols[0]]
        frame["bb_middle"] = bb[cols[1]]
        frame["bb_upper"] = bb[cols[2]]
        if len(cols) > 3:
            frame["bb_bandwidth"] = bb[cols[3]]
        else:
            # Compute bandwidth manually when pandas_ta omits it
            frame["bb_bandwidth"] = (frame["bb_upper"] - frame["bb_lower"]) / frame["bb_middle"]
        return frame

    # ------------------------------------------------------------------
    # Win-rate estimation
    # ------------------------------------------------------------------

    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Fraction of bars matching the BB condition that saw a 0.5 % forward move.

        Args:
            condition: Rule dict with ``indicator``, ``operator``, ``value``, ``action``.
            df: Historical OHLCV data.

        Returns:
            Success fraction in ``[0.0, 1.0]``, or ``0.0`` when no setups found.
        """
        if str(condition.get("indicator", "")).upper() not in ("BB", "BOLLINGER"):
            raise ValueError("BollingerAgent.calculate_win_rate expects indicator 'BB'.")

        frame = self._ensure_bbands(df)
        close = frame["close"].astype(float)
        upper = frame["bb_upper"].astype(float)
        lower = frame["bb_lower"].astype(float)
        action = str(condition["action"]).upper()
        op = str(condition.get("operator", "<"))

        # Mask: price touching lower band (BUY signal) or upper band (SELL signal)
        if action == "BUY":
            mask = close <= lower
        elif action == "SELL":
            mask = close >= upper
        else:
            mask = close <= lower if op in ("<", "<=") else close >= upper

        successes, total = 0, 0
        horizon = self._FORWARD_HORIZON
        threshold = self._MOVE_PCT

        for i in range(len(frame) - horizon):
            if pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]) or not bool(mask.iloc[i]):
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

    def _rule_points(self, action: str, close_val: float, upper: float, lower: float, middle: float) -> float:
        """
        Award up to 40 rule-based points from band position.

        Args:
            action: ``BUY`` or ``SELL``.
            close_val: Latest close price.
            upper: Upper Bollinger Band value.
            lower: Lower Bollinger Band value.
            middle: Middle Bollinger Band value.

        Returns:
            Points in ``[0, 40]``.
        """
        if any(pd.isna(v) for v in (close_val, upper, lower, middle)):
            return 0.0

        band_width = upper - lower
        if band_width == 0.0:
            return 0.0

        # Normalised position: 0 = at lower band, 1 = at upper band
        position = (close_val - lower) / band_width
        act = action.upper()

        if act == "BUY":
            # Price near lower band = strong buy signal
            if position <= 0.0:
                return 40.0
            if position <= 0.15:
                return 32.0
            if position <= 0.3:
                return 20.0
            if position <= 0.5:
                return 10.0
            return 0.0
        if act == "SELL":
            # Price near upper band = strong sell signal
            if position >= 1.0:
                return 40.0
            if position >= 0.85:
                return 32.0
            if position >= 0.7:
                return 20.0
            if position >= 0.5:
                return 10.0
            return 0.0
        return 0.0

    def _squeeze_points(self, frame: pd.DataFrame) -> float:
        """
        Award up to 20 points when a Bollinger squeeze is detected.

        A squeeze (unusually narrow bandwidth) precedes explosive moves, rewarding
        any condition that fires near a squeeze zone.

        Args:
            frame: OHLCV plus BB columns.

        Returns:
            Squeeze component score in ``[0, 20]``.
        """
        if "bb_bandwidth" not in frame.columns or frame.empty:
            return 0.0
        bw = frame["bb_bandwidth"].dropna()
        if len(bw) < 20:
            return 0.0
        current_bw = float(bw.iloc[-1])
        # Compare to rolling 50-bar percentile
        percentile = float((bw.tail(50) < current_bw).mean())
        # Below 20th percentile = squeeze in effect
        if percentile <= 0.2:
            return 20.0
        if percentile <= 0.35:
            return 12.0
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
                "Bollinger band touch with squeeze — textbook setup.",
                "Time the entry with a confirming candlestick pattern.",
            ]
        if s >= 70.0:
            return [
                "Strong band signal; set targets at the opposite band.",
                "Confirm with RSI for overbought/oversold validation.",
            ]
        if s >= 55.0:
            return [
                "Moderate BB signal; volatility is expanding but not extreme.",
                "Wait for a bar close outside the band for conviction.",
            ]
        if s >= 40.0:
            return [
                "Weak band reading; price is mid-range — low-probability setup.",
                "Monitor bandwidth for a squeeze before acting.",
            ]
        return [
            "No BB edge; price is inside bands with no squeeze.",
            "Re-evaluate after bandwidth contracts significantly.",
        ]

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Produce a composite ``AgentResult`` for a Bollinger Bands rule.

        Args:
            condition: ``{"indicator": "BB", "operator": "<", "value": "lower", "action": "BUY"}``.
            df: OHLCV history; Bollinger Bands are computed if absent.

        Returns:
            Populated ``AgentResult`` with score, win rate, and narrative feedback.
        """
        indicator = str(condition.get("indicator", "")).upper()
        if indicator not in ("BB", "BOLLINGER"):
            raise ValueError("BollingerAgent.evaluate_condition expects indicator 'BB'.")

        frame = self._ensure_bbands(df)

        if frame.empty:
            return AgentResult(
                agent_name="BB",
                score=0.0,
                win_rate=0.0,
                feedback=["No rows available for Bollinger Bands evaluation."],
                suggestions=self.generate_suggestions(0.0),
                action_alignment="NEUTRAL",
            )

        latest_close = float(frame["close"].iloc[-1])
        latest_upper = float(frame["bb_upper"].iloc[-1])
        latest_lower = float(frame["bb_lower"].iloc[-1])
        latest_middle = float(frame["bb_middle"].iloc[-1])
        action = str(condition["action"]).upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("condition['action'] must be 'BUY' or 'SELL'.")

        rule = self._rule_points(action, latest_close, latest_upper, latest_lower, latest_middle)
        win_rate = float(self.calculate_win_rate(condition, frame))
        ml_component = 40.0 * win_rate
        squeeze = self._squeeze_points(frame)

        raw_total = rule + ml_component + squeeze
        score = self._safe_score(raw_total)

        if score >= 55.0:
            alignment: Literal["BUY", "SELL", "NEUTRAL"] = action  # type: ignore[assignment]
        else:
            alignment = "NEUTRAL"

        band_width = latest_upper - latest_lower
        position = ((latest_close - latest_lower) / band_width * 100) if band_width else 0.0
        feedback: list[str] = [
            f"BB Upper = {latest_upper:.2f}, Middle = {latest_middle:.2f}, Lower = {latest_lower:.2f}.",
            f"Close = {latest_close:.2f} — band position {position:.1f}%.",
            f"Rule-based credit {rule:.1f}/40.",
            f"Historical win-rate estimate = {win_rate:.2%} contributing {ml_component:.1f}/40.",
            f"Squeeze component = {squeeze:.1f}/20.",
        ]

        return AgentResult(
            agent_name="BB",
            score=score,
            win_rate=win_rate,
            feedback=feedback,
            suggestions=self.generate_suggestions(score),
            action_alignment=alignment,
        )


__all__ = ["BollingerAgent"]
