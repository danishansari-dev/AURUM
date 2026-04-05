"""MACD-focused agent with crossover, histogram, and divergence scoring."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from agents.base_agent import AgentResult, BaseAgent
from indicators.ta_compat import ta


class MACDAgent(BaseAgent):
    """
    Evaluates MACD-based conditions: line crossovers, histogram polarity, and divergence.

    Scoring model (100 points):

    - Up to 40 points from crossover / histogram rules on the latest bar.
    - Up to 40 points from ``calculate_win_rate(condition, df) * 40``.
    - Up to 20 points from MACD-price divergence in the last 20 bars.
    """

    _FORWARD_HORIZON: int = 5
    _MOVE_PCT: float = 0.005
    _DIVERGENCE_LOOKBACK: int = 20
    _FAST: int = 12
    _SLOW: int = 26
    _SIGNAL: int = 9

    def _ensure_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee ``macd``, ``macd_signal``, and ``macd_hist`` columns exist.

        Args:
            df: Input OHLCV frame.

        Returns:
            Copy of *df* with MACD columns populated.
        """
        frame = df.copy()
        if {"macd", "macd_signal", "macd_hist"}.issubset(frame.columns):
            return frame
        if "close" not in frame.columns:
            raise ValueError("DataFrame must include a 'close' column to compute MACD.")
        macd_df = ta.macd(frame["close"], fast=self._FAST, slow=self._SLOW, signal=self._SIGNAL)
        # pandas_ta returns columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        cols = list(macd_df.columns)
        frame["macd"] = macd_df[cols[0]]
        frame["macd_signal"] = macd_df[cols[1]]
        frame["macd_hist"] = macd_df[cols[2]]
        return frame

    # ------------------------------------------------------------------
    # Win-rate estimation
    # ------------------------------------------------------------------

    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Fraction of MACD-condition bars that saw a 0.5 % forward move.

        Uses histogram polarity as the default signal when operator is not specified.

        Args:
            condition: Rule dict with ``indicator``, ``operator``, ``value``, ``action``.
            df: Historical OHLCV data.

        Returns:
            Success fraction in ``[0.0, 1.0]``, or ``0.0`` when no setups found.
        """
        if str(condition.get("indicator", "")).upper() != "MACD":
            raise ValueError("MACDAgent.calculate_win_rate expects indicator 'MACD'.")

        frame = self._ensure_macd(df)
        close = frame["close"].astype(float)
        hist = frame["macd_hist"].astype(float)
        action = str(condition["action"]).upper()
        op = str(condition.get("operator", "crossover_above"))

        # Build mask based on the operator type
        if op in ("crossover_above", ">"):
            mask = hist > 0
        elif op in ("crossover_below", "<"):
            mask = hist < 0
        else:
            mask = hist > 0 if action == "BUY" else hist < 0

        successes, total = 0, 0
        horizon = self._FORWARD_HORIZON
        threshold = self._MOVE_PCT

        for i in range(len(frame) - horizon):
            if pd.isna(hist.iloc[i]) or not bool(mask.iloc[i]):
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

    def _rule_points(self, action: str, macd_val: float, signal_val: float, hist_val: float) -> float:
        """
        Award up to 40 rule-based points from MACD line/signal/histogram analysis.

        Args:
            action: ``BUY`` or ``SELL``.
            macd_val: Latest MACD line value.
            signal_val: Latest signal line value.
            hist_val: Latest histogram value.

        Returns:
            Points in ``[0, 40]``.
        """
        if any(pd.isna(v) for v in (macd_val, signal_val, hist_val)):
            return 0.0

        act = action.upper()
        points = 0.0

        if act == "BUY":
            # MACD above signal → bullish crossover
            if macd_val > signal_val:
                points += 20.0
            # Histogram positive and growing
            if hist_val > 0:
                points += 10.0
            # MACD above zero line → strong trend
            if macd_val > 0:
                points += 10.0
        elif act == "SELL":
            if macd_val < signal_val:
                points += 20.0
            if hist_val < 0:
                points += 10.0
            if macd_val < 0:
                points += 10.0

        return min(40.0, points)

    def _divergence_points(self, action: str, frame: pd.DataFrame) -> float:
        """
        Award up to 20 points when MACD-price divergence aligns with action.

        Args:
            action: ``BUY`` or ``SELL``.
            frame: OHLCV plus MACD columns.

        Returns:
            Divergence component score in ``{0, 20}``.
        """
        tail = frame.tail(self._DIVERGENCE_LOOKBACK)
        if len(tail) < 5:
            return 0.0

        close_arr = tail["close"].to_numpy(dtype=float)
        macd_arr = tail["macd"].to_numpy(dtype=float)

        act = action.upper()
        mid = len(tail) // 2

        if act == "BUY":
            # Bullish divergence: lower price low, higher MACD low
            p1 = float(np.nanmin(close_arr[:mid]))
            p2 = float(np.nanmin(close_arr[mid:]))
            m1 = float(np.nanmin(macd_arr[:mid]))
            m2 = float(np.nanmin(macd_arr[mid:]))
            if p2 < p1 and m2 > m1:
                return 20.0
        elif act == "SELL":
            # Bearish divergence: higher price high, lower MACD high
            p1 = float(np.nanmax(close_arr[:mid]))
            p2 = float(np.nanmax(close_arr[mid:]))
            m1 = float(np.nanmax(macd_arr[:mid]))
            m2 = float(np.nanmax(macd_arr[mid:]))
            if p2 > p1 and m2 < m1:
                return 20.0

        return 0.0

    def is_bearish_crossover(self, df: pd.DataFrame) -> bool:
        """
        Detect whether the latest bar exhibits a bearish MACD crossover.

        A bearish crossover occurs when the MACD line crosses **below** the signal
        line, i.e. the previous bar had ``macd >= signal`` and the current bar has
        ``macd < signal``.

        Args:
            df: OHLCV frame with or without precomputed MACD columns.

        Returns:
            ``True`` if a bearish crossover is detected on the last bar.
        """
        frame = self._ensure_macd(df)
        if len(frame) < 2:
            return False
        prev_macd = float(frame["macd"].iloc[-2])
        prev_sig = float(frame["macd_signal"].iloc[-2])
        curr_macd = float(frame["macd"].iloc[-1])
        curr_sig = float(frame["macd_signal"].iloc[-1])
        if any(pd.isna(v) for v in (prev_macd, prev_sig, curr_macd, curr_sig)):
            return False
        return prev_macd >= prev_sig and curr_macd < curr_sig

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
                "MACD confirmation is textbook-quality; proceed with conviction.",
                "Set take-profit at the next MACD-histogram divergence zone.",
            ]
        if s >= 70.0:
            return [
                "Strong MACD crossover signal; validate with RSI for timing.",
                "Trail the stop using histogram sign-flip.",
            ]
        if s >= 55.0:
            return [
                "Moderate MACD reading; wait for histogram to expand before sizing up.",
                "Check if MACD is near the zero line for added strength.",
            ]
        if s >= 40.0:
            return [
                "Weak MACD signal; crossover may be a whipsaw.",
                "Combine with EMA trend filter to reduce false signals.",
            ]
        return [
            "No compelling MACD edge; avoid trading on MACD alone.",
            "Wait for a clean zero-line crossover before re-entering.",
        ]

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Produce a composite ``AgentResult`` for a MACD rule on current history.

        Args:
            condition: ``{"indicator": "MACD", "operator": "crossover_above", "value": 0, "action": "BUY"}``.
            df: OHLCV history; MACD is computed if absent.

        Returns:
            Populated ``AgentResult`` with score, win rate, and narrative feedback.
        """
        if str(condition.get("indicator", "")).upper() != "MACD":
            raise ValueError("MACDAgent.evaluate_condition expects indicator 'MACD'.")

        frame = self._ensure_macd(df)

        if frame.empty:
            return AgentResult(
                agent_name="MACD",
                score=0.0,
                win_rate=0.0,
                feedback=["No rows available for MACD evaluation."],
                suggestions=self.generate_suggestions(0.0),
                action_alignment="NEUTRAL",
            )

        latest_macd = float(frame["macd"].iloc[-1])
        latest_signal = float(frame["macd_signal"].iloc[-1])
        latest_hist = float(frame["macd_hist"].iloc[-1])
        action = str(condition["action"]).upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("condition['action'] must be 'BUY' or 'SELL'.")

        rule = self._rule_points(action, latest_macd, latest_signal, latest_hist)
        win_rate = float(self.calculate_win_rate(condition, frame))
        ml_component = 40.0 * win_rate
        div = self._divergence_points(action, frame)

        raw_total = rule + ml_component + div
        score = self._safe_score(raw_total)

        if score >= 55.0:
            alignment: Literal["BUY", "SELL", "NEUTRAL"] = action  # type: ignore[assignment]
        else:
            alignment = "NEUTRAL"

        feedback: list[str] = [
            f"MACD = {latest_macd:.4f}, Signal = {latest_signal:.4f}, Hist = {latest_hist:.4f}.",
            f"Rule-based credit {rule:.1f}/40.",
            f"Historical win-rate estimate = {win_rate:.2%} contributing {ml_component:.1f}/40.",
            f"Divergence component = {div:.1f}/20.",
        ]

        return AgentResult(
            agent_name="MACD",
            score=score,
            win_rate=win_rate,
            feedback=feedback,
            suggestions=self.generate_suggestions(score),
            action_alignment=alignment,
        )


__all__ = ["MACDAgent"]
