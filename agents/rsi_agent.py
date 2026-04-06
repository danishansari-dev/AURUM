"""RSI-focused agent with rule, historical, and divergence scoring."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from agents.base_agent import AgentResult, BaseAgent
from indicators.ta_compat import ta


class RSIAgent(BaseAgent):
    """
    Evaluates RSI-based conditions using rules, empirical win rate, and divergence.

    Scoring model (100 points):

    - Up to 40 points from threshold rules on the latest RSI print.
    - Up to 40 points from ``calculate_win_rate(condition, df) * 40``.
    - Up to 20 points from divergence in the last 20 bars (bullish for BUY, bearish for SELL).
    """

    _FORWARD_HORIZON: int = 5
    _MOVE_PCT: float = 0.005
    _DIVERGENCE_LOOKBACK: int = 20

    def _ensure_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure ``df`` contains an ``rsi`` column, computing it via pandas_ta if missing.

        Args:
            df: Input OHLCV frame.

        Returns:
            Copy of ``df`` with ``rsi`` populated.
        """
        frame = df.copy()
        if "rsi" in frame.columns:
            return frame
        if "close" not in frame.columns:
            raise ValueError("DataFrame must include a 'close' column to compute RSI.")
        frame["rsi"] = ta.rsi(frame["close"], length=14)
        return frame

    @staticmethod
    def _rsi_mask(rsi: pd.Series, operator: str, value: float) -> pd.Series:
        """
        Build a boolean mask for rows satisfying ``RSI operator value``.

        Args:
            rsi: RSI series aligned with price.
            operator: Comparison operator as a string.
            value: Threshold applied to RSI.

        Returns:
            Boolean series marking rows where the condition is true.
        """
        op = operator.strip()
        if op == "<":
            return rsi < value
        if op == "<=":
            return rsi <= value
        if op == ">":
            return rsi > value
        if op == ">=":
            return rsi >= value
        if op == "==":
            return rsi == value
        if op == "!=":
            return rsi != value
        raise ValueError(f"Unsupported RSI operator: {operator!r}")

    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Measure how often a 0.5% move in the signal direction occurs within five bars.

        For each bar where the RSI condition matches, compare ``close`` at ``t`` to
        ``close`` at ``t + 5``. BUY requires +0.5% return; SELL requires -0.5%.

        Args:
            condition: Rule dict with ``indicator``, ``operator``, ``value``, ``action``.
            df: Historical OHLCV data.

        Returns:
            Fraction of qualifying setups that succeeded, or ``0.0`` if none.
        """
        if str(condition.get("indicator", "")).upper() != "RSI":
            raise ValueError("RSIAgent.calculate_win_rate expects indicator 'RSI'.")

        frame = self._ensure_rsi(df)
        rsi = frame["rsi"].astype(float)
        close = frame["close"].astype(float)

        op = str(condition["operator"])
        val = float(condition["value"])
        action = str(condition["action"]).upper()

        mask = self._rsi_mask(rsi, op, val)

        successes = 0
        total = 0
        horizon = self._FORWARD_HORIZON
        threshold = self._MOVE_PCT

        for i in range(len(frame) - horizon):
            if not bool(mask.iloc[i]):
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

        if total == 0:
            return 0.0
        return successes / total

    @staticmethod
    def _rule_points(action: str, rsi_value: float) -> float:
        """
        Map the latest RSI level to partial rule credit (max 40).

        Args:
            action: ``BUY`` or ``SELL`` from the condition.
            rsi_value: Latest RSI reading.

        Returns:
            Points in ``[0, 40]``.
        """
        act = action.upper()
        r = float(rsi_value)
        if pd.isna(r):
            return 0.0
        if act == "BUY":
            if r < 30.0:
                return 40.0
            if r < 35.0:
                return 30.0
            if r < 40.0:
                return 15.0
            if r < 45.0:
                return 5.0
            return 0.0
        if act == "SELL":
            if r > 70.0:
                return 40.0
            if r > 65.0:
                return 30.0
            if r > 60.0:
                return 15.0
            return 0.0
        return 0.0

    @staticmethod
    def _bullish_divergence(window: pd.DataFrame) -> bool:
        """
        Detect bullish RSI divergence (two swing lows: lower price low, higher RSI low).

        Args:
            window: Recent bars including ``low`` and ``rsi``.

        Returns:
            ``True`` if a classic two-trough bullish divergence is found.
        """
        if len(window) < 5:
            return False
        lows = window["low"].to_numpy(dtype=float)
        rsis = window["rsi"].to_numpy(dtype=float)

        trough_idx: list[int] = []
        for i in range(1, len(window) - 1):
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                trough_idx.append(i)
        if len(trough_idx) >= 2:
            i_prev, i_last = trough_idx[-2], trough_idx[-1]
            p1, p2 = float(lows[i_prev]), float(lows[i_last])
            r1, r2 = float(rsis[i_prev]), float(rsis[i_last])
            if pd.isna(r1) or pd.isna(r2):
                return False
            return p2 < p1 and r2 > r1

        mid = max(len(window) // 2, 2)
        w1 = window.iloc[:mid]
        w2 = window.iloc[mid:]
        t1 = w1["low"].idxmin()
        t2 = w2["low"].idxmin()
        p1 = float(window.loc[t1, "low"])
        p2 = float(window.loc[t2, "low"])
        r1 = float(window.loc[t1, "rsi"])
        r2 = float(window.loc[t2, "rsi"])
        if pd.isna(r1) or pd.isna(r2):
            return False
        return p2 < p1 and r2 > r1

    @staticmethod
    def _bearish_divergence(window: pd.DataFrame) -> bool:
        """
        Detect bearish RSI divergence (two swing highs: higher price high, lower RSI high).

        Args:
            window: Recent bars including ``high`` and ``rsi``.

        Returns:
            ``True`` if a classic two-peak bearish divergence is found.
        """
        if len(window) < 5:
            return False
        highs = window["high"].to_numpy(dtype=float)
        rsis = window["rsi"].to_numpy(dtype=float)

        peak_idx: list[int] = []
        for i in range(1, len(window) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                peak_idx.append(i)
        if len(peak_idx) < 2:
            mid = len(window) // 2
            w1 = window.iloc[:mid]
            w2 = window.iloc[mid:]
            t1 = w1["high"].idxmax()
            t2 = w2["high"].idxmax()
            p1 = float(window.loc[t1, "high"])
            p2 = float(window.loc[t2, "high"])
            r1 = float(window.loc[t1, "rsi"])
            r2 = float(window.loc[t2, "rsi"])
            if pd.isna(r1) or pd.isna(r2):
                return False
            return p2 > p1 and r2 < r1

        i_prev, i_last = peak_idx[-2], peak_idx[-1]
        p1, p2 = float(highs[i_prev]), float(highs[i_last])
        r1, r2 = float(rsis[i_prev]), float(rsis[i_last])
        if pd.isna(r1) or pd.isna(r2):
            return False
        return p2 > p1 and r2 < r1

    def _divergence_points(self, action: str, frame: pd.DataFrame) -> float:
        """
        Award up to 20 points when divergence aligns with the intended action.

        Args:
            action: ``BUY`` or ``SELL``.
            frame: OHLCV plus ``rsi`` (and ``high`` / ``low``).

        Returns:
            Divergence component score in ``{0, 20}``.
        """
        tail = frame.tail(self._DIVERGENCE_LOOKBACK)
        act = action.upper()
        if act == "BUY":
            return 20.0 if self._bullish_divergence(tail) else 0.0
        if act == "SELL":
            return 20.0 if self._bearish_divergence(tail) else 0.0
        return 0.0

    def generate_suggestions(self, score: float) -> list[str]:
        """
        Translate aggregate score bands into concise follow-up suggestions.

        Args:
            score: Final composite score in ``[0, 100]``.

        Returns:
            Ordered list of recommendation strings.
        """
        s = float(score)
        if pd.isna(s):
            s = 0.0
        s = max(0.0, min(100.0, s))

        if s >= 85.0:
            return [
                "Setup is exceptionally strong versus historical baselines.",
                "Tighten risk controls but consider full-size participation if portfolio rules allow.",
                "Log the entry thesis and indicator snapshots for post-trade review.",
            ]
        if s >= 70.0:
            return [
                "Edge is materially above noise; prioritize execution quality.",
                "Confirm with higher-timeframe trend and upcoming macro events.",
                "Define stop placement using recent swing structure.",
            ]
        if s >= 55.0:
            return [
                "Moderate conviction; scale in or reduce size versus baseline.",
                "Watch for confirmation on the next bar cluster before adding.",
                "Re-evaluate if RSI exits the configured threshold band.",
            ]
        if s >= 40.0:
            return [
                "Signal is marginal; default to patience or paper-trade first.",
                "Seek confluence from another agent or timeframe before committing risk.",
                "Consider waiting for a clearer RSI extreme or divergence.",
            ]
        return [
            "No compelling RSI edge detected under current parameters.",
            "Avoid new exposure unless unrelated models disagree strongly.",
            "Re-run after fresh data arrives or adjust the condition thresholds.",
        ]

    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Produce a composite ``AgentResult`` for an RSI rule against current history.

        Args:
            condition: ``{"indicator": "RSI", "operator": "<", "value": 35, "action": "BUY"}``.
            df: OHLCV history; RSI is computed if absent (via ``pandas_ta``).

        Returns:
            Populated ``AgentResult`` with score, win rate, and narrative fields.
        """
        if str(condition.get("indicator", "")).upper() != "RSI":
            raise ValueError("RSIAgent.evaluate_condition expects indicator 'RSI'.")

        frame = self._ensure_rsi(df)
        if frame.empty:
            return AgentResult(
                agent_name="RSI",
                score=0.0,
                win_rate=0.0,
                feedback=["No rows available for RSI evaluation."],
                suggestions=self.generate_suggestions(0.0),
                action_alignment="NEUTRAL",
            )

        latest_rsi = float(frame["rsi"].iloc[-1])
        action = str(condition["action"]).upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("condition['action'] must be 'BUY' or 'SELL'.")

        rule = self._rule_points(action, latest_rsi)
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
            f"Latest RSI={latest_rsi:.2f} with rule-based credit {rule:.1f}/40.",
            f"Historical win-rate estimate={win_rate:.2%} contributing {ml_component:.1f}/40.",
            f"Divergence component={div:.1f}/20.",
        ]

        suggestions = self.generate_suggestions(score)

        return AgentResult(
            agent_name="RSI",
            score=score,
            win_rate=win_rate,
            feedback=feedback,
            suggestions=suggestions,
            action_alignment=alignment,
        )


__all__ = ["RSIAgent"]
