"""Master orchestrator that coordinates all specialist agents and resolves conflicts.

Uses concurrent.futures.ThreadPoolExecutor for parallel agent evaluation (fan-out)
and an AgentMessage protocol for auditable inter-agent communication.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

import pandas as pd

from agents.base_agent import AgentMessage, AgentResult
from agents.rsi_agent import RSIAgent
from agents.ema_agent import EMAAgent
from agents.sma_agent import SMAAgent
from agents.macd_agent import MACDAgent
from agents.bollinger_agent import BollingerAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central coordinator — receives parsed strategy conditions, dispatches each to the
    appropriate specialist agent, detects inter-agent conflicts, computes a weighted
    final score, and returns a structured evaluation report.

    Orchestration pattern (synchronous fan-out):
        User Input → StrategyParser.parse() → OrchestratorAgent
            ├── RSIAgent.evaluate()       ─┐
            ├── EMAAgent.evaluate()        │ ThreadPoolExecutor
            ├── MACDAgent.evaluate()       │ (parallel, independent)
            ├── SMAAgent.evaluate()        │
            └── BollingerAgent.evaluate() ─┘
                           ↓
            detect_conflicts() → BacktestEngine.run() → final verdict
    """

    # Relative importance of each indicator for Gold (XAUUSD) evaluation.
    _AGENT_WEIGHTS: dict[str, float] = {
        "RSI": 0.25,
        "EMA": 0.22,
        "MACD": 0.20,
        "SMA": 0.15,
        "BB": 0.18,
    }

    # ThreadPool size — 5 agents means 5 workers is ideal
    _MAX_WORKERS: int = 5

    def __init__(self) -> None:
        """Instantiate every specialist agent once so they persist across evaluations."""
        self._rsi_agent = RSIAgent()
        self._ema_agent = EMAAgent()
        self._sma_agent = SMAAgent()
        self._macd_agent = MACDAgent()
        self._bollinger_agent = BollingerAgent()

        # Lookup table: canonical indicator key → agent instance
        self._agents: dict[str, RSIAgent | EMAAgent | SMAAgent | MACDAgent | BollingerAgent] = {
            "RSI": self._rsi_agent,
            "EMA": self._ema_agent,
            "SMA": self._sma_agent,
            "MACD": self._macd_agent,
            "BB": self._bollinger_agent,
            # Allow alternative keys to route to the same agent
            "BOLLINGER": self._bollinger_agent,
        }

        # Auditable log of every AgentMessage exchanged during an evaluation pass
        self.message_log: list[AgentMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_strategy(
        self,
        parsed_conditions: list[dict],
        df: pd.DataFrame,
    ) -> dict:
        """
        Full-pipeline strategy evaluation with parallel agent dispatch.

        Dispatches every condition to its specialist agent using ThreadPoolExecutor,
        detects inter-agent conflicts, computes a weighted final score, and assembles
        a structured report suitable for the dashboard layer.

        Args:
            parsed_conditions: List of condition dicts produced by the StrategyParser.
                Each dict must contain at least ``indicator`` and ``action``.
            df: OHLCV DataFrame (daily or intraday) with a ``close`` column at minimum.

        Returns:
            Dictionary with keys:
                - ``final_score``       (float)  — weighted 0–100 composite.
                - ``rating``            (str)    — human-readable rating label.
                - ``individual_results``(dict)   — ``{indicator: AgentResult}`` map.
                - ``conflicts``         (list)   — conflict descriptors from ``detect_conflicts``.
                - ``suggestions``       (list)   — aggregated, deduplicated suggestion strings.
                - ``backtest_ready``    (bool)   — ``True`` when every condition produced a valid result.
        """
        # Clear message log for this evaluation pass
        self.message_log = []

        # --- Step 1: Route each condition to the correct specialist agent (parallel) ---
        individual_results = self._dispatch_conditions(parsed_conditions, df)

        # --- Step 2: Detect conflicts across agent results ---
        conflicts = self.detect_conflicts(individual_results)

        # Log conflict messages
        for conflict in conflicts:
            self.message_log.append(AgentMessage(
                sender="OrchestratorAgent",
                receiver="Dashboard",
                msg_type="CONFLICT_FLAG",
                payload=conflict,
            ))

        # --- Step 3: Compute weighted composite score ---
        weighted_score = self._compute_weighted_score(individual_results)

        # --- Step 4: Apply conflict penalty ---
        # Critical conflicts erode confidence in the aggregate score
        high_conflicts = sum(1 for c in conflicts if c["severity"] == "HIGH")
        medium_conflicts = sum(1 for c in conflicts if c["severity"] == "MEDIUM")
        if high_conflicts:
            weighted_score *= max(0.5, 1.0 - 0.15 * high_conflicts)
        if medium_conflicts:
            weighted_score *= max(0.7, 1.0 - 0.05 * medium_conflicts)
        weighted_score = max(0.0, min(100.0, weighted_score))

        # --- Step 5: Assemble response ---
        rating = self.score_to_rating(weighted_score)
        suggestions = self._aggregate_suggestions(individual_results, conflicts)
        backtest_ready = (
            len(individual_results) > 0
            and all(r.score > 0.0 for r in individual_results.values())
        )

        return {
            "final_score": round(weighted_score, 2),
            "rating": rating,
            "individual_results": individual_results,
            "conflicts": conflicts,
            "suggestions": suggestions,
            "backtest_ready": backtest_ready,
        }

    # ------------------------------------------------------------------
    # Parallel agent dispatch (fan-out via ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _dispatch_conditions(
        self,
        parsed_conditions: list[dict],
        df: pd.DataFrame,
    ) -> dict[str, AgentResult]:
        """
        Fan-out: dispatch each condition to its specialist agent in parallel.

        Uses ThreadPoolExecutor for concurrent evaluation. Each agent is independent
        and shares no state, so parallelism is safe and improves latency.

        Args:
            parsed_conditions: Parsed condition dicts from the StrategyParser.
            df: OHLCV + indicator DataFrame.

        Returns:
            ``{agent_name: AgentResult}`` map for all successfully evaluated conditions.
        """
        individual_results: dict[str, AgentResult] = {}

        # Build work items: (indicator_key, agent_instance, condition)
        work_items: list[tuple[str, object, dict]] = []
        for condition in parsed_conditions:
            indicator_key = str(condition.get("indicator", "")).upper()
            agent = self._agents.get(indicator_key)
            if agent is None:
                logger.warning("Unknown indicator '%s' — skipping.", indicator_key)
                continue
            work_items.append((indicator_key, agent, condition))

        if not work_items:
            return individual_results

        # Fan-out with ThreadPoolExecutor for parallel agent evaluation
        with ThreadPoolExecutor(max_workers=min(self._MAX_WORKERS, len(work_items))) as executor:
            future_to_key = {
                executor.submit(self._evaluate_single, agent, condition, df): indicator_key
                for indicator_key, agent, condition in work_items
            }

            for future in as_completed(future_to_key):
                indicator_key = future_to_key[future]
                try:
                    result = future.result()
                    individual_results[result.agent_name] = result

                    # Log successful SCORE message
                    self.message_log.append(AgentMessage(
                        sender=f"{result.agent_name}Agent",
                        receiver="OrchestratorAgent",
                        msg_type="SCORE",
                        payload={
                            "score": result.score,
                            "win_rate": result.win_rate,
                            "action": result.action_alignment,
                        },
                    ))
                except (ValueError, KeyError, IndexError) as exc:
                    logger.error("Agent %s failed: %s", indicator_key, exc)
                    # Record a zero-score sentinel so the dashboard can show which agent failed
                    individual_results[indicator_key] = AgentResult(
                        agent_name=indicator_key,
                        score=0.0,
                        win_rate=0.0,
                        feedback=[f"Evaluation failed: {exc}"],
                        suggestions=["Fix the condition syntax and retry."],
                        action_alignment="NEUTRAL",
                    )

        return individual_results

    @staticmethod
    def _evaluate_single(agent: object, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Evaluate a single condition against a single agent.

        Isolated as a static method to be safely submitted to ThreadPoolExecutor.
        Each agent is stateless for evaluation so concurrent calls are safe.

        Args:
            agent: Specialist agent instance (RSIAgent, EMAAgent, etc.).
            condition: Parsed condition dict.
            df: OHLCV + indicator DataFrame.

        Returns:
            AgentResult from the specialist agent.
        """
        return agent.evaluate_condition(condition, df)

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        results: dict[str, AgentResult],
    ) -> list[dict]:
        """
        Identify logical contradictions between agent evaluations.

        Three conflict patterns are checked:

        1. **Action alignment mismatch** — one agent says BUY while another says SELL.
        2. **RSI overbought + EMA uptrend** — RSI signals exhaustion but EMA still
           points up, creating a contradiction.
        3. **MACD bearish crossover + RSI oversold** — MACD says momentum is fading
           while RSI says price is washed out, producing ambiguity.

        Args:
            results: Map of ``{agent_name: AgentResult}`` from the evaluation pass.

        Returns:
            List of conflict dicts, each containing:
                - ``conflict_type``   (str)  — machine-readable label.
                - ``agents_involved`` (list) — names of the clashing agents.
                - ``description``     (str)  — human-readable explanation.
                - ``severity``        (str)  — ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        """
        conflicts: list[dict] = []

        # --- Conflict 1: Action alignment mismatch (BUY vs SELL) ---
        buy_agents = [
            name for name, r in results.items()
            if r.action_alignment == "BUY"
        ]
        sell_agents = [
            name for name, r in results.items()
            if r.action_alignment == "SELL"
        ]

        if buy_agents and sell_agents:
            conflicts.append({
                "conflict_type": "action_alignment_mismatch",
                "agents_involved": buy_agents + sell_agents,
                "description": (
                    f"Directional disagreement: {', '.join(buy_agents)} signal BUY "
                    f"while {', '.join(sell_agents)} signal SELL. "
                    "The strategy contains contradictory signals that may cancel each other out."
                ),
                "severity": "HIGH",
            })

        # --- Conflict 2: RSI overbought (>70) + EMA says uptrend ---
        rsi_result = results.get("RSI")
        ema_result = results.get("EMA")

        if rsi_result and ema_result:
            # Extract the latest RSI value from feedback (format: "Latest RSI=XX.XX ...")
            rsi_value = self._extract_rsi_value(rsi_result)
            ema_alignment = ema_result.action_alignment

            if rsi_value is not None and rsi_value > 70.0 and ema_alignment == "BUY":
                conflicts.append({
                    "conflict_type": "rsi_overbought_ema_uptrend",
                    "agents_involved": ["RSI", "EMA"],
                    "description": (
                        f"RSI is overbought at {rsi_value:.1f} (>70), suggesting exhaustion, "
                        f"but EMA still indicates an uptrend (BUY alignment). "
                        "The trend may continue short-term but reversal risk is elevated."
                    ),
                    "severity": "MEDIUM",
                })

        # --- Conflict 3: MACD bearish crossover + RSI oversold ---
        macd_result = results.get("MACD")

        if macd_result and rsi_result:
            rsi_value = self._extract_rsi_value(rsi_result)
            macd_alignment = macd_result.action_alignment

            if (
                rsi_value is not None
                and rsi_value < 30.0
                and macd_alignment == "SELL"
            ):
                conflicts.append({
                    "conflict_type": "macd_bearish_rsi_oversold",
                    "agents_involved": ["MACD", "RSI"],
                    "description": (
                        f"MACD signals bearish momentum (SELL alignment), but RSI at "
                        f"{rsi_value:.1f} (<30) indicates oversold conditions. "
                        "This is ambiguous — the sell-off may be overdone, or momentum "
                        "collapse could continue."
                    ),
                    "severity": "MEDIUM",
                })

        return conflicts

    # ------------------------------------------------------------------
    # Score → rating mapping
    # ------------------------------------------------------------------

    @staticmethod
    def score_to_rating(score: float) -> str:
        """
        Convert a numeric score into a categorical rating string.

        Args:
            score: Composite strategy score in ``[0, 100]``.

        Returns:
            Rating label with emoji and explanation.
        """
        if score >= 85.0:
            return "🏆 EXCELLENT — Battle-tested Strategy"
        if score >= 70.0:
            return "✅ GOOD — Solid with minor improvements"
        if score >= 55.0:
            return "🟡 AVERAGE — Works but inconsistent"
        if score >= 40.0:
            return "⚠️ WEAK — High risk of losses"
        return "❌ POOR — Do not trade this strategy"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weighted_score(self, results: dict[str, AgentResult]) -> float:
        """
        Weighted average of agent scores, normalised by the total weight of agents
        that actually participated so that missing agents don't drag the score to zero.

        Args:
            results: Map of ``{agent_name: AgentResult}``.

        Returns:
            Weighted composite score in ``[0, 100]``.
        """
        if not results:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for agent_name, result in results.items():
            weight = self._AGENT_WEIGHTS.get(agent_name, 0.0)
            if weight == 0.0:
                # Unknown agent — use a neutral default weight
                weight = 0.10
            weighted_sum += result.score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        # Normalise so the score stays on a 0–100 scale regardless of how many
        # agents participated
        return weighted_sum / total_weight

    @staticmethod
    def _extract_rsi_value(rsi_result: AgentResult) -> float | None:
        """
        Parse the latest RSI reading from an RSIAgent's feedback strings.

        The RSIAgent embeds the value in the first feedback line as
        ``"Latest RSI=XX.XX ..."``. This helper extracts that number.

        Args:
            rsi_result: AgentResult produced by ``RSIAgent.evaluate_condition``.

        Returns:
            RSI float value, or ``None`` if parsing fails.
        """
        import re

        for line in rsi_result.feedback:
            match = re.search(r"RSI\s*=\s*([\d.]+)", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _aggregate_suggestions(
        results: dict[str, AgentResult],
        conflicts: list[dict],
    ) -> list[str]:
        """
        Collect and deduplicate suggestions from all agents and conflict warnings.

        Args:
            results: Map of ``{agent_name: AgentResult}``.
            conflicts: Conflict descriptors from ``detect_conflicts``.

        Returns:
            Ordered, deduplicated list of suggestion strings.
        """
        seen: set[str] = set()
        suggestions: list[str] = []

        # Conflict-derived suggestions first — they carry the highest priority
        for conflict in conflicts:
            severity = conflict["severity"]
            desc = conflict["description"]
            tip = f"[{severity}] {desc}"
            if tip not in seen:
                seen.add(tip)
                suggestions.append(tip)

        # Agent-level suggestions
        for _name, result in results.items():
            for s in result.suggestions:
                if s not in seen:
                    seen.add(s)
                    suggestions.append(s)

        return suggestions


__all__ = ["OrchestratorAgent"]
