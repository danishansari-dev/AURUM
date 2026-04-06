"""Abstract base class for AURUM evaluation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Inter-agent communication protocol
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """
    Typed message envelope for agent-to-orchestrator communication.

    Every exchange between a specialist agent and the Orchestrator is wrapped
    in this dataclass so the system has a uniform, auditable message log.

    Attributes:
        sender:    Name of the originating agent (e.g. "RSIAgent").
        receiver:  Name of the destination (e.g. "OrchestratorAgent").
        msg_type:  Semantic tag — "SCORE", "CONFLICT_FLAG", or "DATA_REQUEST".
        payload:   Arbitrary dict containing the message body.
        timestamp: UTC instant the message was created.
    """

    sender: str
    receiver: str
    msg_type: Literal["SCORE", "CONFLICT_FLAG", "DATA_REQUEST"]
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class AgentResult(BaseModel):
    """
    Standardized output from an agent evaluation pass.

    Attributes:
        agent_name: Logical name of the agent that produced the result.
        score: Composite score on a 0--100 scale.
        win_rate: Historical win-rate fraction in ``[0.0, 1.0]`` for the active rule.
        feedback: Human-readable observations from the evaluation.
        suggestions: Actionable next steps for the user or orchestrator.
        action_alignment: Whether the setup aligns with buy, sell, or neutral posture.
    """

    agent_name: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=100.0)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    feedback: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    action_alignment: Literal["BUY", "SELL", "NEUTRAL"]

    @field_validator("win_rate", mode="before")
    @classmethod
    def _clamp_win_rate(cls, value: float) -> float:
        """Ensure win-rate stays inside ``[0, 1]`` when constructed from raw floats."""
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("win_rate must be numeric") from exc
        return max(0.0, min(1.0, v))


class BaseAgent(ABC):
    """
    Abstract agent that scores market conditions against a rule and price history.

    Subclasses implement indicator-specific logic for ``evaluate_condition`` and
    ``calculate_win_rate`` while reusing shared scoring utilities.
    """

    @abstractmethod
    def evaluate_condition(self, condition: dict, df: pd.DataFrame) -> AgentResult:
        """
        Score how well the latest market context satisfies ``condition``.

        Args:
            condition: Parsed rule describing indicator, operator, threshold, and action.
            df: OHLCV plus any precomputed columns required by the agent.

        Returns:
            Structured ``AgentResult`` including score and narrative fields.
        """

    @abstractmethod
    def calculate_win_rate(self, condition: dict, df: pd.DataFrame) -> float:
        """
        Estimate historical success rate of the rule on ``df``.

        Args:
            condition: Parsed rule describing indicator, operator, threshold, and action.
            df: OHLCV history plus indicators.

        Returns:
            Fraction of successful outcomes in ``[0.0, 1.0]``.
        """

    def _safe_score(self, raw: float) -> float:
        """
        Clamp an arbitrary raw score into ``[0, 100]``.

        Args:
            raw: Unbounded composite score.

        Returns:
            Clamped score suitable for ``AgentResult.score``.
        """
        if pd.isna(raw):
            return 0.0
        return float(max(0.0, min(100.0, raw)))


__all__ = ["AgentResult", "BaseAgent"]
