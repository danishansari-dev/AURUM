"""Strategy text → structured condition list with regex-first, Claude-fallback parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic validation model
# ---------------------------------------------------------------------------

class StrategyCondition(BaseModel):
    """
    Validated representation of a single parsed trading condition.

    Every condition produced by the parser is funnelled through this model so
    downstream agents always receive well-typed, bounded data.
    """

    indicator: str = Field(..., min_length=1, description="Technical indicator key (RSI, EMA, SMA, MACD, BB).")
    operator: str = Field(..., min_length=1, description="Comparison or crossover operator.")
    value: Optional[float] = Field(default=None, description="Threshold value (e.g. 30 for RSI < 30).")
    fast: Optional[int] = Field(default=None, ge=1, description="Fast period for EMA crossover.")
    slow: Optional[int] = Field(default=None, ge=1, description="Slow period for EMA crossover.")
    action: Literal["BUY", "SELL"] = Field(..., description="Inferred trade direction.")

    @field_validator("indicator", mode="before")
    @classmethod
    def _normalise_indicator(cls, v: str) -> str:
        """Canonical uppercase form so agents don't need to normalise themselves."""
        return str(v).strip().upper()

    @field_validator("action", mode="before")
    @classmethod
    def _normalise_action(cls, v: str) -> str:
        """Accept lowercase / mixed-case action strings."""
        return str(v).strip().upper()

    @model_validator(mode="after")
    def _crossover_requires_periods(self) -> "StrategyCondition":
        """EMA crossover conditions must carry both fast and slow periods."""
        if self.indicator == "EMA" and "crossover" in self.operator:
            if self.fast is None or self.slow is None:
                raise ValueError("EMA crossover conditions require both 'fast' and 'slow' periods.")
            if self.fast >= self.slow:
                raise ValueError(f"'fast' ({self.fast}) must be less than 'slow' ({self.slow}).")
        return self


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class StrategyParser:
    """
    Two-mode strategy parser: regex-first for speed and offline use, Claude-fallback
    for complex / ambiguous natural language inputs.

    Usage::

        parser = StrategyParser()
        conditions = parser.parse("Buy when RSI < 30 and EMA20 crosses above EMA50")
    """

    # Regex anchors for buy / sell intent
    _BUY_KEYWORDS = re.compile(r"\b(buy|long|bullish|go\s+long|enter\s+long)\b", re.IGNORECASE)
    _SELL_KEYWORDS = re.compile(r"\b(sell|short|bearish|go\s+short|enter\s+short)\b", re.IGNORECASE)

    # --- RSI patterns ---
    _RSI_BELOW = re.compile(
        r"RSI\s*(?:is\s+)?(?:below|under|<|less\s+than)\s+([\d.]+)",
        re.IGNORECASE,
    )
    _RSI_ABOVE = re.compile(
        r"RSI\s*(?:is\s+)?(?:above|over|>|greater\s+than)\s+([\d.]+)",
        re.IGNORECASE,
    )
    # BUG-005 fix: now handles "RSI crosses above 50" and "RSI crosses below 50"
    # by making the direction word (above/below) an optional captured group.
    _RSI_CROSSES = re.compile(
        r"RSI\s*(?:crosses?|cross)\s+(?:(above|below)\s+)?([\d.]+)",
        re.IGNORECASE,
    )

    # --- EMA crossover patterns ---
    _EMA_CROSS_ABOVE = re.compile(
        r"EMA\s*(\d+)\s+(?:crosses?\s+above|>)\s+EMA\s*(\d+)",
        re.IGNORECASE,
    )
    _EMA_CROSS_BELOW = re.compile(
        r"EMA\s*(\d+)\s+(?:crosses?\s+below|<)\s+EMA\s*(\d+)",
        re.IGNORECASE,
    )

    # --- Single EMA vs price ---
    _EMA_PRICE_ABOVE = re.compile(
        r"(?:price|close)\s+(?:is\s+)?(?:above|over|>)\s+EMA\s*(\d+)",
        re.IGNORECASE,
    )
    _EMA_PRICE_BELOW = re.compile(
        r"(?:price|close)\s+(?:is\s+)?(?:below|under|<)\s+EMA\s*(\d+)",
        re.IGNORECASE,
    )

    # --- SMA patterns ---
    _SMA_ABOVE = re.compile(
        r"(?:price|close)\s+(?:is\s+)?(?:above|over|>)\s+SMA\s*(\d+)",
        re.IGNORECASE,
    )
    _SMA_BELOW = re.compile(
        r"(?:price|close)\s+(?:is\s+)?(?:below|under|<)\s+SMA\s*(\d+)",
        re.IGNORECASE,
    )
    _SMA_CROSS_ABOVE = re.compile(
        r"SMA\s*(\d+)\s+(?:crosses?\s+above|>)\s+SMA\s*(\d+)",
        re.IGNORECASE,
    )
    _SMA_CROSS_BELOW = re.compile(
        r"SMA\s*(\d+)\s+(?:crosses?\s+below|<)\s+SMA\s*(\d+)",
        re.IGNORECASE,
    )

    # --- MACD patterns ---
    _MACD_BULLISH = re.compile(
        r"MACD\s+(?:line\s+)?(?:crosses?\s+above\s+signal|bullish\s+crossover)",
        re.IGNORECASE,
    )
    _MACD_BEARISH = re.compile(
        r"MACD\s+(?:line\s+)?(?:crosses?\s+below\s+signal|bearish\s+crossover)",
        re.IGNORECASE,
    )

    # --- Bollinger Bands patterns ---
    # BUG-006 fix: "upper" and "lower" are now REQUIRED to avoid ambiguous
    # matches where "price touches bollinger band" would fire both patterns.
    _BB_LOWER = re.compile(
        r"(?:price|close)\s+(?:(?:touches?|hits?|near|at|below|breaks?)\s+)?(?:the\s+)?"
        r"lower\s+bollinger\s*(?:band)?",
        re.IGNORECASE,
    )
    _BB_UPPER = re.compile(
        r"(?:price|close)\s+(?:(?:touches?|hits?|near|at|above|breaks?)\s+)?(?:the\s+)?"
        r"upper\s+bollinger\s*(?:band)?",
        re.IGNORECASE,
    )
    # Simpler fallback: just "bollinger lower" or "bollinger upper"
    _BB_LOWER_SIMPLE = re.compile(r"bollinger\s+lower", re.IGNORECASE)
    _BB_UPPER_SIMPLE = re.compile(r"bollinger\s+upper", re.IGNORECASE)
    # Catch ambiguous unqualified "bollinger band" (no upper/lower)
    _BB_AMBIGUOUS = re.compile(
        r"(?:price|close)\s+(?:(?:touches?|hits?|near|at|breaks?)\s+)?(?:the\s+)?"
        r"bollinger\s*(?:band)?(?!\s*(?:upper|lower))",
        re.IGNORECASE,
    )

    # Claude model used for AI-fallback parsing
    _CLAUDE_MODEL = "claude-sonnet-4-20250514"

    _SYSTEM_PROMPT = (
        "You are a trading strategy parser. Extract trading conditions from user text "
        "and return ONLY a JSON array. Each element: {\"indicator\": str, \"operator\": str, "
        "\"value\": number or null, \"fast\": int or null, \"slow\": int or null, "
        "\"action\": \"BUY\" or \"SELL\"}. No explanation, just the JSON array."
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, user_input: str) -> list[dict]:
        """
        Parse a strategy string into validated condition dicts.

        Tries regex first; falls back to Claude API when regex yields fewer than
        two conditions (likely means the input was too complex for pattern matching).

        Args:
            user_input: Free-text trading strategy from the user.

        Returns:
            List of validated condition dicts ready for the OrchestratorAgent.
        """
        conditions = self.parse_regex(user_input)

        # BUG-011 fix: threshold was < 2, which triggered Claude on valid
        # single-condition strategies like "Buy when RSI below 30".  Changed
        # to < 1 so fallback only fires when regex finds nothing at all.
        if len(conditions) < 1:
            logger.info(
                "Regex produced %d condition(s) — falling back to AI parser.",
                len(conditions),
            )
            try:
                ai_conditions = self.parse_with_ai(user_input)
                if ai_conditions:
                    conditions = ai_conditions
            except Exception:
                # If AI parser also fails, return whatever regex found
                logger.warning("AI parser failed; using regex results.", exc_info=True)

        return conditions

    # ------------------------------------------------------------------
    # MODE 1 — Regex parser
    # ------------------------------------------------------------------

    def parse_regex(self, user_input: str) -> list[dict]:
        """
        Extract trading conditions using compiled regex patterns.

        Deterministic, offline, zero-latency. Handles the most common condition
        patterns traders use.

        Args:
            user_input: Raw strategy text.

        Returns:
            List of validated condition dicts (may be empty for unparseable input).
        """
        text = user_input.strip()
        default_action = self._infer_action(text)
        raw_conditions: list[dict[str, Any]] = []

        # --- RSI ---
        for m in self._RSI_BELOW.finditer(text):
            raw_conditions.append({
                "indicator": "RSI",
                "operator": "<",
                "value": float(m.group(1)),
                "action": default_action,
            })

        for m in self._RSI_ABOVE.finditer(text):
            raw_conditions.append({
                "indicator": "RSI",
                "operator": ">",
                "value": float(m.group(1)),
                "action": default_action,
            })
            
        # BUG-005 fix: handle "RSI crosses above 50" and "RSI crosses below 50"
        # by reading the optional direction word captured in group(1).
        for m in self._RSI_CROSSES.finditer(text):
            direction_word = (m.group(1) or "").lower()
            val = float(m.group(2))
            if direction_word == "above":
                op = ">"
            elif direction_word == "below":
                op = "<"
            else:
                # No direction word: infer from value (>= 50 → crossing up)
                op = ">" if val >= 50 else "<"
            raw_conditions.append({
                "indicator": "RSI",
                "operator": op,
                "value": val,
                "action": default_action,
            })

        # --- EMA crossovers ---
        for m in self._EMA_CROSS_ABOVE.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            # Ensure fast < slow regardless of user ordering
            if fast > slow:
                fast, slow = slow, fast
            # BUG-004 fix: use default_action instead of hardcoded "BUY"
            # so the user's explicit direction keyword is respected.
            raw_conditions.append({
                "indicator": "EMA",
                "operator": "crossover_above",
                "fast": fast,
                "slow": slow,
                "action": default_action,
            })

        for m in self._EMA_CROSS_BELOW.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            if fast > slow:
                fast, slow = slow, fast
            # BUG-004 fix: use default_action instead of hardcoded "SELL"
            raw_conditions.append({
                "indicator": "EMA",
                "operator": "crossover_below",
                "fast": fast,
                "slow": slow,
                "action": default_action,
            })

        # --- Single EMA vs price ---
        for m in self._EMA_PRICE_ABOVE.finditer(text):
            raw_conditions.append({
                "indicator": "EMA",
                "operator": ">",
                "value": float(m.group(1)),
                "action": default_action,
            })

        for m in self._EMA_PRICE_BELOW.finditer(text):
            raw_conditions.append({
                "indicator": "EMA",
                "operator": "<",
                "value": float(m.group(1)),
                "action": default_action,
            })

        # --- SMA ---
        for m in self._SMA_ABOVE.finditer(text):
            raw_conditions.append({
                "indicator": "SMA",
                "operator": ">",
                "value": float(m.group(1)),
                "action": default_action,
            })

        for m in self._SMA_BELOW.finditer(text):
            raw_conditions.append({
                "indicator": "SMA",
                "operator": "<",
                "value": float(m.group(1)),
                "action": default_action,
            })

        for m in self._SMA_CROSS_ABOVE.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            if fast > slow:
                fast, slow = slow, fast
            # BUG-004 fix: use default_action instead of hardcoded "BUY"
            raw_conditions.append({
                "indicator": "SMA",
                "operator": "crossover_above",
                "fast": fast,
                "slow": slow,
                "action": default_action,
            })

        for m in self._SMA_CROSS_BELOW.finditer(text):
            fast, slow = int(m.group(1)), int(m.group(2))
            if fast > slow:
                fast, slow = slow, fast
            # BUG-004 fix: use default_action instead of hardcoded "SELL"
            raw_conditions.append({
                "indicator": "SMA",
                "operator": "crossover_below",
                "fast": fast,
                "slow": slow,
                "action": default_action,
            })

        # --- MACD ---
        if self._MACD_BULLISH.search(text):
            # BUG-004 fix: use default_action instead of hardcoded "BUY"
            raw_conditions.append({
                "indicator": "MACD",
                "operator": "crossover_above",
                "action": default_action,
            })

        if self._MACD_BEARISH.search(text):
            # BUG-004 fix: use default_action instead of hardcoded "SELL"
            raw_conditions.append({
                "indicator": "MACD",
                "operator": "crossover_below",
                "action": default_action,
            })

        # --- Bollinger Bands ---
        # BUG-006 fix: patterns now require upper/lower keyword.
        # Ambiguous input ("price touches bollinger band") defaults to lower
        # for BUY and upper for SELL instead of matching both.
        bb_lower_match = self._BB_LOWER.search(text) or self._BB_LOWER_SIMPLE.search(text)
        bb_upper_match = self._BB_UPPER.search(text) or self._BB_UPPER_SIMPLE.search(text)

        if bb_lower_match:
            raw_conditions.append({
                "indicator": "BB",
                "operator": "<",
                "value": 0.0,  # sentinel — agent uses lower band internally
                "action": default_action if default_action != "SELL" else "BUY",
            })

        if bb_upper_match:
            raw_conditions.append({
                "indicator": "BB",
                "operator": ">",
                "value": 0.0,
                "action": default_action if default_action != "BUY" else "SELL",
            })

        # Fallback for ambiguous "bollinger band" without upper/lower qualifier
        if not bb_lower_match and not bb_upper_match and self._BB_AMBIGUOUS.search(text):
            # Default: lower band for BUY, upper band for SELL
            if default_action == "BUY":
                raw_conditions.append({
                    "indicator": "BB",
                    "operator": "<",
                    "value": 0.0,
                    "action": "BUY",
                })
                logger.info("Ambiguous 'bollinger band' — defaulting to lower band for BUY.")
            else:
                raw_conditions.append({
                    "indicator": "BB",
                    "operator": ">",
                    "value": 0.0,
                    "action": "SELL",
                })
                logger.info("Ambiguous 'bollinger band' — defaulting to upper band for SELL.")

        return self._validate_conditions(raw_conditions)

    # ------------------------------------------------------------------
    # MODE 2 — Claude API parser
    # ------------------------------------------------------------------

    def parse_with_ai(self, user_input: str) -> list[dict]:
        """
        Fall back to Claude for complex / ambiguous strategy text.

        Requires the ``anthropic`` package and a valid ``ANTHROPIC_API_KEY``
        environment variable.

        Args:
            user_input: Raw strategy text.

        Returns:
            List of validated condition dicts parsed from the Claude response.

        Raises:
            ImportError: If ``anthropic`` is not installed.
            RuntimeError: If the API call or JSON parsing fails.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AI-based parsing. "
                "Install it with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic()

        try:
            response = client.messages.create(
                model=self._CLAUDE_MODEL,
                max_tokens=1024,
                system=self._SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_input},
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"Claude API call failed: {exc}") from exc

        # Extract JSON from the response text
        raw_text = response.content[0].text.strip()
        raw_conditions = self._extract_json_array(raw_text)

        if raw_conditions is None:
            raise RuntimeError(f"Could not parse JSON from Claude response: {raw_text!r}")

        return self._validate_conditions(raw_conditions)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_action(self, text: str) -> Literal["BUY", "SELL"]:
        """
        Determine trade direction from contextual keywords in the full input.

        Defaults to ``"BUY"`` when the text is ambiguous, matching the convention
        that most retail traders state buy-side conditions.

        Args:
            text: Full user strategy string.

        Returns:
            ``"BUY"`` or ``"SELL"``.
        """
        has_buy = bool(self._BUY_KEYWORDS.search(text))
        has_sell = bool(self._SELL_KEYWORDS.search(text))

        if has_sell and not has_buy:
            return "SELL"
        # Default to BUY when ambiguous or when both appear (primary intent is usually buy)
        return "BUY"

    @staticmethod
    def _extract_json_array(text: str) -> list[dict] | None:
        """
        Robustly extract a JSON array from a string that may contain markdown fences.

        Args:
            text: Raw Claude response text.

        Returns:
            Parsed list of dicts, or ``None`` if extraction fails.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip().rstrip("`")

        # Find the outermost [ ... ] block
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _validate_conditions(raw: list[dict]) -> list[dict]:
        """
        Validate each raw condition dict through the ``StrategyCondition`` model.

        Invalid conditions are logged and silently dropped so one bad clause
        doesn't nuke the entire strategy.

        Args:
            raw: Unvalidated condition dicts.

        Returns:
            List of dicts that passed Pydantic validation.
        """
        validated: list[dict] = []
        for item in raw:
            try:
                condition = StrategyCondition(**item)
                validated.append(condition.model_dump(exclude_none=True))
            except Exception:
                logger.warning("Dropped invalid condition: %s", item, exc_info=True)
        return validated


__all__ = ["StrategyParser", "StrategyCondition"]
