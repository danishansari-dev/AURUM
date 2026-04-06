# Multi-Agent Gold Trading Strategy Evaluator
## Complete Project Documentation

**Project Title:** AURUM — AI-Powered Multi-Agent Gold Trading Strategy Evaluator  
**Course:** B.Tech Major Project (Final Semester)  
**Document Version:** 1.0  
**Date:** April 2026  
**Status:** Approved for Development  

---

---

# SECTION 1 — PROJECT REQUIREMENT DOCUMENT (PRD)

---

## 1.1 Project Overview

AURUM is an AI-powered multi-agent system that evaluates user-defined trading strategies for Gold (XAUUSD) using specialist intelligent agents — each trained on a specific technical indicator. The system accepts natural language trading conditions from users, routes them to the appropriate specialist agents, runs backtesting on historical Gold price data, and produces a detailed, explainable scorecard with suggestions for improvement.

---

## 1.2 Problem Statement

Retail traders and finance students often design trading strategies based on technical indicators such as RSI, EMA, MACD, and Bollinger Bands — but have no reliable, automated way to:

- Evaluate how good their strategy actually is
- Understand why a strategy succeeds or fails per indicator
- Get actionable improvement suggestions
- See real backtested results on Gold-specific historical data

Existing tools like TradingView provide backtesting but offer no intelligent, per-indicator explanation or conflict detection between signals.

---

## 1.3 Objectives

1. Build a multi-agent system where each agent is a domain expert on one technical indicator
2. Allow users to define trading strategies in plain English or structured form
3. Automatically parse, evaluate, and score the strategy from each agent's perspective
4. Detect conflicts between agents (e.g., RSI says BUY while MACD says SELL)
5. Run backtesting on 5 years of historical Gold price data
6. Present results in a clear, interactive dashboard with visual evidence

---

## 1.4 Scope

### In Scope
- Gold asset (XAUUSD) analysis only
- Technical indicators: RSI, SMA, EMA, MACD, Bollinger Bands, Volume
- Historical data: 2019–2024 (minimum 5 years)
- Near real-time data feed integration (15-minute delay via yfinance / Alpha Vantage)
- Strategy evaluation, scoring, conflict detection, and suggestions
- Backtesting engine with key performance metrics
- Streamlit-based interactive dashboard

### Out of Scope
- Live trade execution or brokerage integration
- Multi-asset support (only Gold in this version)
- Mobile application
- User account management or persistent strategy storage
- Options or derivatives strategies

---

## 1.5 Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | System shall accept trading strategy conditions via text input | HIGH |
| FR-02 | System shall parse conditions and extract indicator, operator, value, and action | HIGH |
| FR-03 | System shall route each parsed condition to the appropriate specialist agent | HIGH |
| FR-04 | Each agent shall evaluate the condition and return a score (0–100) with reasoning | HIGH |
| FR-05 | Orchestrator agent shall aggregate scores and produce a weighted final verdict | HIGH |
| FR-06 | System shall detect and flag conflicting signals between agents | HIGH |
| FR-07 | System shall run backtesting on the complete strategy over historical Gold data | HIGH |
| FR-08 | System shall display backtest metrics: Win Rate, Sharpe Ratio, Max Drawdown, Total Trades | HIGH |
| FR-09 | System shall generate improvement suggestions per agent | MEDIUM |
| FR-10 | System shall display live Gold price on the dashboard | MEDIUM |
| FR-11 | System shall visualize candlestick chart with indicator overlays | MEDIUM |
| FR-12 | System shall allow user to compare two strategies side by side | LOW |
| FR-13 | System shall export evaluation report as PDF | LOW |

---

## 1.6 Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | Strategy evaluation response time | < 15 seconds |
| NFR-02 | Backtesting execution time for 5 years data | < 30 seconds |
| NFR-03 | Dashboard load time | < 5 seconds |
| NFR-04 | System uptime during demonstration | 99.9% |
| NFR-05 | Code maintainability | Modular OOP architecture |
| NFR-06 | Historical data completeness | Minimum 5 years, no gaps > 3 consecutive days |

---

## 1.7 Users and Stakeholders

| Role | Description |
|---|---|
| Primary User | Retail trader or finance student who wants to evaluate their strategy |
| Evaluator | B.Tech faculty coordinator assessing the project |
| Developer | Final year B.Tech student building the system |

---

## 1.8 Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| Language | Python 3.10+ | Industry standard for ML and data science |
| Data (Historical) | yfinance | Free, reliable, 5+ years of OHLCV Gold data |
| Data (Real-time) | Alpha Vantage API / Metals-API | Free tier, real-time XAUUSD spot price |
| Indicators | pandas-ta | Comprehensive TA library, easy API |
| ML / Agents | scikit-learn, custom OOP classes | Win-rate classifiers per agent |
| NLP Parser | spaCy + Regex rules | Strategy text parsing |
| Backtesting | vectorbt | Fast, vectorized backtesting engine |
| Dashboard | Streamlit | Rapid UI, excellent for demos |
| Charting | Plotly | Interactive candlestick + indicator charts |
| Storage | SQLite + CSV | Lightweight, no server required |
| Scheduler | APScheduler | Auto-refresh real-time price every 15 min |
| Version Control | Git + GitHub | Collaboration and submission |

---

## 1.9 Assumptions and Constraints

**Assumptions:**
- Internet connectivity is available during demonstration
- Historical Gold data from yfinance is sufficiently accurate for backtesting
- Faculty demo environment is a standard laptop with Python 3.10+

**Constraints:**
- No paid APIs — all data sources must have a free tier
- No live trade execution (regulatory and complexity reasons)
- Project must be completable within one semester (10–12 weeks)

---

## 1.10 Deliverables

1. Complete source code repository on GitHub
2. Functional Streamlit dashboard (runnable locally)
3. This documentation (HLD + LLD + PRD)
4. Project report (IEEE format)
5. Live demonstration with real-time Gold price feed

---

---

# SECTION 2 — HIGH-LEVEL DESIGN (HLD)

---

## 2.1 System Overview

AURUM follows a layered, modular architecture inspired by a multi-expert consultation system. Each layer has a single responsibility, and data flows top-down from user input to final output.

**Core Design Philosophy:** Separation of Concerns — each agent owns one indicator domain, knows nothing about others, and communicates only through the Orchestrator.

---

## 2.2 High-Level Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                        PRESENTATION LAYER                        ║
║                    Streamlit Dashboard (UI)                      ║
║       [Live Chart] [Strategy Input] [Scorecard] [Backtest]       ║
╚═════════════════════════════┬════════════════════════════════════╝
                              │
╔═════════════════════════════▼════════════════════════════════════╗
║                      APPLICATION LAYER                           ║
║                                                                  ║
║  ┌─────────────────────┐      ┌─────────────────────────────┐   ║
║  │   Strategy Parser   │      │    Orchestrator Agent       │   ║
║  │   (NLP Engine)      │─────▶│    (Master Controller)      │   ║
║  └─────────────────────┘      └──────────┬──────────────────┘   ║
║                                          │                       ║
║              ┌───────────┬──────────┬────┴──────┬──────────┐    ║
║              ▼           ▼          ▼           ▼          ▼    ║
║          ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐ ║
║          │  RSI  │  │  EMA  │  │  SMA  │  │ MACD  │  │  BB   │ ║
║          │ Agent │  │ Agent │  │ Agent │  │ Agent │  │ Agent │ ║
║          └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘ ║
║              └──────────┴──────────┴───────────┴──────────┘    ║
║                                   │                             ║
║              ┌────────────────────▼──────────────────────┐      ║
║              │          Backtesting Engine                │      ║
║              │          Scoring & Verdict Engine          │      ║
║              └───────────────────────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════╝
                              │
╔═════════════════════════════▼════════════════════════════════════╗
║                         DATA LAYER                               ║
║                                                                  ║
║   ┌────────────────┐    ┌──────────────────┐   ┌─────────────┐  ║
║   │  yfinance API  │    │  Alpha Vantage   │   │  SQLite DB  │  ║
║   │ (Historical)   │    │  (Real-time)     │   │  (Cache)    │  ║
║   └────────────────┘    └──────────────────┘   └─────────────┘  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 2.3 Component Descriptions

### 2.3.1 Presentation Layer — Streamlit Dashboard
The user-facing interface. Responsible for accepting strategy input, displaying real-time Gold price, rendering evaluation scorecards, backtest charts, and agent feedback. No business logic lives here.

### 2.3.2 Strategy Parser (NLP Engine)
Converts free-text trading conditions into structured JSON conditions. Uses spaCy for entity recognition and regex rules for operator/threshold extraction. Output is a list of condition objects passed to the Orchestrator.

### 2.3.3 Orchestrator Agent (Master Controller)
The central coordinator. Receives parsed conditions, identifies which agents to invoke, dispatches conditions to specialist agents, collects responses, detects inter-agent conflicts, and calls the Scoring Engine for final verdict. Acts like the managing director of the entire agent team.

### 2.3.4 Specialist Agents (RSI, EMA, SMA, MACD, BB, Volume)
Each agent is a self-contained module responsible for one indicator. It:
- Understands the domain rules of its indicator
- Holds a pre-trained ML classifier (scikit-learn) trained on historical Gold data
- Evaluates the user's condition against both domain rules and ML win-rate data
- Returns a structured response: score, reasoning, historical win rate, suggestions

### 2.3.5 Backtesting Engine
Accepts the full strategy (all conditions combined with AND logic) and simulates trades on 5 years of historical Gold OHLCV data using vectorbt. Returns key performance metrics.

### 2.3.6 Scoring and Verdict Engine
Aggregates individual agent scores using predefined weights. Detects conflicts. Generates overall rating, conflict warnings, and strategy-level suggestions.

### 2.3.7 Data Layer
- **yfinance:** Pulls 5+ years of daily/hourly OHLCV Gold data (XAUUSD=X or GLD)
- **Alpha Vantage / Metals-API:** Real-time Gold spot price for live dashboard display
- **SQLite:** Caches historical data locally to avoid repeated API calls

---

## 2.4 Agent Weight Distribution

Agent weights reflect the relative reliability and significance of each indicator specifically for Gold (XAUUSD), based on academic literature and trading practice:

| Agent | Weight | Justification |
|---|---|---|
| RSI Agent | 25% | Gold is highly momentum-driven; RSI is most reliable signal |
| MACD Agent | 20% | Strong trend momentum confirmation for Gold |
| EMA Agent | 20% | Dynamic trend following; EMA crossovers are widely trusted |
| SMA Agent | 15% | Longer-term trend context |
| Bollinger Bands Agent | 12% | Volatility measurement; useful but secondary |
| Volume Agent | 8% | Volume data on spot Gold is less reliable than equities |

---

## 2.5 Data Flow Summary

```
User types strategy text
        ↓
Strategy Parser → Structured conditions JSON
        ↓
Orchestrator receives conditions
        ↓
Orchestrator dispatches to relevant agents
        ↓
Each agent evaluates + returns score + feedback
        ↓
Backtesting Engine runs strategy on 5yr data
        ↓
Scoring Engine computes weighted final score
        ↓
Conflict Detector flags disagreements
        ↓
Dashboard renders full report
```

---

## 2.6 Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Agent communication | Synchronous function calls | Simpler, no latency for BTech demo |
| ML model per agent | scikit-learn Logistic Regression | Interpretable, fast, sufficient accuracy |
| Backtesting library | vectorbt | 100x faster than backtrader for large datasets |
| UI framework | Streamlit | Zero frontend code, Python-only, impressive for demo |
| NLP approach | spaCy + Regex | No API dependency, works offline |

---

## 2.7 Scoring System Design

```
Final Score = Σ (Agent_Score_i × Agent_Weight_i)

Rating Scale:
  80–100 → EXCELLENT  — Battle-tested strategy
  65–79  → GOOD       — Solid, minor improvements needed
  50–64  → AVERAGE    — Works inconsistently
  35–49  → WEAK       — High false signal rate
  0–34   → POOR       — Do not trade with this strategy
```

---

## 2.8 Conflict Detection Logic

A conflict is flagged when:
- Agent A signals BUY direction but Agent B signals SELL direction for the same timeframe
- Example: RSI < 30 (oversold = bullish) AND MACD is bearish divergence (bearish = sell)
- Conflict severity: CRITICAL if both conflicting agents have weight ≥ 20%, WARNING otherwise

---

---

# SECTION 3 — LOW-LEVEL DESIGN (LLD)

---

## 3.1 Project Directory Structure

```
aurum/
├── main.py                        # Streamlit app entry point
├── requirements.txt               # All dependencies
├── config.py                      # API keys, constants, file paths
├── README.md
│
├── data/
│   ├── data_fetcher.py            # yfinance + Alpha Vantage API calls
│   ├── data_preprocessor.py       # Cleaning, OHLCV normalization
│   ├── indicator_calculator.py    # pandas-ta wrapper — all indicators
│   └── gold_data.db               # SQLite cache
│
├── agents/
│   ├── base_agent.py              # Abstract base class for all agents
│   ├── rsi_agent.py               # RSI specialist agent
│   ├── ema_agent.py               # EMA specialist agent
│   ├── sma_agent.py               # SMA specialist agent
│   ├── macd_agent.py              # MACD specialist agent
│   ├── bollinger_agent.py         # Bollinger Bands specialist agent
│   ├── volume_agent.py            # Volume specialist agent
│   └── orchestrator.py            # Master orchestrator agent
│
├── parser/
│   ├── strategy_parser.py         # NLP + regex condition extractor
│   └── condition_validator.py     # Validates parsed conditions
│
├── backtesting/
│   ├── backtest_engine.py         # vectorbt backtesting runner
│   └── metrics_calculator.py     # Win rate, Sharpe, drawdown, etc.
│
├── scoring/
│   ├── scoring_engine.py          # Weighted score aggregation
│   ├── conflict_detector.py       # Cross-agent conflict detection
│   └── verdict_generator.py       # Final rating + recommendations
│
├── ml/
│   ├── agent_trainer.py           # Trains ML classifier per agent
│   ├── win_rate_model.py          # Historical win-rate lookup model
│   └── models/                    # Saved .pkl model files per agent
│       ├── rsi_model.pkl
│       ├── ema_model.pkl
│       ├── macd_model.pkl
│       └── ...
│
├── dashboard/
│   ├── charts.py                  # Plotly chart builders
│   ├── scorecard_ui.py            # Scorecard rendering components
│   └── live_price_widget.py       # Real-time price display
│
└── utils/
    ├── logger.py                  # Logging configuration
    └── helpers.py                 # Shared utility functions
```

---

## 3.2 Data Layer — Detailed Design

### 3.2.1 `data_fetcher.py`

```python
class GoldDataFetcher:
    """
    Responsible for all external data retrieval.
    Acts as the single gateway between the system and external APIs.
    """

    GOLD_TICKER = "GC=F"              # Gold Futures on Yahoo Finance
    GOLD_TICKER_ALT = "XAUUSD=X"     # Forex Gold spot rate

    def fetch_historical_data(
        self,
        period: str = "5y",           # "1y", "2y", "5y", "max"
        interval: str = "1d"          # "1d", "1h", "15m"
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        [Date, Open, High, Low, Close, Volume]
        Caches result to SQLite to avoid repeated API calls.
        """

    def fetch_realtime_price(self) -> dict:
        """
        Tries Alpha Vantage first, falls back to yfinance.
        Returns: {"price": 2345.67, "change": +12.3, "change_pct": +0.53, "timestamp": ...}
        """

    def _cache_to_sqlite(self, df: pd.DataFrame, table_name: str) -> None:
        """Stores DataFrame to local SQLite database."""

    def _load_from_cache(self, table_name: str, max_age_hours: int = 12) -> pd.DataFrame:
        """Loads cached data if fresh enough, else returns None."""
```

### 3.2.2 `indicator_calculator.py`

```python
class IndicatorCalculator:
    """
    Computes all technical indicators on a given OHLCV DataFrame.
    Wraps pandas-ta for clean, consistent output.
    """

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds these columns to df and returns it:
        - RSI_14
        - SMA_20, SMA_50, SMA_200
        - EMA_9, EMA_20, EMA_50, EMA_200
        - MACD_line, MACD_signal, MACD_histogram
        - BB_upper, BB_middle, BB_lower, BB_bandwidth
        - Volume_SMA_20, Volume_ratio
        """

    def compute_rsi(self, df, period=14) -> pd.Series
    def compute_sma(self, df, period) -> pd.Series
    def compute_ema(self, df, period) -> pd.Series
    def compute_macd(self, df, fast=12, slow=26, signal=9) -> pd.DataFrame
    def compute_bollinger(self, df, period=20, std=2) -> pd.DataFrame
    def compute_volume_indicators(self, df) -> pd.DataFrame
```

---

## 3.3 Agent Layer — Detailed Design

### 3.3.1 `base_agent.py` — Abstract Base Class

```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents.
    Every agent MUST implement these methods.
    Think of this like a job description that every specialist must fulfill.
    """

    agent_name: str = "BaseAgent"
    indicator_name: str = ""
    weight: float = 0.0

    @abstractmethod
    def evaluate_condition(
        self,
        condition: dict,
        df_with_indicators: pd.DataFrame
    ) -> dict:
        """
        Main evaluation method.

        Args:
            condition: {
                "indicator": "RSI",
                "operator": "<",
                "value": 35,
                "action": "BUY"
            }
            df_with_indicators: Full historical DataFrame with all indicators computed

        Returns: {
            "agent": str,
            "score": float (0-100),
            "win_rate": float,
            "signal_count": int,       # How many times this condition triggered historically
            "feedback": list[str],     # Bullet points of reasoning
            "suggestions": list[str],  # Improvement tips
            "direction": "BUY"/"SELL"/"NEUTRAL"
        }
        """

    @abstractmethod
    def get_domain_rules(self) -> dict:
        """Returns the hardcoded domain knowledge for this indicator."""

    def calculate_historical_win_rate(
        self,
        condition: dict,
        df: pd.DataFrame,
        forward_periods: int = 5
    ) -> tuple[float, int]:
        """
        Common method inherited by all agents.
        Finds all historical points where condition was met.
        Checks if price moved in the predicted direction within forward_periods candles.
        Returns (win_rate, signal_count).
        """
        signals = self._find_signal_points(condition, df)
        if len(signals) == 0:
            return 0.0, 0

        wins = 0
        for idx in signals:
            if idx + forward_periods >= len(df):
                continue
            future_price = df['Close'].iloc[idx + forward_periods]
            entry_price = df['Close'].iloc[idx]

            if condition['action'] == 'BUY' and future_price > entry_price:
                wins += 1
            elif condition['action'] == 'SELL' and future_price < entry_price:
                wins += 1

        return wins / len(signals), len(signals)

    def _find_signal_points(self, condition: dict, df: pd.DataFrame) -> list[int]:
        """Finds row indices where the indicator condition is True."""
```

---

### 3.3.2 `rsi_agent.py` — RSI Specialist

```python
class RSIAgent(BaseAgent):

    agent_name = "RSI Agent"
    indicator_name = "RSI"
    weight = 0.25

    RSI_ZONES = {
        "BUY": {
            (0, 20):  {"score": 50, "label": "Extreme Oversold", "reliability": "Very High"},
            (20, 30): {"score": 40, "label": "Classic Oversold", "reliability": "High"},
            (30, 35): {"score": 28, "label": "Mild Oversold",    "reliability": "Medium"},
            (35, 40): {"score": 15, "label": "Neutral-Low",      "reliability": "Low"},
            (40, 100):{"score": 5,  "label": "Not Oversold",     "reliability": "Very Low"},
        },
        "SELL": {
            (80, 100):{"score": 50, "label": "Extreme Overbought","reliability": "Very High"},
            (70, 80): {"score": 40, "label": "Classic Overbought","reliability": "High"},
            (65, 70): {"score": 28, "label": "Mild Overbought",  "reliability": "Medium"},
            (60, 65): {"score": 15, "label": "Neutral-High",     "reliability": "Low"},
            (0, 60):  {"score": 5,  "label": "Not Overbought",   "reliability": "Very Low"},
        }
    }

    def evaluate_condition(self, condition, df):
        score = 0
        feedback = []
        suggestions = []
        action = condition['action']
        threshold = condition['value']

        # Step 1: Zone-based domain rule scoring (50 pts max)
        zone_score, zone_label, reliability = self._get_zone_score(threshold, action)
        score += zone_score
        feedback.append(f"Threshold {threshold} falls in '{zone_label}' zone — Reliability: {reliability}")

        # Step 2: ML-based historical win rate (40 pts max)
        win_rate, signal_count = self.calculate_historical_win_rate(condition, df)
        ml_score = win_rate * 40
        score += ml_score
        feedback.append(f"Historical Win Rate on Gold: {win_rate*100:.1f}% over {signal_count} signals (2019–2024)")

        # Step 3: Signal frequency check (10 pts max)
        frequency_score = self._evaluate_signal_frequency(signal_count, len(df))
        score += frequency_score
        if signal_count < 10:
            feedback.append(f"⚠️ Only {signal_count} signals in 5 years — strategy may be too rare to be reliable")
            suggestions.append("Consider relaxing RSI threshold to generate more trade opportunities")
        elif signal_count > 200:
            feedback.append(f"⚠️ {signal_count} signals in 5 years — strategy may be too frequent with low quality")
            suggestions.append("Consider tightening RSI threshold or adding a confirming indicator")
        else:
            feedback.append(f"✅ {signal_count} signals in 5 years — healthy signal frequency")

        # Step 4: Generate improvement suggestion based on score
        suggestions += self._generate_rsi_suggestions(threshold, action, score)

        return {
            "agent": self.agent_name,
            "score": min(round(score, 2), 100),
            "win_rate": round(win_rate * 100, 1),
            "signal_count": signal_count,
            "feedback": feedback,
            "suggestions": suggestions,
            "direction": action,
            "zone_label": zone_label
        }

    def _get_zone_score(self, threshold, action) -> tuple:
        zones = self.RSI_ZONES.get(action, {})
        for (low, high), info in zones.items():
            if low <= threshold < high:
                return info['score'], info['label'], info['reliability']
        return 0, "Unknown Zone", "Unknown"

    def _evaluate_signal_frequency(self, signal_count, total_bars) -> float:
        ideal_min, ideal_max = 20, 150
        if ideal_min <= signal_count <= ideal_max:
            return 10.0
        elif signal_count < ideal_min:
            return max(0, (signal_count / ideal_min) * 10)
        else:
            return max(0, 10 - ((signal_count - ideal_max) / 100))

    def _generate_rsi_suggestions(self, threshold, action, score) -> list:
        suggestions = []
        if score < 40:
            if action == 'BUY':
                suggestions.append(f"RSI {threshold} generates very weak buy signals on Gold. Try RSI < 30 for classic oversold.")
            else:
                suggestions.append(f"RSI {threshold} generates very weak sell signals on Gold. Try RSI > 70 for classic overbought.")
        elif score < 65:
            suggestions.append("Combine this RSI condition with an EMA trend filter to reduce false signals.")
        else:
            suggestions.append("Strong RSI condition. Ensure stop-loss is placed at recent swing low/high.")
        return suggestions

    def get_domain_rules(self):
        return {
            "oversold_classic": 30,
            "overbought_classic": 70,
            "neutral_low": 40,
            "neutral_high": 60,
            "period": 14,
            "best_for_gold": "Daily and 4H timeframes"
        }
```

---

### 3.3.3 `macd_agent.py` — MACD Specialist

```python
class MACDAgent(BaseAgent):

    agent_name = "MACD Agent"
    indicator_name = "MACD"
    weight = 0.20

    MACD_CONDITIONS = {
        "crossover_above": {
            "description": "MACD line crosses above Signal line",
            "reliability": "High",
            "base_score": 45
        },
        "crossover_below": {
            "description": "MACD line crosses below Signal line",
            "reliability": "High",
            "base_score": 45
        },
        "histogram_positive": {
            "description": "MACD histogram above zero",
            "reliability": "Medium",
            "base_score": 30
        },
        "histogram_negative": {
            "description": "MACD histogram below zero",
            "reliability": "Medium",
            "base_score": 30
        },
        "zero_cross_above": {
            "description": "MACD line crosses above zero line",
            "reliability": "Very High",
            "base_score": 50
        },
    }

    def evaluate_condition(self, condition, df):
        # Evaluates MACD crossover, histogram, divergence conditions
        # Detects bullish/bearish divergence with price (advanced feature)
        pass

    def _detect_divergence(self, df) -> dict:
        """
        Bullish divergence: Price makes lower low, MACD makes higher low → BUY signal
        Bearish divergence: Price makes higher high, MACD makes lower high → SELL signal
        Returns: {"type": "bullish"/"bearish"/"none", "strength": 0-1}
        """
        pass

    def get_domain_rules(self):
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "strongest_signal": "zero_line_crossover",
            "best_for_gold": "Daily timeframe"
        }
```

---

### 3.3.4 `ema_agent.py` — EMA Specialist

```python
class EMAAgent(BaseAgent):

    agent_name = "EMA Agent"
    indicator_name = "EMA"
    weight = 0.20

    KNOWN_CROSSOVER_COMBOS = {
        (9, 21):   {"name": "Fast Scalping Cross",   "reliability": "Medium", "base_score": 30},
        (20, 50):  {"name": "Short-Medium Cross",    "reliability": "High",   "base_score": 40},
        (50, 200): {"name": "Golden/Death Cross",    "reliability": "Very High","base_score": 50},
        (20, 200): {"name": "Long-term Bias Cross",  "reliability": "High",   "base_score": 42},
    }

    def evaluate_condition(self, condition, df):
        """
        Evaluates:
        1. Single EMA vs Price (price above/below EMA)
        2. EMA crossover (fast EMA vs slow EMA)
        3. EMA slope analysis (trend strength)
        """
        pass

    def _analyze_ema_slope(self, ema_series, lookback=5) -> dict:
        """
        Measures slope of EMA to determine trend strength.
        Steep upward slope = strong uptrend (good for BUY)
        Flat slope = sideways market = weak signal
        Returns: {"slope": float, "trend_strength": "strong"/"moderate"/"weak"/"flat"}
        """
        pass

    def get_domain_rules(self):
        return {
            "supported_periods": [9, 20, 21, 50, 100, 200],
            "golden_cross_periods": (50, 200),
            "best_for_gold": "Daily timeframe; EMA reacts faster than SMA to price changes"
        }
```

---

### 3.3.5 `orchestrator.py` — Master Orchestrator

```python
class OrchestratorAgent:
    """
    The central coordinator — manages all specialist agents.
    Analogy: Like the managing director who receives client requirements,
    delegates to department heads, collects their reports, and
    synthesizes the final executive summary.
    """

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {
            "RSI":     RSIAgent(),
            "EMA":     EMAAgent(),
            "SMA":     SMAAgent(),
            "MACD":    MACDAgent(),
            "BB":      BollingerAgent(),
            "VOLUME":  VolumeAgent(),
        }
        self.scoring_engine = ScoringEngine()
        self.conflict_detector = ConflictDetector()
        self.backtest_engine = BacktestEngine()

    def evaluate_strategy(
        self,
        parsed_conditions: list[dict],
        df: pd.DataFrame
    ) -> dict:
        """
        Main orchestration method. Full pipeline execution.

        Returns complete evaluation result:
        {
            "final_score": float,
            "rating": str,
            "agent_results": dict,
            "conflicts": list,
            "backtest": dict,
            "verdict": str,
            "overall_suggestions": list
        }
        """

        # Step 1: Route each condition to the right agent
        agent_results = {}
        for condition in parsed_conditions:
            indicator = condition['indicator'].upper()
            if indicator in self.agents:
                result = self.agents[indicator].evaluate_condition(condition, df)
                agent_results[indicator] = result

        # Step 2: Detect conflicts between agents
        conflicts = self.conflict_detector.detect(agent_results)

        # Step 3: Run backtesting on the full strategy
        backtest_result = self.backtest_engine.run(parsed_conditions, df)

        # Step 4: Compute final weighted score
        final_score = self.scoring_engine.compute(agent_results)

        # Step 5: Apply conflict penalty
        if any(c['severity'] == 'CRITICAL' for c in conflicts):
            final_score *= 0.75   # 25% penalty for critical conflicts

        # Step 6: Generate overall verdict
        verdict = self._generate_verdict(final_score, conflicts, agent_results, backtest_result)

        return {
            "final_score": round(final_score, 2),
            "rating": self._score_to_rating(final_score),
            "agent_results": agent_results,
            "conflicts": conflicts,
            "backtest": backtest_result,
            "verdict": verdict,
            "overall_suggestions": self._aggregate_suggestions(agent_results, conflicts)
        }

    def _score_to_rating(self, score: float) -> str:
        if score >= 80: return "🏆 EXCELLENT"
        elif score >= 65: return "✅ GOOD"
        elif score >= 50: return "🟡 AVERAGE"
        elif score >= 35: return "⚠️ WEAK"
        else: return "❌ POOR"

    def _generate_verdict(self, score, conflicts, agent_results, backtest) -> str:
        """Generates a human-readable paragraph verdict combining all signals."""
        pass

    def _aggregate_suggestions(self, agent_results, conflicts) -> list[str]:
        """Collects and deduplicates suggestions from all agents."""
        pass
```

---

## 3.4 Strategy Parser — Detailed Design

### 3.4.1 `strategy_parser.py`

```python
class StrategyParser:
    """
    Converts free-text trading strategy into structured condition list.

    Input:  "Buy Gold when RSI is below 35 and 20 EMA crosses above 50 EMA"
    Output: [
                {"indicator": "RSI",  "operator": "<",               "value": 35,  "action": "BUY"},
                {"indicator": "EMA",  "operator": "crossover_above", "fast": 20, "slow": 50, "action": "BUY"}
            ]
    """

    INDICATOR_KEYWORDS = {
        "RSI": ["rsi", "relative strength"],
        "EMA": ["ema", "exponential moving average"],
        "SMA": ["sma", "simple moving average", "ma"],
        "MACD": ["macd", "moving average convergence"],
        "BB":   ["bollinger", "bb", "bands"],
        "VOLUME": ["volume", "vol"],
    }

    OPERATOR_PATTERNS = {
        r"(below|under|less than|<|drops below)":         "<",
        r"(above|over|greater than|>|rises above)":       ">",
        r"(crosses above|cross above|crossover above)":   "crossover_above",
        r"(crosses below|cross below|crossover below)":   "crossover_below",
        r"(equals|equal to|=|at)":                        "==",
    }

    ACTION_KEYWORDS = {
        "BUY":  ["buy", "long", "enter long", "purchase"],
        "SELL": ["sell", "short", "enter short", "exit"],
    }

    def parse(self, strategy_text: str) -> list[dict]:
        """Main parse method. Returns list of condition dicts."""
        strategy_text = strategy_text.lower().strip()
        action = self._extract_action(strategy_text)
        conditions = self._extract_conditions(strategy_text, action)
        return self._validate_conditions(conditions)

    def _extract_action(self, text: str) -> str:
        for action, keywords in self.ACTION_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return action
        return "BUY"  # Default

    def _extract_conditions(self, text: str, action: str) -> list[dict]:
        conditions = []
        # Split on 'and' / 'AND' / 'also' / 'with'
        clauses = re.split(r'\band\b|\balso\b|\bwith\b', text)
        for clause in clauses:
            condition = self._parse_clause(clause.strip(), action)
            if condition:
                conditions.append(condition)
        return conditions

    def _parse_clause(self, clause: str, action: str) -> dict | None:
        """Parses a single condition clause into a structured dict."""
        indicator = self._identify_indicator(clause)
        if not indicator:
            return None
        operator = self._identify_operator(clause)
        value = self._extract_numeric_value(clause)
        periods = self._extract_periods(clause)
        return {
            "indicator": indicator,
            "operator": operator,
            "value": value,
            "action": action,
            **periods  # Adds fast/slow for crossovers, period for single indicators
        }

    def _identify_indicator(self, clause: str) -> str | None:
        for indicator, keywords in self.INDICATOR_KEYWORDS.items():
            if any(kw in clause for kw in keywords):
                return indicator
        return None

    def _extract_numeric_value(self, clause: str) -> float | None:
        numbers = re.findall(r'\b\d+\.?\d*\b', clause)
        return float(numbers[0]) if numbers else None

    def _extract_periods(self, clause: str) -> dict:
        numbers = re.findall(r'\b\d+\b', clause)
        if len(numbers) >= 2:
            return {"fast": int(numbers[0]), "slow": int(numbers[1])}
        elif len(numbers) == 1:
            return {"period": int(numbers[0])}
        return {}

    def _validate_conditions(self, conditions: list) -> list:
        """Removes invalid/incomplete conditions."""
        return [c for c in conditions if c.get('indicator') and c.get('operator')]
```

---

## 3.5 Backtesting Engine — Detailed Design

```python
class BacktestEngine:
    """
    Runs strategy simulation on historical Gold data.
    Uses vectorbt for high-performance vectorized backtesting.

    Analogy: Like running a flight simulator — test the strategy
    on past market conditions before risking real money.
    """

    INITIAL_CAPITAL = 10000     # USD
    COMMISSION = 0.001          # 0.1% per trade

    def run(self, conditions: list[dict], df: pd.DataFrame) -> dict:
        """
        Generates BUY/SELL signals from all conditions combined (AND logic).
        Runs simulation and returns performance metrics.
        """
        buy_signals  = self._generate_signals(conditions, df, 'BUY')
        sell_signals = self._generate_signals(conditions, df, 'SELL')

        # If no explicit SELL conditions, use opposite of BUY as exit
        if not sell_signals.any():
            sell_signals = ~buy_signals

        portfolio = vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=buy_signals,
            exits=sell_signals,
            init_cash=self.INITIAL_CAPITAL,
            fees=self.COMMISSION,
            freq='D'
        )

        return {
            "total_return_pct":   round(portfolio.total_return() * 100, 2),
            "win_rate":           round(portfolio.trades.win_rate() * 100, 1),
            "total_trades":       int(portfolio.trades.count()),
            "sharpe_ratio":       round(portfolio.sharpe_ratio(), 2),
            "max_drawdown_pct":   round(portfolio.max_drawdown() * 100, 2),
            "avg_trade_duration": str(portfolio.trades.duration.mean()),
            "profit_factor":      round(portfolio.trades.profit_factor(), 2),
            "final_value":        round(portfolio.final_value(), 2),
            "equity_curve":       portfolio.value().to_dict()  # For chart
        }

    def _generate_signals(self, conditions, df, action) -> pd.Series:
        """
        Generates boolean signal series.
        All conditions for the given action are AND-combined.
        """
        action_conditions = [c for c in conditions if c['action'] == action]
        if not action_conditions:
            return pd.Series(False, index=df.index)

        signal = pd.Series(True, index=df.index)
        for condition in action_conditions:
            indicator_signal = self._condition_to_signal(condition, df)
            signal = signal & indicator_signal

        return signal

    def _condition_to_signal(self, condition: dict, df: pd.DataFrame) -> pd.Series:
        """Converts a single condition dict to a boolean Series."""
        indicator = condition['indicator']
        operator  = condition['operator']
        value     = condition.get('value')
        col_map = {
            "RSI":  "RSI_14",
            "MACD": "MACD_histogram",
            "SMA":  f"SMA_{condition.get('period', 20)}",
            "EMA":  f"EMA_{condition.get('period', 20)}",
        }
        col = col_map.get(indicator)
        if col and col in df.columns and value is not None:
            if operator == "<":  return df[col] < value
            if operator == ">":  return df[col] > value
            if operator == "==": return df[col] == value
        return pd.Series(False, index=df.index)
```

---

## 3.6 Conflict Detector — Detailed Design

```python
class ConflictDetector:
    """
    Detects when agents give opposing directional signals.
    Analogy: Like a hospital second-opinion system — if two senior
    specialists give contradictory diagnoses, flag it as critical.
    """

    HIGH_WEIGHT_THRESHOLD = 0.20   # Agents with weight >= 20% are "senior"

    def detect(self, agent_results: dict) -> list[dict]:
        conflicts = []
        agents_list = list(agent_results.items())

        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                name_a, result_a = agents_list[i]
                name_b, result_b = agents_list[j]

                dir_a = result_a.get('direction')
                dir_b = result_b.get('direction')

                if dir_a and dir_b and dir_a != dir_b and "NEUTRAL" not in [dir_a, dir_b]:
                    weight_a = self._get_weight(name_a)
                    weight_b = self._get_weight(name_b)
                    severity = "CRITICAL" if (weight_a >= self.HIGH_WEIGHT_THRESHOLD
                                              and weight_b >= self.HIGH_WEIGHT_THRESHOLD) else "WARNING"
                    conflicts.append({
                        "agent_a":     name_a,
                        "agent_b":     name_b,
                        "direction_a": dir_a,
                        "direction_b": dir_b,
                        "severity":    severity,
                        "message":     (f"{name_a} signals {dir_a} but {name_b} signals {dir_b}. "
                                        f"Wait for alignment before entering trade.")
                    })

        return conflicts

    def _get_weight(self, agent_name: str) -> float:
        weights = {"RSI": 0.25, "MACD": 0.20, "EMA": 0.20, "SMA": 0.15, "BB": 0.12, "VOLUME": 0.08}
        return weights.get(agent_name, 0.10)
```

---

## 3.7 Streamlit Dashboard — Component Map

```python
# main.py — Entry point

def main():
    st.set_page_config(page_title="AURUM — Gold Strategy Evaluator",
                       page_icon="🏅", layout="wide")

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    timeframe = st.sidebar.selectbox("Backtest Timeframe", ["1y", "2y", "5y"])
    show_indicators = st.sidebar.multiselect("Show on Chart",
                       ["RSI", "MACD", "EMA_20", "EMA_50", "BB"])

    # ── Header ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🏅 AURUM — Gold Strategy Evaluator")
    with col2:
        # Live price widget — auto-refreshes every 15 min
        live_price_widget()
    with col3:
        st.metric("24h Change", "+$12.30", "+0.53%")

    # ── Strategy Input ────────────────────────────────────────────────
    st.subheader("📝 Enter Your Trading Strategy")
    strategy_text = st.text_area(
        "Describe your strategy in plain English:",
        placeholder="e.g. Buy Gold when RSI is below 35 and 20 EMA is above 50 EMA and MACD histogram is positive",
        height=100
    )

    if st.button("🔍 Evaluate Strategy", type="primary"):
        with st.spinner("Agents evaluating your strategy..."):
            result = run_full_evaluation(strategy_text, timeframe)
        render_results(result)

    # ── Results Section ───────────────────────────────────────────────
    # render_results() displays:
    # 1. Final score + rating (large metric card)
    # 2. Individual agent score cards in columns
    # 3. Conflict warning boxes (if any)
    # 4. Backtest metrics table
    # 5. Equity curve chart (Plotly)
    # 6. Candlestick chart with indicator overlays
    # 7. Suggestions list
```

---

## 3.8 ML Training Pipeline

```python
class AgentTrainer:
    """
    Trains a simple win-rate ML model for each agent.
    This is done ONCE offline and models saved as .pkl files.
    During evaluation, agents load pre-trained models for instant prediction.
    """

    def train_all_agents(self, df_with_indicators: pd.DataFrame):
        """Trains and saves models for all agents."""
        self.train_rsi_model(df_with_indicators)
        self.train_macd_model(df_with_indicators)
        self.train_ema_model(df_with_indicators)
        # ... etc

    def train_rsi_model(self, df: pd.DataFrame):
        """
        Features: [RSI_value, RSI_slope_3d, price_momentum_5d, volume_ratio]
        Target:   1 if price increased in next 5 days, 0 otherwise
        Model:    LogisticRegression (interpretable + fast)
        """
        X = df[['RSI_14', 'RSI_slope', 'price_momentum_5d', 'Volume_ratio']].dropna()
        y = (df['Close'].shift(-5) > df['Close']).astype(int).loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"RSI Model Accuracy: {accuracy:.2%}")

        joblib.dump(model, 'ml/models/rsi_model.pkl')
```

---

## 3.9 Database Schema

```sql
-- Historical Gold price cache
CREATE TABLE IF NOT EXISTS gold_ohlcv (
    date        TEXT PRIMARY KEY,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Pre-computed indicators cache
CREATE TABLE IF NOT EXISTS gold_indicators (
    date            TEXT PRIMARY KEY,
    rsi_14          REAL,
    sma_20          REAL,
    sma_50          REAL,
    sma_200         REAL,
    ema_20          REAL,
    ema_50          REAL,
    ema_200         REAL,
    macd_line       REAL,
    macd_signal     REAL,
    macd_histogram  REAL,
    bb_upper        REAL,
    bb_middle       REAL,
    bb_lower        REAL,
    volume_sma_20   REAL,
    FOREIGN KEY (date) REFERENCES gold_ohlcv(date)
);

-- Strategy evaluation history (optional logging)
CREATE TABLE IF NOT EXISTS evaluation_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_text   TEXT,
    final_score     REAL,
    rating          TEXT,
    win_rate        REAL,
    sharpe_ratio    REAL,
    evaluated_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3.10 Key Python Dependencies (`requirements.txt`)

```
# Data
yfinance==0.2.37
requests==2.31.0
pandas==2.1.4
numpy==1.26.2

# Technical Indicators
pandas-ta==0.3.14b0

# Machine Learning
scikit-learn==1.3.2
joblib==1.3.2

# Backtesting
vectorbt==0.26.1

# NLP / Parsing
spacy==3.7.2

# Dashboard & Charts
streamlit==1.31.0
plotly==5.18.0

# Scheduling
APScheduler==3.10.4

# Database
sqlalchemy==2.0.23

# Utilities
python-dotenv==1.0.0
loguru==0.7.2
```

---

## 3.11 Condition Object — Full Specification

```
Condition Schema:
{
    "indicator":  str,       # "RSI" | "EMA" | "SMA" | "MACD" | "BB" | "VOLUME"
    "operator":   str,       # "<" | ">" | "==" | "crossover_above" | "crossover_below"
                             # "histogram_positive" | "histogram_negative"
    "value":      float,     # Numeric threshold (e.g., 35 for RSI, 0 for MACD)
    "action":     str,       # "BUY" | "SELL"
    "period":     int,       # Optional — for single-period indicators (SMA_20, EMA_50)
    "fast":       int,       # Optional — for crossover (EMA_20 crosses EMA_50 → fast=20)
    "slow":       int,       # Optional — for crossover (EMA_20 crosses EMA_50 → slow=50)
    "raw_clause": str        # Original text clause for debugging
}
```

---

## 3.12 Error Handling Strategy

| Error Scenario | Handling |
|---|---|
| API rate limit exceeded | Fallback to SQLite cache; display "Using cached data" notice |
| Unrecognized indicator in strategy text | Skip condition, display "Could not parse: [clause]" warning |
| Insufficient historical data (< 100 bars) | Display error: "Not enough data to evaluate this condition" |
| Agent returns null result | Orchestrator skips that agent's score with warning |
| Backtest produces no trades | Display: "Strategy conditions too strict — 0 signals generated in 5 years" |
| Network timeout on real-time fetch | Display last known price with "Data delayed" label |

---

## 3.13 Project Milestones & Testing Checklist

### Unit Tests Required

- [ ] `StrategyParser` — test 20+ natural language inputs
- [ ] `RSIAgent.evaluate_condition` — test all RSI zones
- [ ] `OrchestratorAgent.evaluate_strategy` — end-to-end test
- [ ] `ConflictDetector.detect` — test conflict + no-conflict cases
- [ ] `BacktestEngine.run` — verify metrics against manual calculation
- [ ] `IndicatorCalculator.compute_all` — verify against known TA values

### Integration Tests Required

- [ ] Full pipeline: text input → parsed conditions → agent results → backtest → scorecard
- [ ] Real-time price feed integration
- [ ] SQLite cache read/write cycle
- [ ] Dashboard renders without errors for all result types

---

*End of Document — AURUM Multi-Agent Gold Trading Strategy Evaluator*  
*B.Tech Major Project Documentation v1.0*
