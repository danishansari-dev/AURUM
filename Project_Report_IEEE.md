# AURUM: An AI-Powered Multi-Agent System for Evaluating Gold (XAUUSD) Trading Strategies

**Author:** Danish Ansari  
**Course:** B.Tech Major Project (Final Semester)  
**Date:** April 2026

---

## Abstract
Retail traders and students often rely on technical indicators to design trading strategies for financial markets. However, evaluating the efficacy of these strategies, particularly on highly liquid and volatile assets like Gold (XAUUSD), is usually limited to simple backtesting without explainable reasoning. This paper presents **AURUM**, a multi-agent artificial intelligence system designed to intelligently evaluate, score, and backtest user-defined Gold trading strategies. Utilizing a composite NLP parsing engine, AURUM translates natural language strategies into structured rules, routes them to specialist AI agents (RSI, EMA, SMA, MACD, Bollinger Bands), detects inter-agent conflicts, and performs rigorous vectorized backtesting using `vectorbt`. The system provides near real-time visualization of strategy performance, offering an explainable scorecard and actionable improvement suggestions.

---

## 1. Introduction
The financial markets, specifically the spot Gold (XAUUSD) market, represent one of the most traded assets globally. Technical Analysis (TA), using indicators like the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Exponential Moving Average (EMA), remains the cornerstone of retail trading. 

Currently, traders have access to backtesting platforms such as TradingView. However, these platforms fall short in providing *explainability* and *conflict detection*. For instance, an RSI indicator may signal "oversold" (Buy), while a MACD indicator shows strong "bearish divergence" (Sell). Traditional platforms execute these conflicting rules blindly, leading to suboptimal trade management and confusion for novice traders.

To address this gap, we developed **AURUM**—a modular, multi-agent AI framework where each technical indicator is represented by an independent "Specialist Agent". The system parses natural language inputs, aggregates the verdicts of each agent using a weighted scoring engine, detects logic conflicts, and returns a comprehensive, explainable trading scorecard.

---

## 2. System Architecture

AURUM follows a layered, modular architecture based on the Multi-Agent System (MAS) paradigm. Each layer encapsulates a specific responsibility:

### 2.1 Presentation Layer (Streamlit)
The user interface is built using Streamlit, providing an interactive, web-based dashboard. It accepts multi-condition strategy inputs via text, fetches real-time Gold spot prices (using the Alpha Vantage or yfinance APIs), and renders evaluation reports. Data visualization is powered by Plotly, generating dynamic candlestick charts with overlaid indicators and equity curves from backtesting.

### 2.2 NLP Parsing Engine
AURUM employs a robust, dual-mode parsing engine:
1. **Regex-First Approach:** High-speed pattern matching detects standard indicator states and crossovers (e.g., "EMA20 crosses above EMA50", "RSI below 30").
2. **AI-Fallback:** For complex, unstructured natural language queries, the engine falls back to an Anthropic LLM (Claude) to map semantic requests into strict `StrategyCondition` JSON structures validated by Pydantic.

### 2.3 Orchestrator Agent (Master Controller)
The core of the system is the `OrchestratorAgent`. It serves as the master dispatcher, routing parsed conditions to their respective Specialist Agents. It performs three critical operations:
- **Routing:** Identifying the correct agent (e.g., RSI) based on the condition's target indicator.
- **Conflict Detection:** Flagging inter-agent contradictions (e.g., Action Alignment, Trend-Momentum clashes).
- **Weighted Scoring:** Aggregating the individual agent scores based on an empirically derived weighting system specific to Gold's market behavior.

### 2.4 Specialist Agents
The Specialist Agents encapsulate domain knowledge for specific technical indicators. Each specialist (RSI, MACD, EMA, SMA, Bollinger Bands) conforms to an Abstract Base Class (`BaseAgent`). Upon receiving a condition, the agent evaluates it using:
1. **Domain Logic:** Hardcoded rules (e.g., an RSI of 10 is an extreme oversold state).
2. **Historical Validation:** A dynamic lookup against 5 years of historical Gold OHLCV data to compute the immediate historical win rate (`calculate_win_rate`).

---

## 3. Implementation and Methodology

### 3.1 Weighted Evaluation Scheme
Agent weights reflect the relative reliability of each indicator when traded on XAUUSD, determined via literature review and historical testing:
- **RSI Agent:** 25% (Gold is highly momentum-driven).
- **EMA Agent:** 22% (Dynamic trend following).
- **MACD Agent:** 20% (Momentum confirmation).
- **Bollinger Bands Agent:** 18% (Volatility squeezing).
- **SMA Agent:** 15% (Macro trend mapping).

The final score is normalized based on the participating agents, ensuring that strategies using fewer variables are not artificially down-scaled. The system then translates the score (0-100) into a 5-tier rating (EXCELLENT to POOR).

### 3.2 Vectorized Backtesting
To ensure scalability and performance, AURUM utilizes `vectorbt`. Instead of iterating bar-by-bar, conditions are aggregated into continuous boolean arrays (`AND` logic masks). The engine mimics non-overlapping trades, assuming a fixed forward-holding period and a standard commission slice. Metrics such as Sharpe Ratio, Win Rate, Max Drawdown, and absolute Profit Factor are reported instantaneously alongside an interactive equity curve chart.

---

## 4. Results and Discussion

The implemented framework successfully evaluates standard and complex multi-indicator strategies within seconds. 

During testing, standard strategies like the "Golden Cross" (SMA 50 crossing above SMA 200) were correctly identified by the SMA Agent as historically high-probability events, yielding scores in the GOOD to EXCELLENT band. Conversely, contrarian strategies input by the user (e.g., "Buy when RSI > 80") were appropriately penalized by the RSI Agent due to extreme overbought domain rules and subsequently generated low historical win rates, leading to a WEAK final rating and actionable improvement suggestions.

Furthermore, the Conflict Detection engine successfully identified hidden strategy flaws. When tested with a strategy combining "RSI below 30" (Buy) and "MACD bearish crossover" (Sell), the Orchestrator intercepted the conflict, downgraded the final score, and issued a CRITICAL warning to the user on the dashboard, explaining the contradiction.

---

## 5. Conclusion and Future Work
AURUM effectively bridges the gap between raw backtesting math and explainable trading logic. By translating trading rules into an orchestra of domain-expert AI agents, retail traders can gain immediate, qualitative, and quantitative feedback on their strategies.

**Future Enhancements include:**
1. **Machine Learning Classifier Models:** Upgrading the real-time historical scan with pre-trained Logistic Regression or Gradient Boosting models (`agent_trainer.py`) to classify indicator feature sets per agent.
2. **Multi-Asset Support:** Expanding beyond Gold (XAUUSD) to indices (e.g., S&P 500, NASDAQ) and analyzing context-dependent weight shifting.
3. **Automated Optimization:** Allowing the Orchestrator to automatically iteratively optimize user parameters (e.g., changing an RSI 30 rule to RSI 28) using Optuna to maximize the Sharpe ratio.

---

## References
1. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.
2. Polakow, O. (2020). *Vectorbt: Fast, vectorized backtesting in Python*. GitHub Repository.
3. Anthropic (2024). *Claude API Documentation* - Advanced natural language reasoning.
4. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
