# AURUM ⚡ Gold Strategy Evaluator

**An AI-Powered Multi-Agent System for Evaluating Gold (XAUUSD) Trading Strategies**

AURUM is an intelligent, multi-agent trading strategy evaluator designed specifically for the Gold (XAUUSD) market. It allows retail traders and students to input trading strategies in plain English or structured text, parses those conditions, and routes them to a team of specialist AI agents — each an expert on a specific technical indicator. 

The system runs a rigorous historical backtest of the strategy over 5 years, detects conflicting signals between indicators, and produces a comprehensive scorecard with actionable suggestions. 

![AURUM Dashboard Concept](https://img.shields.io/badge/Status-Active-success)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Key Features

1. **Natural Language Strategy Parsing**
   - Input your trading strategy in plain English (e.g., *"Buy Gold when RSI is below 35 and EMA20 crosses above EMA50"*).
   - A dual-mode parser uses Regex for lightning-fast standard parsing and falls back to an advanced AI (Claude) for complex sentences.
   
2. **Multi-Agent Orchestration**
   - **RSI Agent:** Momentum analysis and overbought/oversold detection.
   - **MACD Agent:** Fast/slow momentum crossover and divergence.
   - **EMA Agent:** Dynamic trend-following and crossover logic.
   - **SMA Agent:** Macro trend context.
   - **Bollinger Bands Agent:** Volatility squeeze and band proximity measurement.
   
3. **Intelligent Conflict Detection**
   - Flags when different indicators contradict each other (e.g., RSI signals a BUY due to oversold conditions, but MACD signals a SELL due to bearish divergence).

4. **Robust Vectorized Backtesting**
   - Powered by `vectorbt` for blisteringly fast simulation over 5+ years of daily Gold prices.
   - Generates exact metrics: Total Return, Win Rate, Sharpe Ratio, Profit Factor, and Max Drawdown.

5. **Interactive Dashboard**
   - Built with Streamlit, the dashboard provides a seamless 2-panel interface. 
   - Interactive candlestick charts and equity curves rendered with Plotly.

---

## 🛠️ Tech Stack

| Layer | Technologies | Use Case |
|-------|--------------|----------|
| **Core Parser** | Python `re`, Anthropic API | Regex + AI NLP processing |
| **Data Fetching** | `yfinance`, `requests` | Historical & real-time XAUUSD data |
| **Indicators** | `pandas-ta` | High-performance TA calculation |
| **Backtesting** | `vectorbt`, `pandas`, `numpy` | Vectorized strategy simulation |
| **Dashboard** | `streamlit`, `plotly` | UI & Interactive charting |
| **Validation** | `pydantic` | Strict typing and condition enforcement |

---

## 🚀 Installation

Ensure you have Python 3.10 or higher installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danishansari-dev/AURUM.git
   cd AURUM
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API Keys:**
   Create a `.env` file or export the following keys in your terminal.
   ```bash
   export ANTHROPIC_API_KEY="your_claude_api_key"        # (Required for AI parsing fallback)
   export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key" # (Optional: For real-time price fetching)
   ```

---

## ⚡ Usage

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

### How to use the Dashboard:

1. **Enter a Strategy:** Type your conditions into the text area. 
   - *Example:* `Buy when RSI is below 35 and EMA20 crosses above EMA50`
2. **Select Backtest Period:** Choose how many years of historical data to test against (1 to 5 years).
3. **Evaluate Strategy:** Click the evaluate button to trigger the pipeline.
4. **Analyze Results:**
   - **Scorecard:** View the weighted score from all agents on the left panel.
   - **Conflicts:** Read specific warnings if your indicators are contradicting each other.
   - **Charts:** View the interactive candlestick chart to see the entry conditions visually.
   - **Backtest Metrics:** Review Win Rate, Sharpe Ratio, and Equity Curve.
   - **Agent Reports:** Expand individual agent rows to see their exact feedback and improvement suggestions.

---

## 🏗️ Project Architecture

AURUM separates logic cleanly across multiple internal modules:

- `/agents`: Contains the Master `OrchestratorAgent` and all single-indicator specialist agents.
- `/backtest`: Contains the `vectorbt`-based vectorised historical simulation engine.
- `/data`: Modules for fetching, caching, and standardizing OHLCV price histories.
- `/indicators`: Applies Technical Analysis via `pandas-ta`.
- `/parser`: Holds the dual-mode string `StrategyParser` and Pydantic validation models.
- `/dashboard`: Contains the Streamlit visual presentation layer.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add a new Agent (e.g., Stochastic, Ichimoku, Volume Profile), please follow the `BaseAgent` abstraction in `agents/base_agent.py`.

1. Fork the repo
2. Create a new feature branch (`git checkout -b feature/Support-SuperTrend-Agent`)
3. Commit your changes (`git commit -m 'Added SuperTrend agent validation'`)
4. Push to the branch (`git push origin feature/Support-SuperTrend-Agent`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
