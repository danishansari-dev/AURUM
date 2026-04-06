"""
AURUM ⚡ Gold Strategy Evaluator — Streamlit Dashboard

Launch:
    streamlit run dashboard/app.py

Replit:
    Starts automatically via .replit run command on port 8501.

The dashboard wires together every layer of the AURUM pipeline:
    yfinance → indicators → parser → orchestrator → backtest → visualisation

Optimised for Replit free-tier:
    - Lazy imports inside functions to reduce cold-start memory
    - API keys read exclusively from os.environ (Replit Secrets or .env)
    - Health check sidebar for live diagnostics
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Ensure the project root is on sys.path so absolute imports work
# when Streamlit runs this file from any working directory.
# ------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════

_GOLD = "#FFD700"
_GOLD_DIM = "#B8860B"
_BG_DARK = "#0D1117"
_BG_CARD = "#161B22"
_GREEN = "#00FF88"
_RED = "#FF4444"
_TEXT = "#E6EDF3"
_TEXT_DIM = "#8B949E"
_ACCENT_BLUE = "#58A6FF"

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AURUM — Gold Strategy Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    /* ── Global ──────────────────────────────────────────────── */
    .stApp {{
        background-color: {_BG_DARK};
        color: {_TEXT};
    }}
    header[data-testid="stHeader"] {{
        background-color: {_BG_DARK};
    }}

    /* ── Metric cards ────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {_BG_CARD};
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {_TEXT_DIM} !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {_GOLD} !important;
        font-weight: 700 !important;
    }}

    /* ── Tabs ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {_BG_CARD};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {_TEXT_DIM};
        border-radius: 8px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: {_GOLD} !important;
        background: rgba(255, 215, 0, 0.08) !important;
    }}

    /* ── Expanders ───────────────────────────────────────────── */
    details[data-testid="stExpander"] {{
        background: {_BG_CARD};
        border: 1px solid #30363d;
        border-radius: 12px;
    }}

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {_GOLD}, {_GOLD_DIM});
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.35);
    }}

    /* ── Text area ───────────────────────────────────────────── */
    .stTextArea textarea {{
        background: {_BG_CARD} !important;
        color: {_TEXT} !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }}

    /* ── Slider ──────────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] {{
        padding-top: 0 !important;
    }}

    /* ── Header banner ───────────────────────────────────────── */
    .aurum-header {{
        background: linear-gradient(135deg, rgba(255,215,0,0.08), rgba(184,134,11,0.05));
        border: 1px solid rgba(255,215,0,0.15);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
    }}
    .aurum-title {{
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, {_GOLD}, #FFF8DC, {_GOLD});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .aurum-subtitle {{
        color: {_TEXT_DIM};
        font-size: 1rem;
        margin-top: 4px;
    }}
    .price-badge {{
        display: inline-block;
        background: {_BG_CARD};
        border: 1px solid {_GOLD_DIM};
        border-radius: 8px;
        padding: 6px 16px;
        color: {_GOLD};
        font-weight: 700;
        font-size: 1.15rem;
        margin-top: 8px;
    }}

    /* ── Score ring ───────────────────────────────────────────── */
    .score-ring {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        border: 4px solid {_GOLD};
        margin: 0 auto 8px;
        font-size: 2rem;
        font-weight: 800;
        color: {_GOLD};
        background: rgba(255,215,0,0.06);
    }}
    .rating-label {{
        text-align: center;
        font-size: 0.95rem;
        color: {_TEXT};
        font-weight: 600;
    }}

    /* ── Conflict box ────────────────────────────────────────── */
    .conflict-box {{
        background: rgba(255, 68, 68, 0.08);
        border-left: 4px solid {_RED};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: {_TEXT};
    }}
    .conflict-box.medium {{
        border-left-color: {_GOLD};
        background: rgba(255, 215, 0, 0.06);
    }}
    .conflict-box.low {{
        border-left-color: {_ACCENT_BLUE};
        background: rgba(88, 166, 255, 0.06);
    }}
    .severity-badge {{
        display: inline-block;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }}
    .severity-HIGH {{ background: {_RED}; color: #fff; }}
    .severity-MEDIUM {{ background: {_GOLD}; color: #000; }}
    .severity-LOW {{ background: {_ACCENT_BLUE}; color: #000; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# LAZY IMPORT HELPERS (reduce cold‑start memory on Replit free tier)
# ══════════════════════════════════════════════════════════════════════


def _get_yfinance():
    """Lazy-import yfinance only when data fetching is needed."""
    import yfinance as yf
    return yf


def _get_plotly():
    """Lazy-import plotly.graph_objects for chart building."""
    import plotly.graph_objects as go
    return go


def _get_pandas():
    """Lazy-import pandas."""
    import pandas as pd
    return pd


def _get_orchestrator():
    """Lazy-import OrchestratorAgent to avoid loading all agents on boot."""
    from agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent


def _get_backtest_engine():
    """Lazy-import BacktestEngine to defer numpy/vectorbt loading."""
    from backtest.engine import BacktestEngine
    return BacktestEngine


def _get_parser():
    """Lazy-import StrategyParser."""
    from parser.strategy_parser import StrategyParser
    return StrategyParser


def _get_fetcher():
    """Lazy-import the data fetcher function."""
    from data.fetcher import fetch_xauusd_history
    return fetch_xauusd_history


def _get_indicator_calculator():
    """Lazy-import the indicator calculator function."""
    from indicators.calculator import add_all_indicators
    return add_all_indicators


# ══════════════════════════════════════════════════════════════════════
# DATA HELPERS (cached)
# ══════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_gold_price() -> tuple[float | None, str]:
    """
    Grab the latest Gold Futures close from yfinance.

    Cached for 15 minutes to avoid hammering the API on every rerun.
    Uses GC=F (Gold Futures) which is more reliably available than XAUUSD=X.

    Returns:
        Tuple of (price_or_None, iso_timestamp_string).
    """
    yf = _get_yfinance()
    ts = datetime.now(timezone.utc).isoformat()
    try:
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return None, ts
        return float(hist["Close"].iloc[-1]), ts
    except Exception:
        return None, ts


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(period: str):
    """
    Fetch XAUUSD history via the project's own data fetcher.

    Falls back to GC=F directly if the primary symbol fails.
    Cached for 1 hour per period value.
    """
    pd = _get_pandas()
    yf = _get_yfinance()
    fetch_xauusd = _get_fetcher()

    try:
        return fetch_xauusd(period=period)
    except RuntimeError:
        # Fallback to Gold Futures
        ticker = yf.Ticker("GC=F")
        raw = ticker.history(period=period, interval="1d")
        raw.columns = [str(c).lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close", "volume"]]
        raw.index.name = "date"
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        return raw


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_indicators(_df):
    """Run add_all_indicators with caching (underscore prefix avoids hash issues)."""
    add_all = _get_indicator_calculator()
    return add_all(_df)


# ══════════════════════════════════════════════════════════════════════
# HEALTH CHECK SIDEBAR
# ══════════════════════════════════════════════════════════════════════


def _render_health_check() -> None:
    """
    Sidebar diagnostics panel showing system health:
      - Last data fetch timestamp
      - API connection status (green/red indicators)
      - Current Gold spot price
    """
    with st.sidebar:
        st.markdown("## 🩺 System Health")
        st.markdown("---")

        # --- Gold price & fetch time ---
        gold_price, fetch_ts = _fetch_gold_price()

        st.markdown("**Last Data Fetch**")
        try:
            dt = datetime.fromisoformat(fetch_ts)
            st.caption(dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
        except (ValueError, TypeError):
            st.caption(fetch_ts)

        st.markdown("**Gold Spot Price**")
        if gold_price is not None:
            st.metric("GC=F", f"${gold_price:,.2f}")
        else:
            st.error("Unavailable")

        st.markdown("---")

        # --- API connection status ---
        st.markdown("**API Connections**")

        # yfinance (always available — no key)
        yf_status = gold_price is not None
        if yf_status:
            st.markdown("🟢 **yfinance** — Connected")
        else:
            st.markdown("🔴 **yfinance** — Disconnected")

        # Alpha Vantage
        av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if av_key:
            st.markdown("🟢 **Alpha Vantage** — Key configured")
        else:
            st.markdown("🟡 **Alpha Vantage** — No key (optional)")

        # Anthropic (Claude)
        anth_key = os.environ.get("ANTHROPIC_API_KEY")
        if anth_key:
            st.markdown("🟢 **Anthropic (Claude)** — Key configured")
        else:
            st.markdown("🟡 **Anthropic** — No key (regex-only mode)")

        st.markdown("---")

        # --- Environment info ---
        st.markdown("**Environment**")
        import platform
        st.caption(f"Python {platform.python_version()}")
        st.caption(f"OS: {platform.system()} {platform.machine()}")

        # Detect Replit
        if os.environ.get("REPL_ID"):
            st.markdown("🟢 Running on **Replit**")
        else:
            st.markdown("🖥️ Running **locally**")


# ══════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════


def _build_candlestick(df, n_days: int = 90):
    """
    Build a themed candlestick chart with EMA20 / EMA50 overlays.

    Shows the most recent *n_days* bars.

    Args:
        df: OHLCV + indicator DataFrame.
        n_days: Number of trailing days to display.

    Returns:
        Plotly Figure ready for ``st.plotly_chart``.
    """
    go = _get_plotly()
    tail = df.tail(n_days).copy()

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["open"],
            high=tail["high"],
            low=tail["low"],
            close=tail["close"],
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            increasing_fillcolor=_GREEN,
            decreasing_fillcolor=_RED,
            name="XAUUSD",
        )
    )

    # EMA overlays
    if "ema20" in tail.columns:
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail["ema20"],
                mode="lines",
                name="EMA 20",
                line=dict(color=_GOLD, width=1.5),
            )
        )
    if "ema50" in tail.columns:
        fig.add_trace(
            go.Scatter(
                x=tail.index,
                y=tail["ema50"],
                mode="lines",
                name="EMA 50",
                line=dict(color=_ACCENT_BLUE, width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_BG_DARK,
        font_color=_TEXT,
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Price (USD)"),
    )
    return fig


def _build_equity_chart(equity_curve: dict[str, float]):
    """
    Build a themed equity-curve line chart from the backtest engine output.

    Args:
        equity_curve: ``{date_string: equity_value}`` mapping.

    Returns:
        Plotly Figure.
    """
    go = _get_plotly()

    if not equity_curve:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG_DARK,
            plot_bgcolor=_BG_DARK,
            annotations=[
                dict(
                    text="No trades generated",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color=_TEXT_DIM),
                    x=0.5,
                    y=0.5,
                )
            ],
            height=350,
        )
        return fig

    dates = list(equity_curve.keys())
    values = list(equity_curve.values())

    # Determine gain/loss colour
    line_colour = _GREEN if values[-1] >= values[0] else _RED

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Equity",
            line=dict(color=line_colour, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(line_colour.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.08)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_BG_DARK,
        font_color=_TEXT,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="#21262d", title="Date"),
        yaxis=dict(gridcolor="#21262d", title="Equity (USD)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════


def _render_conflict(conflict: dict) -> None:
    """Render a single conflict as a styled HTML box."""
    severity = conflict.get("severity", "LOW")
    css_class = "medium" if severity == "MEDIUM" else ("low" if severity == "LOW" else "")
    agents = ", ".join(conflict.get("agents_involved", []))
    desc = conflict.get("description", "")
    st.markdown(
        f"""
        <div class="conflict-box {css_class}">
            <span class="severity-badge severity-{severity}">{severity}</span>
            <strong>{agents}</strong><br/>
            <span style="font-size:0.9rem;">{desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry-point: assemble the full AURUM dashboard."""

    # ── Health check sidebar (always rendered) ───────────────────────
    _render_health_check()

    # ── Header ───────────────────────────────────────────────────────
    gold_price, _ = _fetch_gold_price()
    price_str = f"${gold_price:,.2f}" if gold_price else "unavailable"

    st.markdown(
        f"""
        <div class="aurum-header">
            <h1 class="aurum-title">AURUM ⚡ Gold Strategy Evaluator</h1>
            <p class="aurum-subtitle">
                Multi-agent AI system for evaluating, scoring, and backtesting
                Gold (XAUUSD) trading strategies
            </p>
            <span class="price-badge">🪙 Live Gold &nbsp;{price_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-panel layout ─────────────────────────────────────────────
    col_left, col_right = st.columns([2, 3], gap="large")

    # ══════════════════════════════════════════════════════════════════
    # LEFT PANEL — Strategy Input
    # ══════════════════════════════════════════════════════════════════
    with col_left:
        st.markdown("### 📝 Strategy Input")

        strategy_text = st.text_area(
            label="Enter your Gold trading strategy",
            placeholder=(
                "Buy Gold when RSI is below 35 and EMA20 crosses above "
                "EMA50 and MACD bullish crossover"
            ),
            height=130,
            label_visibility="collapsed",
        )

        period_years = st.slider(
            "Backtest period (years)",
            min_value=1,
            max_value=5,
            value=3,
            help="Historical lookback window for evaluation and backtesting.",
        )

        evaluate_btn = st.button("⚡  Evaluate Strategy", use_container_width=True)

        # ── Trigger full pipeline ────────────────────────────────────
        if evaluate_btn and strategy_text.strip():
            _run_pipeline(strategy_text, period_years, col_left, col_right)
        elif evaluate_btn and not strategy_text.strip():
            st.warning("Please enter a strategy before evaluating.")
        else:
            # Show the right panel with a default chart when no evaluation
            with col_right:
                _render_default_chart()


def _run_pipeline(
    strategy_text: str,
    period_years: int,
    col_left,
    col_right,
) -> None:
    """
    Execute the full AURUM pipeline and render results.

    Steps:
        1. Fetch OHLCV data via yfinance
        2. Compute technical indicators
        3. Parse strategy text → conditions list
        4. Orchestrator evaluates conditions against data (parallel fan-out)
        5. Backtest engine simulates trades
        6. Render everything in the dashboard panels
    """
    period_map = {1: "1y", 2: "2y", 3: "3y", 4: "4y", 5: "5y"}
    period_str = period_map.get(period_years, "3y")

    # ── Step 1 & 2: Fetch + Indicators (cached) ─────────────────────
    with st.spinner("📡 Fetching Gold price data…"):
        try:
            df_raw = _fetch_history(period_str)
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
            return

    with st.spinner("📊 Computing indicators…"):
        try:
            df = _compute_indicators(df_raw)
        except Exception as exc:
            st.error(f"Indicator computation failed: {exc}")
            return

    # ── Step 3: Parse strategy (lazy import) ─────────────────────────
    with st.spinner("🧠 Parsing strategy…"):
        StrategyParser = _get_parser()
        parser = StrategyParser()
        conditions = parser.parse_regex(strategy_text)

    if not conditions:
        st.error(
            "Could not parse any trading conditions from your input. "
            "Try: *Buy when RSI below 30 and EMA20 crosses above EMA50*"
        )
        return

    # ── Step 4: Orchestrator evaluation (lazy import + ThreadPoolExecutor) ──
    with st.spinner("🤖 Agents evaluating strategy…"):
        OrchestratorAgent = _get_orchestrator()
        orchestrator = OrchestratorAgent()
        evaluation = orchestrator.evaluate_strategy(conditions, df)

    # ── Step 5: Backtest (lazy import) ───────────────────────────────
    with st.spinner("⏳ Running backtest…"):
        BacktestEngine = _get_backtest_engine()
        engine = BacktestEngine()
        backtest = engine.run(conditions, df)

    # Store results in session state for persistence across reruns
    st.session_state["evaluation"] = evaluation
    st.session_state["backtest"] = backtest
    st.session_state["df"] = df
    st.session_state["conditions"] = conditions

    # ── Render results ───────────────────────────────────────────────
    _render_left_results(col_left, evaluation)
    _render_right_results(col_right, df, evaluation, backtest)


# ══════════════════════════════════════════════════════════════════════
# LEFT PANEL RESULTS
# ══════════════════════════════════════════════════════════════════════


def _render_left_results(col, evaluation: dict) -> None:
    """Render score ring, agent cards, and conflict warnings in the left panel."""
    with col:
        st.markdown("---")

        # ── Final score ring ─────────────────────────────────────────
        final_score = evaluation.get("final_score", 0.0)
        rating = evaluation.get("rating", "")
        st.markdown(
            f"""
            <div class="score-ring">{final_score:.0f}</div>
            <div class="rating-label">{rating}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ── Agent scorecard (metric cards) ───────────────────────────
        results = evaluation.get("individual_results", {})
        if results:
            st.markdown("#### Agent Scores")
            agent_names = list(results.keys())

            # Lay out in rows of 3
            for row_start in range(0, len(agent_names), 3):
                row_agents = agent_names[row_start : row_start + 3]
                metric_cols = st.columns(len(row_agents))
                for mc, name in zip(metric_cols, row_agents):
                    r = results[name]
                    with mc:
                        st.metric(
                            label=f"{name} Agent",
                            value=f"{r.score:.0f}/100",
                            delta=f"{r.win_rate:.0%} win rate",
                        )

        # ── Conflicts ────────────────────────────────────────────────
        conflicts = evaluation.get("conflicts", [])
        if conflicts:
            st.markdown("#### ⚠️ Conflicts Detected")
            for c in conflicts:
                _render_conflict(c)


# ══════════════════════════════════════════════════════════════════════
# RIGHT PANEL RESULTS
# ══════════════════════════════════════════════════════════════════════


def _render_right_results(col, df, evaluation: dict, backtest: dict) -> None:
    """Render tabs: Price Chart, Backtest Results, Agent Reports."""
    with col:
        tab_chart, tab_backtest, tab_reports = st.tabs(
            ["📈 Price Chart", "🧪 Backtest Results", "🤖 Agent Reports"]
        )

        # ── Tab 1: Candlestick chart ─────────────────────────────────
        with tab_chart:
            fig = _build_candlestick(df, n_days=90)
            st.plotly_chart(fig, key="price_chart")

        # ── Tab 2: Backtest Results ──────────────────────────────────
        with tab_backtest:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                ret = backtest.get("total_return_pct", 0.0)
                st.metric("Total Return", f"{ret:+.2f}%", delta=None)
            with m2:
                st.metric("Win Rate", f"{backtest.get('win_rate', 0):.1f}%")
            with m3:
                st.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
            with m4:
                st.metric("Max Drawdown", f"-{backtest.get('max_drawdown_pct', 0):.2f}%")

            # Second row
            m5, m6, m7, _ = st.columns(4)
            with m5:
                st.metric("Total Trades", backtest.get("total_trades", 0))
            with m6:
                pf = backtest.get("profit_factor", 0)
                pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
                st.metric("Profit Factor", pf_str)
            with m7:
                st.metric("Final Value", f"${backtest.get('final_value', 10000):,.2f}")

            st.markdown("")
            st.markdown("##### Equity Curve")
            equity_fig = _build_equity_chart(backtest.get("equity_curve", {}))
            st.plotly_chart(equity_fig, key="equity_chart")

        # ── Tab 3: Agent Reports ─────────────────────────────────────
        with tab_reports:
            results = evaluation.get("individual_results", {})
            if not results:
                st.info("No agent results to display.")
            else:
                for name, result in results.items():
                    with st.expander(
                        f"{'🟢' if result.score >= 55 else '🔴'} {name} Agent — "
                        f"Score: {result.score:.0f}/100  |  "
                        f"Alignment: {result.action_alignment}",
                        expanded=False,
                    ):
                        # Feedback
                        st.markdown("**Feedback**")
                        for fb in result.feedback:
                            st.markdown(f"- {fb}")

                        # Suggestions
                        if result.suggestions:
                            st.markdown("**Suggestions**")
                            for sg in result.suggestions:
                                st.markdown(f"- 💡 {sg}")

                        st.caption(
                            f"Win Rate: {result.win_rate:.2%}  •  "
                            f"Action: {result.action_alignment}"
                        )

            # ── Aggregated Suggestions ───────────────────────────
            suggestions = evaluation.get("suggestions", [])
            if suggestions:
                st.markdown("---")
                st.markdown("#### 💡 Strategy Suggestions")
                for s in suggestions:
                    st.markdown(f"- {s}")


# ══════════════════════════════════════════════════════════════════════
# DEFAULT STATE (no evaluation yet)
# ══════════════════════════════════════════════════════════════════════


def _render_default_chart() -> None:
    """Show a candlestick chart before any evaluation is triggered."""
    st.markdown("### 📈 Gold Price — Last 90 Days")
    try:
        df_raw = _fetch_history("1y")
        df = _compute_indicators(df_raw)
        fig = _build_candlestick(df, n_days=90)
        st.plotly_chart(fig, key="default_chart")
    except Exception:
        st.info("Enter a strategy and click **Evaluate** to begin.")


# ══════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
else:
    # Streamlit imports and runs the module directly
    main()
