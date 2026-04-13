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

# ── Must run before ANY numpy/scipy import to avoid OpenBLAS crash on Windows ──
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Load .env file for local API keys (safe no-op if file missing) ──
try:
    from pathlib import Path as _P
    from dotenv import load_dotenv
    _env_path = _P(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed — keys must be set manually

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
import streamlit.components.v1 as components

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

    /* ── Score ring (legacy fallback for demo mode) ────────────── */
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

    /* ── Agent score cards (custom HTML, no truncation) ────────── */
    .agent-cards-row {{
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }}
    .agent-card {{
        flex: 1 1 180px;
        min-width: 180px;
        max-width: 260px;
        background: {_BG_CARD};
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 18px;
        overflow: visible;
        word-wrap: break-word;
    }}
    .agent-card-name {{
        color: {_TEXT_DIM};
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 6px;
    }}
    .agent-card-score {{
        font-size: 1.7rem;
        font-weight: 800;
        color: {_GOLD};
        margin-bottom: 6px;
    }}
    .agent-card-bar {{
        width: 100%;
        height: 6px;
        background: #21262d;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 8px;
    }}
    .agent-card-bar-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }}
    .agent-card-meta {{
        font-size: 0.8rem;
        color: {_TEXT_DIM};
        line-height: 1.5;
    }}
    .agent-card-meta strong {{
        color: {_TEXT};
        font-weight: 600;
    }}

    /* ── Action alignment badges ───────────────────────────────── */
    .action-badge {{
        display: inline-block;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 4px;
        letter-spacing: 0.3px;
    }}
    .action-badge.buy {{ background: rgba(0,255,136,0.15); color: {_GREEN}; }}
    .action-badge.sell {{ background: rgba(255,68,68,0.15); color: {_RED}; }}
    .action-badge.neutral {{ background: rgba(139,148,158,0.15); color: {_TEXT_DIM}; }}

    /* ── Backtest warning box ──────────────────────────────────── */
    .bt-warning-box {{
        background: rgba(255, 215, 0, 0.06);
        border: 1px solid rgba(255, 215, 0, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        color: {_TEXT};
        margin: 12px 0;
    }}
    .bt-warning-box h4 {{
        color: {_GOLD};
        margin: 0 0 8px;
    }}
    .bt-warning-box p {{
        color: {_TEXT_DIM};
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }}

    /* ── Streak info bar ──────────────────────────────────────── */
    .streak-bar {{
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        background: {_BG_CARD};
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 20px;
        margin: 10px 0 16px;
    }}
    .streak-item {{
        color: {_TEXT_DIM};
        font-size: 0.88rem;
    }}
    .streak-item strong {{
        color: {_TEXT};
        font-weight: 700;
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
def _fetch_gold_price():
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

        st.markdown("**Gold Spot Price (Live)**")
        components.html(
            """
            <div id="price-container">
                <span id="price-symbol">XAUUSD</span>
                <span id="price-value" class="text-neutral">Connecting...</span>
                <span id="price-arrow"></span>
            </div>
            <script>
                const priceValue = document.getElementById("price-value");
                const priceArrow = document.getElementById("price-arrow");
                let lastPrice = null;

                // Using Binance paxgusdt ticker as a highly reliable, free, zero-latency 
                // proxy for real-time XAUUSD spot prices without needing API keys.
                const ws = new WebSocket("wss://stream.binance.com:9443/ws/paxgusdt@ticker");
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    const currentPrice = parseFloat(data.c);
                    
                    if (lastPrice === null) {
                        lastPrice = currentPrice;
                        priceValue.textContent = "$" + currentPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                        return;
                    }
                    if (currentPrice === lastPrice) return;
                    
                    priceValue.textContent = "$" + currentPrice.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    
                    priceValue.classList.remove("flash-green", "flash-red", "text-up", "text-down", "text-neutral");
                    void priceValue.offsetWidth; // Force reflow to restart animation
                    
                    if (currentPrice > lastPrice) {
                        priceValue.classList.add("flash-green", "text-up");
                        priceArrow.textContent = "▲";
                        priceArrow.className = "arrow-up";
                    } else if (currentPrice < lastPrice) {
                        priceValue.classList.add("flash-red", "text-down");
                        priceArrow.textContent = "▼";
                        priceArrow.className = "arrow-down";
                    } else {
                        priceValue.classList.add("text-neutral");
                        priceArrow.textContent = "";
                    }
                    
                    lastPrice = currentPrice;
                };
            </script>
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                }
                #price-container {
                    background: #161B22; 
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 14px 20px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: #E6EDF3;
                    box-sizing: border-box;
                    width: 100%;
                }
                #price-symbol {
                    color: #8B949E;
                    font-size: 0.85rem;
                    font-weight: 500;
                    margin-right: auto;
                }
                #price-value {
                    font-size: 1.6rem;
                    font-weight: 700;
                    transition: color 0.8s ease-out;
                }
                #price-arrow {
                    font-size: 1.2rem;
                    font-weight: 700;
                }
                
                .arrow-up { color: #00FF88; }
                .arrow-down { color: #FF4444; }
                
                .text-up { color: #87e0b5 !important; }
                .text-down { color: #e88b8b !important; }
                .text-neutral { color: #E6EDF3 !important; }
                
                .flash-green { animation: flashGreen 1.2s ease-out forwards; }
                .flash-red { animation: flashRed 1.2s ease-out forwards; }
                
                @keyframes flashGreen {
                    0% { color: #00FF88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.6); }
                    100% { color: #87e0b5; text-shadow: none; }
                }
                @keyframes flashRed {
                    0% { color: #FF4444; text-shadow: 0 0 10px rgba(255, 68, 68, 0.6); }
                    100% { color: #e88b8b; text-shadow: none; }
                }
            </style>
            """,
            height=85,
        )

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


def _build_equity_chart(equity_curve):
    """
    Build a themed equity-curve line chart from the backtest engine output.

    Accepts either a dict ``{date_string: value}`` or a list ``[value, ...]``.
    The engine returns a list of cumulative PnL in pips; we plot it directly
    as pip-based PnL with a gold-coloured line and a zero reference line.

    Args:
        equity_curve: Cumulative PnL list (pips) or date-keyed dict.

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

    # Normalise: engine returns list[float] (cumulative pip PnL), dashboard
    # may also receive dict[str, float] from demo mode.
    if isinstance(equity_curve, dict):
        x_axis = list(equity_curve.keys())
        values = list(equity_curve.values())
        x_label = "Date"
    else:
        # List of cumulative pip PnL — plot as pips directly
        values = list(equity_curve)
        x_axis = list(range(1, len(values) + 1))
        x_label = "Trade #"

    # Determine gain/loss colour from final value
    line_colour = _GREEN if values[-1] >= 0 else _RED

    fig = go.Figure()

    # Horizontal zero reference line
    fig.add_hline(
        y=0, line_width=1, line_dash="dash",
        line_color="#484f58",
        annotation_text="Break-even",
        annotation_font_color=_TEXT_DIM,
        annotation_font_size=10,
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=values,
            mode="lines",
            name="Cumulative P&L",
            line=dict(color=_GOLD, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255, 215, 0, 0.06)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG_DARK,
        plot_bgcolor=_BG_DARK,
        font_color=_TEXT,
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="#21262d", title=x_label),
        yaxis=dict(gridcolor="#21262d", title="Cumulative P&L (pips)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════


def _render_score_gauge(score: float, rating: str) -> None:
    """
    Render an animated SVG arc gauge for the final strategy score.

    Uses stroke-dasharray/dashoffset technique to draw a partial arc.
    Colour gradient: red (<40), orange (<55), yellow (<70), green (>=70).
    Animated fill from 0 to final score over 1.2s on load.

    Args:
        score: Numeric score 0-100.
        rating: Human-readable rating string from the orchestrator.
    """
    # Choose arc colour based on score bands
    if score >= 70:
        arc_color = "#00FF88"
    elif score >= 55:
        arc_color = "#FFD700"
    elif score >= 40:
        arc_color = "#FF8C00"
    else:
        arc_color = "#FF4444"

    # SVG arc math: circumference of the gauge arc (270° of a circle, r=80)
    radius = 80
    circumference = 2 * 3.14159 * radius  # ~502.65
    arc_length = circumference * 0.75      # 270° arc = 75% of full circle
    fill_length = arc_length * (score / 100.0)
    gap = arc_length - fill_length

    # Strip emoji from rating for the inner label (emoji renders poorly in SVG)
    rating_short = rating.split("\u2014")[0].strip() if "\u2014" in rating else rating
    # Remove leading emoji characters
    for prefix in ["\U0001f3c6", "\u2705", "\U0001f7e1", "\u26a0\ufe0f", "\u274c"]:
        rating_short = rating_short.replace(prefix, "").strip()

    svg_html = f"""
    <div style="text-align:center; margin: 8px 0 12px;">
      <svg width="200" height="200" viewBox="0 0 200 200" style="display:block; margin:0 auto;">
        <!-- Background arc (dark gray, 270°) -->
        <circle cx="100" cy="100" r="{radius}"
                fill="none" stroke="#2a2a2a" stroke-width="12"
                stroke-dasharray="{arc_length:.1f} {circumference:.1f}"
                stroke-dashoffset="0"
                stroke-linecap="round"
                transform="rotate(135 100 100)" />
        <!-- Colored arc (animated fill) -->
        <circle cx="100" cy="100" r="{radius}"
                fill="none" stroke="{arc_color}" stroke-width="12"
                stroke-dasharray="{fill_length:.1f} {gap:.1f}"
                stroke-dashoffset="{fill_length:.1f}"
                stroke-linecap="round"
                transform="rotate(135 100 100)"
                style="filter: drop-shadow(0 0 6px {arc_color}40);">
          <animate attributeName="stroke-dashoffset"
                   from="{arc_length:.1f}" to="0"
                   dur="1.2s" fill="freeze"
                   calcMode="spline"
                   keySplines="0.25 0.1 0.25 1" />
        </circle>
        <!-- Score number -->
        <text x="100" y="95" text-anchor="middle" dominant-baseline="central"
              fill="{arc_color}" font-size="48" font-weight="800"
              font-family="-apple-system, BlinkMacSystemFont, sans-serif">
          {score:.0f}
        </text>
        <!-- Rating label inside ring -->
        <text x="100" y="130" text-anchor="middle" dominant-baseline="central"
              fill="#8B949E" font-size="13" font-weight="600"
              font-family="-apple-system, BlinkMacSystemFont, sans-serif">
          {rating_short}
        </text>
      </svg>
      <div style="color: #E6EDF3; font-size: 0.9rem; font-weight: 600; margin-top: 4px;">
        {rating}
      </div>
    </div>
    """
    components.html(svg_html, height=260)


def _render_agent_cards(results: dict) -> None:
    """
    Render agent score cards as custom HTML to prevent st.metric truncation.

    Uses components.html (iframe) instead of st.markdown because Streamlit's
    markdown sanitiser strips complex/nested HTML, causing raw tags to show.

    Each card shows: full agent name, score/100 in large font, a coloured
    progress bar, win rate percentage, and feedback count — all with
    word-wrap so nothing is ever clipped.

    Args:
        results: ``{agent_name: AgentResult}`` dict from the orchestrator.
    """
    if not results:
        return

    # Build individual card divs
    card_divs = ""
    for name, r in results.items():
        # Bar colour: green >=65, yellow >=45, red <45
        if r.score >= 65:
            bar_color = _GREEN
        elif r.score >= 45:
            bar_color = _GOLD
        else:
            bar_color = _RED

        # Feedback count = number of feedback strings the agent produced
        fb_count = len(r.feedback) if hasattr(r, 'feedback') else 0
        plural = "s" if fb_count != 1 else ""

        card_divs += f"""
        <div class="agent-card">
            <div class="agent-card-name">{name} Agent</div>
            <div class="agent-card-score">{r.score:.0f} <span class="score-suffix">/ 100</span></div>
            <div class="agent-card-bar">
                <div class="agent-card-bar-fill" style="width:{r.score}%; background:{bar_color};"></div>
            </div>
            <div class="agent-card-meta">
                <strong>Win Rate:</strong> {r.win_rate:.1%}<br/>
                {fb_count} signal{plural} analyzed
            </div>
        </div>
        """

    # Self-contained HTML with inline styles (rendered in an iframe via components.html)
    num_cards = len(results)
    row_height = 160 if num_cards <= 3 else 340  # Stack wrap if >3 agents

    full_html = f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0; padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .agent-cards-row {{
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .agent-card {{
            flex: 1 1 160px;
            min-width: 160px;
            max-width: 280px;
            background: {_BG_CARD};
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 16px 18px;
            box-sizing: border-box;
        }}
        .agent-card-name {{
            color: {_TEXT_DIM};
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }}
        .agent-card-score {{
            font-size: 1.7rem;
            font-weight: 800;
            color: {_GOLD};
            margin-bottom: 6px;
        }}
        .score-suffix {{
            font-size: 0.9rem;
            color: #8B949E;
            font-weight: 500;
        }}
        .agent-card-bar {{
            width: 100%;
            height: 6px;
            background: #21262d;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        .agent-card-bar-fill {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.8s ease;
        }}
        .agent-card-meta {{
            font-size: 0.8rem;
            color: {_TEXT_DIM};
            line-height: 1.5;
        }}
        .agent-card-meta strong {{
            color: {_TEXT};
            font-weight: 600;
        }}
    </style>
    </head>
    <body>
        <div class="agent-cards-row">
            {card_divs}
        </div>
    </body>
    </html>
    """
    components.html(full_html, height=row_height)


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

    st.markdown(
        f"""
        <div class="aurum-header">
            <h1 class="aurum-title">AURUM ⚡ Gold Strategy Evaluator</h1>
            <p class="aurum-subtitle">
                Multi-agent AI system for evaluating, scoring, and backtesting
                Gold (XAUUSD) trading strategies
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── TradingView Live Preview ───────────────────────────────────────
    components.html(
        """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js" async>
          {
          "symbol": "OANDA:XAUUSD",
          "width": "100%",
          "isTransparent": true,
          "colorTheme": "dark",
          "locale": "en"
        }
          </script>
        </div>
        """,
        height=130,
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
    import os
    import json
    import time
    from pathlib import Path

    period_map = {1: "1y", 2: "2y", 3: "3y", 4: "4y", 5: "5y"}
    period_str = period_map.get(period_years, "3y")

    # ── DEMO MODE FALLBACK ───────────────────────────────────────────────
    DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE:
        with st.spinner("🤖 [DEMO MODE] Evaluating and Backtesting Strategy..."):
            time.sleep(2)  # Simulate processing delay
            demo_file = Path(__file__).resolve().parent.parent / "demo_data" / "sample_evaluation.json"
            try:
                with open(demo_file, "r") as f:
                    demo_data = json.load(f)
                
                # Reconstruct Pydantic models from dicts
                from agents.base_agent import AgentResult
                evaluation = demo_data["evaluation"]
                for agent, result_dict in evaluation["individual_results"].items():
                    evaluation["individual_results"][agent] = AgentResult(**result_dict)
                backtest = demo_data["backtest"]

                # Fetch real price data for chart if possible, else mock simple df
                try:
                    df = _compute_indicators(_fetch_history(period_str))
                except Exception:
                    # Generic empty df fallback to prevent UI crash
                    import pandas as pd
                    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "EMA_20", "EMA_50"])

                st.session_state["evaluation"] = evaluation
                st.session_state["backtest"] = backtest
                st.session_state["df"] = df
                st.session_state["conditions"] = [{"indicator": "RSI", "operator": "<", "value": 35, "action": "BUY"}]
                
                _render_left_results(col_left, evaluation)
                _render_right_results(col_right, df, evaluation, backtest)
                return
            except Exception as e:
                st.error(f"Demo Mode Error: {e}")
                import traceback
                st.write(traceback.format_exc())
                return

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
        conditions = parser.parse(strategy_text)

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
    # Default SL/TP config — conservative 1.5% stop with 2:1 reward-risk
    with st.spinner("⏳ Running backtest…"):
        BacktestEngine = _get_backtest_engine()
        engine = BacktestEngine()
        backtest = engine.run(
            df=df,
            conditions=conditions,
            stop_loss_config={"type": "percentage", "value": 1.5},
            risk_reward_ratio=2.0,
        )

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
    """Render SVG score gauge, custom agent cards, and conflict warnings in the left panel."""
    with col:
        st.markdown("---")

        # ── Final score gauge (SVG arc) ──────────────────────────────
        final_score = evaluation.get("final_score", 0.0)
        rating = evaluation.get("rating", "")
        _render_score_gauge(final_score, rating)

        # ── Agent score cards (custom HTML, no truncation) ───────────
        results = evaluation.get("individual_results", {})
        if results:
            st.markdown("#### Agent Scores")
            _render_agent_cards(results)

        # ── Conflicts ────────────────────────────────────────────────
        conflicts = evaluation.get("conflicts", [])
        if conflicts and len(conflicts) > 0:
            st.markdown("### \u26a1 Conflicts Detected")
            for conflict in conflicts:
                severity_icon = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f535"}
                icon = severity_icon.get(conflict.get("severity", "LOW"), "\U0001f535")
                st.warning(
                    f"{icon} **{conflict.get('conflict_type', 'Unknown')}** \u2014 "
                    f"{conflict.get('description', '')}"
                )
                st.caption(f"Agents involved: {', '.join(conflict.get('agents_involved', []))}")
        else:
            st.success("\u2705 No indicator conflicts detected \u2014 all agents agree on direction.")


# ══════════════════════════════════════════════════════════════════════
# RIGHT PANEL RESULTS
# ══════════════════════════════════════════════════════════════════════


def _render_right_results(col, df, evaluation: dict, backtest: dict) -> None:
    """Render tabs: Price Chart, Backtest Results, Agent Reports."""
    with col:
        tab_chart, tab_backtest, tab_reports = st.tabs(
            ["\U0001f4c8 Price Chart", "\U0001f9ea Backtest Results", "\U0001f916 Agent Reports"]
        )

        # ── Tab 1: Candlestick chart ─────────────────────────────────
        with tab_chart:
            fig = _build_candlestick(df, n_days=90)
            st.plotly_chart(fig, key="price_chart")

        # ── Tab 2: Backtest Results ──────────────────────────────────
        with tab_backtest:
            total_trades = backtest.get("total_trades", 0)

            if total_trades == 0:
                # ── Zero trades: styled warning + empty metric cards ──
                period_info = backtest.get("backtest_period", {})
                candles = period_info.get("total_candles", "N/A")
                st.markdown(
                    f"""
                    <div class="bt-warning-box">
                        <h4>\u26a0\ufe0f No Trades Generated</h4>
                        <p>
                            Your conditions are too strict for the selected backtest period
                            ({candles} candles). Try relaxing one condition
                            (e.g., raise RSI threshold from 30 to 35) or extend the
                            backtest period to capture more signal opportunities.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Still show metric cards with 0 values so tab is never blank
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Trades", 0)
                with m2:
                    st.metric("Accuracy", "0.0%")
                with m3:
                    st.metric("Net P&L", "0.00 pips")
                with m4:
                    st.metric("Max Drawdown", "0.00%")

            else:
                # ── Row 1: Primary metrics (4 cards) ─────────────────
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Trades", total_trades)
                with m2:
                    st.metric("Accuracy", f"{backtest.get('accuracy', 0):.1f}%")
                with m3:
                    net_pips = backtest.get("net_pnl_pips", 0.0)
                    color_prefix = "+" if net_pips >= 0 else ""
                    st.metric("Net P&L", f"{color_prefix}{net_pips:.2f} pips")
                with m4:
                    st.metric("Max Drawdown", f"-{backtest.get('max_drawdown_pct', 0):.2f}%")

                # ── Row 2: Secondary metrics (3 cards) ───────────────
                m5, m6, m7, _ = st.columns(4)
                with m5:
                    st.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
                with m6:
                    pf = backtest.get("profit_factor", 0)
                    pf_str = f"{pf:.2f}" if pf != float("inf") else "\u221e"
                    st.metric("Profit Factor", pf_str)
                with m7:
                    avg_win = backtest.get("avg_win_pips", 0.0)
                    avg_loss = abs(backtest.get("avg_loss_pips", 0.0)) or 1.0
                    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
                    st.metric("Avg Win/Loss", f"{wl_ratio:.2f}")

                # ── Row 3: Win/Loss streaks ──────────────────────────
                max_win_streak = backtest.get("max_win_streak", 0)
                max_loss_streak = backtest.get("max_loss_streak", 0)
                current_streak = backtest.get("current_streak", {})
                cs_type = current_streak.get("type", "WIN")
                cs_count = current_streak.get("count", 0)
                st.markdown(
                    f"""
                    <div class="streak-bar">
                        <div class="streak-item">\U0001f525 Max Win Streak: <strong>{max_win_streak}</strong></div>
                        <div class="streak-item">\U0001f4c9 Max Loss Streak: <strong>{max_loss_streak}</strong></div>
                        <div class="streak-item">\U0001f3af Current: <strong>{cs_count} {cs_type}{"s" if cs_count != 1 else ""}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Equity Curve (always shown — handles zero-trade case gracefully) ──
            st.markdown("##### Equity Curve")
            equity_fig = _build_equity_chart(backtest.get("equity_curve", []))
            st.plotly_chart(equity_fig, key="equity_chart")

        # ── Tab 3: Agent Reports ─────────────────────────────────────
        with tab_reports:
            results = evaluation.get("individual_results", {})
            if not results:
                st.info(
                    "Agent details unavailable \u2014 re-run evaluation to populate."
                )
            else:
                for name, result in results.items():
                    with st.expander(
                        f"{name} Agent \u2014 Score: {result.score:.0f}/100",
                        expanded=False,
                    ):
                        # Score progress bar
                        st.progress(min(result.score / 100.0, 1.0))

                        # Feedback section
                        st.markdown("**\U0001f4cb Feedback:**")
                        if result.feedback:
                            for fb in result.feedback:
                                st.markdown(f"- {fb}")
                        else:
                            st.caption("No feedback generated.")

                        # Suggestions section
                        st.markdown("**\U0001f4a1 Suggestions:**")
                        if result.suggestions:
                            for sg in result.suggestions:
                                st.markdown(f"- {sg}")
                        else:
                            st.caption("No suggestions generated.")

                        # Action alignment as colored badge
                        action = result.action_alignment
                        badge_class = action.lower() if action in ("BUY", "SELL", "NEUTRAL") else "neutral"
                        st.markdown(
                            f'**\U0001f4ca Action Alignment:** '
                            f'<span class="action-badge {badge_class}">{action}</span>',
                            unsafe_allow_html=True,
                        )

                        st.caption(
                            f"Win Rate: {result.win_rate:.2%}  \u2022  "
                            f"Signals analyzed: {len(result.feedback)}"
                        )

            # ── Aggregated Suggestions ───────────────────────────
            suggestions = evaluation.get("suggestions", [])
            if suggestions:
                st.markdown("---")
                st.markdown("#### \U0001f4a1 Strategy Suggestions")
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
