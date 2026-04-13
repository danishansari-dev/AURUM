import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, ComposedChart, Bar 
} from 'recharts';

// -----------------------------------------------------------------------------
// INTERFACES
// -----------------------------------------------------------------------------

export interface AURUMDashboardProps {
  livePrice: {
    price: number;
    change_pct: number;
    change_abs: number;
    direction: "up" | "down";
    timestamp: string;
  };
  systemHealth: {
    last_fetch: string;
    apis: {
      yfinance: "Connected" | "Disconnected";
      alpha_vantage: "Key configured" | "Key not set" | "Disconnected";
      anthropic: "Key configured" | "Key not set" | "Disconnected";
    };
  };
  priceHistory: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    ema20: number;
    ema50: number;
  }>;
  evaluationResult: null | {
    final_score: number;
    rating: string;
    rating_full: string;
    individual_results: {
      [key: string]: {
        score: number;
        win_rate: number;
        feedback: string[];
        suggestions: string[];
        action_alignment: "BUY" | "SELL" | "NEUTRAL";
        signals_analyzed: number;
      };
    };
    conflicts_detected: Array<{
      conflict_type: string;
      agents_involved: string[];
      description: string;
      severity: "HIGH" | "MEDIUM" | "LOW";
    }>;
    all_suggestions: string[];
    backtest: {
      total_trades: number;
      winning_trades: number;
      losing_trades: number;
      accuracy: number;
      net_pnl_pct: number;
      max_drawdown_pct: number;
      sharpe_ratio: number;
      profit_factor: number;
      max_win_streak: number;
      max_loss_streak: number;
      current_streak: { type: "WIN" | "LOSS"; count: number };
      equity_curve: number[];
      candles_analyzed: number;
    };
  };
  isLoading: boolean;
  onEvaluate: (strategy: string, years: number) => Promise<void>;
}

// -----------------------------------------------------------------------------
// MOCK DATA GENERATOR
// -----------------------------------------------------------------------------

const generateMockPriceHistory = () => {
  const data = [];
  let price = 4700;
  let ema20 = 4700;
  let ema50 = 4700;
  const now = new Date("2026-04-13T00:00:00Z");
  
  for (let i = 89; i >= 0; i--) {
    const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
    const open = price;
    const change = (Math.random() - 0.48) * 40;
    const close = open + change;
    const high = Math.max(open, close) + Math.random() * 20;
    const low = Math.min(open, close) - Math.random() * 20;
    
    ema20 = ema20 * 0.9 + close * 0.1;
    ema50 = ema50 * 0.95 + close * 0.05;
    price = close;
    
    data.push({
      date: date.toISOString().split("T")[0],
      open, high, low, close,
      ema20, ema50,
      wickRange: [low, high] // For Recharts to compute full height
    });
  }
  return data;
};

const MOCK_DATA: AURUMDashboardProps = {
  livePrice: {
    price: 4740.815,
    change_pct: -0.19,
    change_abs: -8.87,
    direction: "down",
    timestamp: "2026-04-13 19:31:23 UTC"
  },
  systemHealth: {
    last_fetch: "2026-04-13 19:31:23 UTC",
    apis: {
      yfinance: "Connected",
      alpha_vantage: "Key configured",
      anthropic: "Key configured"
    }
  },
  priceHistory: generateMockPriceHistory(),
  evaluationResult: {
    final_score: 35,
    rating: "POOR",
    rating_full: "❌ POOR — Do not trade this strategy",
    individual_results: {
      "EMA Agent": { 
        score: 57, 
        win_rate: 0.5793, 
        feedback: [
          "EMA(20) = 4733.46, Close = 4761.90 (+0.60%).", 
          "Rule-based credit 30.0/40.", 
          "Historical win-rate estimate = 57.93% contributing 23.2/40.", 
          "Slope component = 3.4/20."
        ], 
        suggestions: [
          "Moderate EMA signal; reduce size and wait for slope confirmation.", 
          "Watch if price retests the EMA before adding exposure."
        ], 
        action_alignment: "BUY", 
        signals_analyzed: 4 
      },
      "RSI Agent": { 
        score: 24, 
        win_rate: 0.60, 
        feedback: [
          "Latest RSI=50.17 with rule-based credit 0.0/40.", 
          "Historical win-rate estimate=60.00% contributing 24.0/40.", 
          "Divergence component=0.0/20."
        ], 
        suggestions: [
          "No compelling RSI edge detected under current parameters.", 
          "Avoid new exposure unless unrelated models disagree strongly.", 
          "Re-run after fresh data arrives or adjust the condition thresholds."
        ], 
        action_alignment: "NEUTRAL", 
        signals_analyzed: 3 
      },
      "MACD Agent": { 
        score: 24, 
        win_rate: 0.5876, 
        feedback: [
          "MACD = -44.8820, Signal = 31.7829, Hist = -76.6649.", 
          "Rule-based credit 0.0/40.", 
          "Historical win-rate estimate = 58.76% contributing 23.5/40.", 
          "Divergence component = 0.0/20."
        ], 
        suggestions: [
          "No compelling MACD edge; avoid trading on MACD alone.", 
          "Wait for a clean zero-line crossover before re-entering."
        ], 
        action_alignment: "NEUTRAL", 
        signals_analyzed: 4 
      }
    },
    conflicts_detected: [],
    all_suggestions: [
      "Moderate EMA signal; reduce size and wait for slope confirmation.",
      "Watch if price retests the EMA before adding exposure.",
      "No compelling RSI edge detected under current parameters.",
      "Avoid new exposure unless unrelated models disagree strongly.",
      "Re-run after fresh data arrives or adjust the condition thresholds.",
      "No compelling MACD edge; avoid trading on MACD alone.",
      "Wait for a clean zero-line crossover before re-entering."
    ],
    backtest: {
      total_trades: 0,
      winning_trades: 0,
      losing_trades: 0,
      accuracy: 0.0,
      net_pnl_pct: 0.0,
      max_drawdown_pct: 0.0,
      sharpe_ratio: 0,
      profit_factor: 0,
      max_win_streak: 0,
      max_loss_streak: 0,
      current_streak: { type: "WIN", count: 0 },
      equity_curve: [],
      candles_analyzed: 253
    }
  },
  isLoading: false,
  onEvaluate: async () => {
    return new Promise(resolve => setTimeout(resolve, 1500));
  }
};

// -----------------------------------------------------------------------------
// HELPER COMPONENTS
// -----------------------------------------------------------------------------

const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=Geist:wght@400;500;600&display=swap');

    :root {
      --gold: #FFD700;
      --gold-dim: #B8960C;
      --gold-text: #E6C200;
      --green: #00FF88;
      --red: #FF4444;
      --amber: #FFA500;
      --blue-accent: #4a90e2;
      --text-primary: #F0F0E8;
      --text-muted: #6B7560;
      --surface: #141714;
      --border: #252925;
    }

    body {
      margin: 0; padding: 0;
      background: #0D0F0D;
      color: var(--text-primary);
      font-family: 'Geist', sans-serif;
      overflow: hidden;
    }

    * { box-sizing: border-box; }

    .font-bebas { font-family: 'Bebas Neue', sans-serif; }
    .font-mono { font-family: 'DM Mono', monospace; }
    .font-geist { font-family: 'Geist', sans-serif; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #252925; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3e3a; }

    .aurum-logo-text {
      background: linear-gradient(90deg, #FFD700, #FFA500);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    textarea:focus { border-color: var(--gold) !important; outline: none; }

    .evaluate-btn {
      background: linear-gradient(135deg, #FFD700 0%, #B8960C 100%);
      color: #0D0F0D;
      transition: all 150ms ease;
    }
    .evaluate-btn:hover:not(:disabled) { filter: brightness(1.12); transform: translateY(-1px); }
    .evaluate-btn:active:not(:disabled) { filter: brightness(0.95); transform: translateY(0); }
    .evaluate-btn:disabled { opacity: 0.85; cursor: not-allowed; }

    .custom-range { -webkit-appearance: none; width: 100%; background: transparent; }
    .custom-range::-webkit-slider-thumb {
      -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%;
      background: var(--gold); cursor: pointer; margin-top: -6px; transition: transform 0.1s;
    }
    .custom-range::-webkit-slider-thumb:hover { transform: scale(1.15); box-shadow: 0 0 6px var(--gold-dim); }
    .custom-range::-webkit-slider-runnable-track { width: 100%; height: 4px; cursor: pointer; background: #2a2d2a; border-radius: 2px; }

    .spinner {
      display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(0,0,0,0.3);
      border-top-color: #0D0F0D; border-radius: 50%; animation: spin 1s linear infinite;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    @keyframes arcFill { from { stroke-dasharray: 0 503; } }

    .metric-value { font-size: clamp(16px, 2.2vw, 28px); white-space: nowrap; text-overflow: clip; overflow: visible; }
  `}</style>
);

const getScoreColor = (score: number) => {
  if (score >= 70) return 'var(--green)';
  if (score >= 55) return 'var(--gold)';
  if (score >= 40) return 'var(--amber)';
  return 'var(--red)';
};

// -----------------------------------------------------------------------------
// RECHARTS CANDLESTICK SHAPE
// -----------------------------------------------------------------------------
const CandlestickShape = (props: any) => {
  const { x, y, width, height, payload } = props;
  const isGrowing = payload.close >= payload.open;
  const color = isGrowing ? 'var(--green)' : 'var(--red)';
  
  const valRange = payload.high - payload.low;
  const pxPerVal = valRange === 0 ? 0 : height / valRange;
  
  const bodyTopVal = Math.max(payload.open, payload.close);
  const bodyBottomVal = Math.min(payload.open, payload.close);
  
  const bodyY = y + (payload.high - bodyTopVal) * pxPerVal;
  const bodyHeight = Math.max((bodyTopVal - bodyBottomVal) * pxPerVal, 2); 
  const lineX = x + width / 2;
  
  return (
    <g>
      <line x1={lineX} y1={y} x2={lineX} y2={y + height} stroke={color} strokeWidth={1} />
      <rect x={x} y={bodyY} width={width} height={bodyHeight} fill={color} />
    </g>
  );
};

// -----------------------------------------------------------------------------
// MAIN COMPONENT
// -----------------------------------------------------------------------------
export default function AURUMDashboard(props: Partial<AURUMDashboardProps> = {}) {
  const data = { ...MOCK_DATA, ...props };
  const [activeTab, setActiveTab] = useState(0);
  const [strategy, setStrategy] = useState("Buy Gold when RSI is below 35 and EMA20\ncrosses above EMA50 and MACD bullish\ncrossover");
  const [years, setYears] = useState(5);
  const [isEvaluating, setIsEvaluating] = useState(data.isLoading);
  const [scoreAnimationTrigger, setScoreAnimationTrigger] = useState(0);

  useEffect(() => {
    setIsEvaluating(data.isLoading);
    if (!data.isLoading && data.evaluationResult) {
      setScoreAnimationTrigger(prev => prev + 1); // Rerender animation logic
    }
  }, [data.isLoading, data.evaluationResult]);

  const handleEvaluate = async () => {
    setIsEvaluating(true);
    await data.onEvaluate(strategy, years);
    setIsEvaluating(false);
  };

  const evalRes = data.evaluationResult;

  // --- SVG arc calc ---
  const r = 80;
  const circ = 2 * Math.PI * r; 
  const activeArc = circ * 0.75; 
  const finalScore = evalRes ? evalRes.final_score : 0;
  const arcColor = getScoreColor(finalScore);
  const fillAmount = evalRes ? (finalScore / 100) * activeArc : 0;

  return (
    <div className="flex flex-row h-screen overflow-hidden" style={{ background: '#0D0F0D' }}>
      <GlobalStyles />
      
      {/* SIDEBAR */}
      <aside className="flex-shrink-0 overflow-y-auto" style={{ width: '200px', background: '#0A0C0A', borderRight: '1px solid #1a1e1a', padding: '20px 16px' }}>
        <div style={{ marginBottom: 20 }}>
          <div className="font-geist" style={{ fontSize: 13, color: 'var(--gold)', letterSpacing: '1.5px' }}>⚙ System Health</div>
        </div>
        
        <div style={{ marginBottom: 16 }}>
          <div className="font-geist uppercase" style={{ fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: 4 }}>Last Data Fetch</div>
          <div className="font-mono" style={{ fontSize: 11, color: 'var(--text-primary)' }}>{data.systemHealth.last_fetch}</div>
        </div>

        <div style={{ marginBottom: 24 }}>
          <div className="font-geist" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Gold Spot Price (Live)</div>
          <div style={{ background: '#1a1e1a', border: '1px solid #252925', borderRadius: 8, padding: '8px 12px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="font-mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>XAUUSD</div>
            <div className="font-bebas" style={{ fontSize: 20, color: data.livePrice.direction === 'up' ? 'var(--green)' : 'var(--red)' }}>
              ${data.livePrice.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div style={{ color: data.livePrice.direction === 'up' ? 'var(--green)' : 'var(--red)', fontSize: 14 }}>
              {data.livePrice.direction === 'up' ? '▲' : '▼'}
            </div>
          </div>
        </div>

        <div style={{ borderTop: '1px solid #1a1e1a', width: '100%', marginBottom: 20 }}></div>

        <div>
          <div className="font-geist uppercase" style={{ fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: 12 }}>API Connections</div>
          {[
            { name: "yfinance", status: data.systemHealth.apis.yfinance },
            { name: "Alpha Vantage", status: data.systemHealth.apis.alpha_vantage },
            { name: "Anthropic (Claude)", status: data.systemHealth.apis.anthropic }
          ].map((api, i) => (
            <div key={i} style={{ height: 28, display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: api.status.includes('Connected') || api.status.includes('configured') ? 'var(--green)' : (api.status.includes('not set') ? 'var(--amber)' : 'var(--red)') }}></div>
              <div className="font-mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>{api.name.split(' ')[0]}</div>
              <div className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>— {api.status}</div>
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 overflow-y-auto" style={{ padding: '0 28px 40px 28px' }}>
        
        {/* HERO BANNER */}
        <div style={{ background: '#1a1d1a', border: '1px solid #252925', borderRadius: 8, padding: '24px 28px', marginBottom: 16, marginTop: 24 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span className="font-bebas aurum-logo-text" style={{ fontSize: 44 }}>AURUM</span>
            <span style={{ fontSize: 36 }}>⚡</span>
            <span className="font-bebas" style={{ fontSize: 44, color: 'var(--text-primary)' }}>Gold Strategy Evaluator</span>
          </div>
          <div className="font-mono" style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 6 }}>
            Multi-agent AI system for evaluating, scoring, and backtesting Gold (XAUUSD) trading strategies
          </div>
        </div>

        {/* LIVE TICKER BAR */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 20px', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 28 }}>🪙</span>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span className="font-bebas" style={{ fontSize: 15, color: 'var(--text-muted)', letterSpacing: 2 }}>XAUUSD</span>
              <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-muted)' }}>GOLD SPOT / U.S. DOLLAR</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span className="font-bebas" style={{ fontSize: 34, color: 'var(--text-primary)' }}>{data.livePrice.price.toLocaleString('en-US', { minimumFractionDigits: 3 })}</span>
            <span className="font-mono" style={{ fontSize: 14, color: data.livePrice.direction === 'up' ? 'var(--green)' : 'var(--red)' }}>
              {data.livePrice.direction === 'up' ? '+' : ''}{data.livePrice.change_pct}% ({Math.abs(data.livePrice.change_abs)})
            </span>
          </div>
          <div style={{ marginLeft: 'auto', width: 32, height: 32, background: '#2a2d2a', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>TV</span>
          </div>
        </div>

        {/* TWO PANEL AREA */}
        <div style={{ display: 'flex', flexDirection: 'row', gap: 20, alignItems: 'flex-start' }}>
          
          {/* LEFT PANEL */}
          <div style={{ width: 420, flexShrink: 0 }}>
            
            {/* STRATEGY INPUT CARD */}
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: 20 }}>
              <div className="font-geist" style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 12 }}>📋 Strategy Input</div>
              <textarea 
                value={strategy}
                onChange={e => setStrategy(e.target.value)}
                className="font-mono"
                style={{ width: '100%', height: 100, resize: 'vertical', background: '#0D0F0D', border: '1px solid #252925', borderRadius: 6, padding: 12, color: 'var(--text-primary)', lineHeight: 1.6 }}
                spellCheck={false}
              />
              
              <div style={{ position: 'relative', width: '100%', marginTop: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span className="font-geist" style={{ fontSize: 13, color: 'var(--text-muted)' }}>Backtest period (years)</span>
                  <span className="font-geist" style={{ fontSize: 13, color: 'var(--text-muted)', cursor: 'help' }} title="Select how many years of Gold data to use for backtesting">ⓘ</span>
                </div>
                <div style={{ position: 'relative', height: 24 }}>
                   <div className="font-bebas" style={{ 
                     position: 'absolute', left: `calc(${((years - 1) / 4) * 100}% - 12px)`, top: 0, 
                     background: 'var(--gold)', color: '#0D0F0D', padding: '1px 8px', borderRadius: 10, 
                     fontSize: 14, whiteSpace: 'nowrap', transition: 'left 0.1s ease', zIndex: 10
                   }}>{years}</div>
                </div>
                <input type="range" className="custom-range" min="1" max="5" step="1" value={years} onChange={(e) => setYears(parseInt(e.target.value))} 
                  style={{ background: `linear-gradient(to right, var(--gold) ${((years - 1) / 4) * 100}%, #2a2d2a ${((years - 1) / 4) * 100}%)`}} />
              </div>

              <button className="evaluate-btn font-geist" style={{ width: '100%', height: 48, borderRadius: 6, border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: 16, fontWeight: 700, fontSize: 15 }} onClick={handleEvaluate} disabled={isEvaluating}>
                {isEvaluating ? <><span className="spinner"></span>⟳ Analyzing...</> : "⚡ Evaluate Strategy"}
              </button>
            </div>

            {/* SCORE RING */}
            {evalRes && (
              <div style={{ marginTop: 24, width: '100%', textAlign: 'center' }}>
                <svg viewBox="0 0 200 200" width="200" height="200" style={{ margin: '0 auto', display: 'block' }} key={scoreAnimationTrigger}>
                  <circle cx="100" cy="100" r={r} fill="none" stroke="#2a2d2a" strokeWidth="10" strokeLinecap="round" transform="rotate(135 100 100)" strokeDasharray={`${activeArc} ${circ}`} />
                  <circle cx="100" cy="100" r={r} fill="none" stroke={arcColor} strokeWidth="10" strokeLinecap="round" transform="rotate(135 100 100)" strokeDasharray={`${fillAmount} ${circ}`} style={{ animation: 'arcFill 1.2s ease-out forwards' }} />
                  <foreignObject x="0" y="0" width="200" height="200">
                     <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                       <div className="font-bebas" style={{ fontSize: 52, color: arcColor, lineHeight: 1 }}>{finalScore}</div>
                       <div className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)', letterSpacing: 3, marginTop: 4 }}>{evalRes.rating}</div>
                     </div>
                  </foreignObject>
                </svg>
                <div className="font-mono" style={{ fontSize: 13, color: evalRes.rating.includes('POOR') ? 'var(--red)' : (evalRes.rating.includes('GOOD') || evalRes.rating.includes('EXCELLENT') ? 'var(--green)' : 'var(--amber)'), marginTop: 8 }}>
                  {evalRes.rating_full}
                </div>
              </div>
            )}

            {/* AGENT SCORES */}
            {evalRes && (
              <div style={{ marginTop: 20 }}>
                <div className="font-geist" style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 12 }}>Agent Scores</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
                  {Object.entries(evalRes.individual_results).map(([name, data]) => {
                    const clr = getScoreColor(data.score);
                    return (
                      <div key={name} style={{ background: '#0D0F0D', border: '1px solid #252925', borderLeft: `3px solid ${clr}`, borderRadius: 6, padding: 14, minWidth: 0 }}>
                        <div className="font-mono uppercase" style={{ fontSize: 11, color: 'var(--text-muted)', letterSpacing: 1, marginBottom: 6 }}>{name}</div>
                        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, marginBottom: 8 }}>
                          <span className="font-bebas" style={{ fontSize: 30, color: clr }}>{data.score}</span>
                          <span className="font-mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>/ 100</span>
                        </div>
                        <div style={{ width: '100%', height: 4, background: '#2a2d2a', borderRadius: 2 }}>
                          <div style={{ height: '100%', width: `${data.score}%`, background: clr, borderRadius: 2, transition: 'width 0.8s ease-out' }}></div>
                        </div>
                        <div className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, whiteSpace: 'nowrap', overflow: 'visible' }}>Win Rate: {(data.win_rate * 100).toFixed(1)}%</div>
                        <div className="font-mono" style={{ fontSize: 10, color: '#3a3e3a', marginTop: 2 }}>{data.signals_analyzed} signals analyzed</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* CONFLICTS */}
            {evalRes && (
              <div style={{ marginTop: 16 }}>
                {evalRes.conflicts_detected.length === 0 ? (
                  <div style={{ background: 'rgba(0,255,136,0.08)', border: '1px solid rgba(0,255,136,0.3)', borderRadius: 8, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 18 }}>✅</span>
                    <span className="font-mono" style={{ fontSize: 12, color: 'var(--green)' }}>No indicator conflicts detected — all agents agree on direction.</span>
                  </div>
                ) : (
                  evalRes.conflicts_detected.map((c, i) => {
                    const clr = c.severity === 'HIGH' ? 'var(--red)' : c.severity === 'MEDIUM' ? 'var(--amber)' : 'var(--blue-accent)';
                    return (
                      <div key={i} style={{ background: 'rgba(255,165,0,0.07)', borderLeft: `3px solid ${clr}`, borderRadius: 4, padding: '10px 14px', marginBottom: 8 }}>
                        <div className="font-geist" style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 600 }}>{c.conflict_type}</div>
                        <div className="font-mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>{c.description}</div>
                        <div className="font-mono" style={{ fontSize: 11, color: '#4a4e4a' }}>Agents: {c.agents_involved.join(', ')}</div>
                      </div>
                    );
                  })
                )}
              </div>
            )}
          </div>

          {/* RIGHT PANEL */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {/* TAB BAR */}
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '8px 8px 0 0', display: 'flex', borderBottom: '1px solid var(--border)', padding: '0 20px' }}>
              {["📈 Price Chart", "📊 Backtest Results", "🤖 Agent Reports"].map((tab, idx) => (
                <div 
                  key={idx} className="font-geist"
                  style={{
                    padding: '12px 18px', fontSize: 13, cursor: 'pointer', userSelect: 'none',
                    color: activeTab === idx ? 'var(--text-primary)' : 'var(--text-muted)',
                    borderBottom: `2px solid ${activeTab === idx ? 'var(--gold)' : 'transparent'}`,
                    fontWeight: activeTab === idx ? 600 : 400,
                    transition: 'color 150ms, border-color 150ms'
                  }}
                  onClick={() => setActiveTab(idx)}
                >
                  {tab}
                </div>
              ))}
            </div>
            
            {/* TAB PANEL CONTAINER */}
            <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderTop: 'none', borderRadius: '0 0 8px 8px', padding: 20, minHeight: 480 }}>
               
              {activeTab === 0 && (
                <div>
                   <div style={{ height: 380, width: '100%' }}>
                     <ResponsiveContainer width="100%" height="100%">
                       <ComposedChart data={data.priceHistory}>
                         <CartesianGrid stroke="#1e221e" vertical={false} strokeDasharray="3 3"/>
                         <XAxis dataKey="date" tickFormatter={d => { const dt=new Date(d); return dt.toLocaleString('en-US', {month:'short', year:'numeric'}) }} tick={{fill:'var(--text-muted)', fontSize: 10, fontFamily: 'DM Mono'}} axisLine={false} tickLine={false} minTickGap={50} />
                         <YAxis orientation="right" domain={['auto', 'auto']} tickFormatter={v => '$'+v.toLocaleString()} tick={{fill:'var(--text-muted)', fontSize: 10, fontFamily: 'DM Mono'}} axisLine={false} tickLine={false} label={{ value: 'Price (USD)', angle: -90, position: 'insideRight', fill: 'var(--text-muted)', fontSize: 11 }} />
                         <RechartsTooltip content={(props) => {
                           if (props.active && props.payload && props.payload.length) {
                             const p = props.payload[0].payload;
                             return (
                               <div style={{ background: '#1a1d1a', border: '1px solid #252925', borderRadius: 4, padding: '8px 12px' }} className="font-mono text-[12px] text-white">
                                 <div style={{ marginBottom: 4, color: 'var(--text-muted)' }}>{p.date}</div>
                                 <div style={{ color: 'var(--text-primary)'}}>Open: {p.open.toFixed(2)}</div>
                                 <div style={{ color: 'var(--text-primary)'}}>High: {p.high.toFixed(2)}</div>
                                 <div style={{ color: 'var(--text-primary)'}}>Low: {p.low.toFixed(2)}</div>
                                 <div style={{ color: 'var(--text-primary)'}}>Close: {p.close.toFixed(2)}</div>
                               </div>
                             );
                           }
                           return null;
                         }} />
                         <Bar dataKey="wickRange" shape={<CandlestickShape />} isAnimationActive={false} />
                         <Line type="monotone" dataKey="ema20" stroke="var(--gold)" dot={false} strokeWidth={2} isAnimationActive={false} />
                         <Line type="monotone" dataKey="ema50" stroke="var(--blue-accent)" dot={false} strokeWidth={2} strokeDasharray="4 4" isAnimationActive={false} />
                       </ComposedChart>
                     </ResponsiveContainer>
                   </div>
                   <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 16, marginTop: 8 }}>
                     <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}><span style={{color:'var(--green)'}}>●</span> XAUUSD</span>
                     <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}><span style={{color:'var(--gold)'}}>—</span> EMA 20</span>
                     <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}><span style={{color:'var(--blue-accent)'}}>--</span> EMA 50</span>
                   </div>
                </div>
              )}

              {activeTab === 1 && !evalRes && (
                 <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 400, color: 'var(--text-muted)' }} className="font-mono text-[13px]">
                   Run a strategy evaluation to see results here.
                 </div>
              )}
              {activeTab === 1 && evalRes && (
                <div>
                   {evalRes.backtest.total_trades === 0 && (
                      <div style={{ background: 'rgba(255,165,0,0.08)', border: '1px solid var(--amber)', borderRadius: 8, padding: 20, marginBottom: 16 }}>
                         <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <span style={{ fontSize: 24, color: 'var(--amber)' }}>⚠</span>
                            <span className="font-geist" style={{ color: 'var(--amber)', fontSize: 16, fontWeight: 600 }}>No Trades Generated</span>
                         </div>
                         <div className="font-mono" style={{ color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6, marginTop: 8 }}>
                            Your conditions are too strict for the selected backtest period ({evalRes.backtest.candles_analyzed} candles). Try relaxing one condition (e.g., raise RSI threshold from 30 to 35) or extend the backtest period to capture more signal opportunities.
                         </div>
                      </div>
                   )}
                   
                   <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10 }}>
                     <MetricCard label="Total Trades" value={evalRes.backtest.total_trades.toString()} color="var(--gold)" />
                     <MetricCard label="Accuracy" value={evalRes.backtest.accuracy.toFixed(1) + '%'} color={evalRes.backtest.accuracy > 60 && evalRes.backtest.total_trades > 0 ? 'var(--green)' : 'var(--gold)'} />
                     <MetricCard label="Net P&L" value={(evalRes.backtest.net_pnl_pct > 0 ? '+' : '') + evalRes.backtest.net_pnl_pct.toFixed(1) + '%'} color={evalRes.backtest.net_pnl_pct > 0 ? 'var(--green)' : 'var(--gold)'} />
                     <MetricCard label="Max Drawdown" value={evalRes.backtest.max_drawdown_pct.toFixed(1) + '%'} color={evalRes.backtest.total_trades === 0 ? 'var(--gold)' : 'var(--red)'} />
                   </div>
                   
                   {evalRes.backtest.total_trades > 0 && (
                     <>
                       <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 10, marginTop: 10 }}>
                         <MetricCard label="Sharpe Ratio" value={evalRes.backtest.sharpe_ratio.toFixed(2)} color="var(--gold)" />
                         <MetricCard label="Profit Factor" value={evalRes.backtest.profit_factor.toFixed(2)} color="var(--gold)" />
                         <MetricCard label="Win/Loss %" value={`${evalRes.backtest.winning_trades} / ${evalRes.backtest.losing_trades}`} color="var(--gold)" />
                       </div>
                       <div style={{ background: '#1a1d1a', border: '1px solid var(--border)', borderRadius: 20, padding: '10px 16px', marginTop: 10, display: 'flex', justifyContent: 'space-between' }}>
                         <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>Max Streaks: <span style={{color: 'var(--green)'}}>{evalRes.backtest.max_win_streak} W</span> / <span style={{color: 'var(--red)'}}>{evalRes.backtest.max_loss_streak} L</span></span>
                         <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>Current Streak: <span style={{color: evalRes.backtest.current_streak.type === 'WIN' ? 'var(--green)' : 'var(--red)'}}>{evalRes.backtest.current_streak.count} {evalRes.backtest.current_streak.type.charAt(0)}</span></span>
                       </div>
                     </>
                   )}

                   <div style={{ marginTop: 20 }}>
                      <div className="font-geist" style={{ color: 'var(--text-primary)', fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Equity Curve</div>
                      <div style={{ height: 200, background: '#0D0F0D', border: '1px solid var(--border)', borderRadius: 6, padding: 16, position: 'relative' }}>
                         {evalRes.backtest.total_trades === 0 ? (
                           <div className="font-mono" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: 'var(--text-muted)', fontSize: 13 }}>
                             No trades generated
                           </div>
                         ) : (
                            <ResponsiveContainer width="100%" height="100%">
                               <LineChart data={evalRes.backtest.equity_curve.map((v, i) => ({ trade: i, value: v }))}>
                                  <CartesianGrid stroke="#1e221e" vertical={false} strokeDasharray="3 3"/>
                                  <XAxis dataKey="trade" hide />
                                  <YAxis domain={['auto', 'auto']} hide />
                                  <RechartsTooltip />
                                  <Line type="stepAfter" dataKey="value" stroke="var(--gold)" strokeWidth={2} dot={false} />
                               </LineChart>
                            </ResponsiveContainer>
                         )}
                      </div>
                   </div>
                </div>
              )}

              {activeTab === 2 && !evalRes && (
                 <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 400, color: 'var(--text-muted)' }} className="font-mono text-[13px]">
                   Run a strategy evaluation to see results here.
                 </div>
              )}
              {activeTab === 2 && evalRes && (
                <div>
                  {Object.entries(evalRes.individual_results).map(([name, r]) => (
                    <AgentReportAccordion key={name} name={name} data={r} />
                  ))}
                </div>
              )}

            </div>

            {/* STRATEGY SUGGESTIONS */}
            {evalRes && (
              <div style={{ marginTop: 20, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: 20 }}>
                <div className="font-geist" style={{ fontSize: 16, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 14 }}>💡 Strategy Suggestions</div>
                <div>
                  {evalRes.all_suggestions.map((sug, idx) => (
                    <div key={idx} style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
                      <span className="font-mono" style={{ color: 'var(--gold)', fontSize: 14, flexShrink: 0, marginTop: 1 }}>•</span>
                      <span className="font-mono" style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.6 }}>{sug}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
          </div>
        </div>
      </main>
    </div>
  );
}

// -----------------------------------------------------------------------------
// METRIC CARD COMPONENT
// -----------------------------------------------------------------------------
const MetricCard = ({ label, value, color }: { label: string, value: string, color: string }) => (
  <div style={{ background: '#0D0F0D', border: '1px solid var(--border)', borderRadius: 6, padding: 16, minWidth: 0, minHeight: 80, display: 'flex', flexDirection: 'column', justifyContent: 'center', overflow: 'hidden' }}>
    <div className="font-mono" style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
    <div className="font-bebas metric-value" style={{ color }}>{value}</div>
  </div>
);

// -----------------------------------------------------------------------------
// AGENT REPORT ACCORDION COMPONENT
// -----------------------------------------------------------------------------
const AgentReportAccordion = ({ name, data }: { name: string, data: any }) => {
  const [isOpen, setIsOpen] = useState(false);
  const scoreColor = getScoreColor(data.score);
  
  let badgeProps = { bg: '', color: '', border: '', label: '', emoji: '' };
  if (data.action_alignment === 'BUY') {
    badgeProps = { bg: 'rgba(0,255,136,0.12)', color: 'var(--green)', border: '1px solid rgba(0,255,136,0.3)', label: 'BUY', emoji: '🟢' };
  } else if (data.action_alignment === 'SELL') {
    badgeProps = { bg: 'rgba(255,68,68,0.12)', color: 'var(--red)', border: '1px solid rgba(255,68,68,0.3)', label: 'SELL', emoji: '🔴' };
  } else {
    badgeProps = { bg: 'rgba(107,117,96,0.2)', color: 'var(--text-muted)', border: '1px solid #3a3e3a', label: 'NEUTRAL', emoji: '⬜' };
  }

  return (
    <>
      <div onClick={() => setIsOpen(!isOpen)} style={{ background: '#1a1d1a', border: '1px solid var(--border)', borderRadius: 6, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer', marginBottom: isOpen ? 0 : 6 }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)', transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}>▼</span>
        <span className="font-geist" style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 600 }}>{name}</span>
        <span className="font-mono" style={{ background: 'rgba(255,215,0,0.1)', borderRadius: 12, padding: '3px 10px', fontSize: 12, color: 'var(--gold)', fontWeight: 600 }}>Score: {data.score}/100</span>
        <div style={{ flex: 1 }}></div>
        <div className="font-mono" style={{ background: badgeProps.bg, color: badgeProps.color, border: badgeProps.border, borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>
          {badgeProps.label}
        </div>
      </div>
      
      {isOpen && (
        <div style={{ background: '#111311', border: '1px solid var(--border)', borderTop: 'none', borderRadius: '0 0 6px 6px', padding: 16, marginTop: -6, marginBottom: 10, position: 'relative', zIndex: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ flex: 1, height: 6, background: '#2a2d2a', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${data.score}%`, background: scoreColor, borderRadius: 3, transition: 'width 0.8s ease' }}></div>
            </div>
            <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{data.score}/100</span>
          </div>

          <div className="font-geist" style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginTop: 14, marginBottom: 8 }}>📋 Feedback:</div>
          {data.feedback.map((f: string, i: number) => (
            <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <span className="font-mono" style={{ color: 'var(--gold)', fontSize: 13, flexShrink: 0 }}>•</span>
              <span className="font-mono" style={{ color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>{f}</span>
            </div>
          ))}

          <div className="font-geist" style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 600, marginTop: 14, marginBottom: 8 }}>💡 Suggestions:</div>
          {data.suggestions.map((s: string, i: number) => (
            <div key={i} className="font-mono" style={{ background: 'rgba(255,215,0,0.04)', borderLeft: '2px solid var(--gold-dim)', borderRadius: '0 4px 4px 0', padding: '8px 12px', marginBottom: 6, color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>
              {s}
            </div>
          ))}

          <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="font-geist" style={{ fontSize: 12, color: 'var(--text-muted)' }}>Action Alignment: </span>
            <span className="font-mono" style={{ background: badgeProps.bg, color: badgeProps.color, border: badgeProps.border, borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>{badgeProps.label}</span>
            <span style={{ fontSize: 12 }}>{badgeProps.emoji}</span>
          </div>

          <div className="font-mono" style={{ marginTop: 10, borderTop: '1px solid #1e221e', paddingTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            Win Rate: {(data.win_rate * 100).toFixed(2)}% • Signals analyzed: {data.signals_analyzed}
          </div>
        </div>
      )}
    </>
  );
};
