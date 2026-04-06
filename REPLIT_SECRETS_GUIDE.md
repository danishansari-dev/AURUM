# 🔐 Replit Secrets Configuration Guide — AURUM

AURUM reads API keys from **environment variables**, never from hardcoded
values. On Replit, these are managed via the **Secrets** tab (🔒 icon in the
left sidebar).

---

## Required Secrets

| Key | Purpose | How to get it |
|-----|---------|---------------|
| `ANTHROPIC_API_KEY` | Claude AI-fallback parser (used when regex can't parse complex strategy text) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) — free credits on first sign-up |
| `ALPHA_VANTAGE_API_KEY` | Real-time Gold (XAU/USD) spot price via Alpha Vantage API | [www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) — free, instant |

---

## How to add Secrets on Replit

1. Open your AURUM Repl.
2. Click the **🔒 Secrets** tab in the left sidebar (or press `Ctrl+Shift+S`).
3. Click **+ New Secret**.
4. Enter the key name exactly as shown above (e.g. `ANTHROPIC_API_KEY`).
5. Paste your API key value.
6. Click **Add Secret**.

Secrets are injected as environment variables and are available via
`os.environ["ANTHROPIC_API_KEY"]` in Python. They are **not** committed to Git.

---

## What happens without these keys?

| Missing Key | Behaviour |
|-------------|-----------|
| `ANTHROPIC_API_KEY` | The AI parser fallback is disabled. Regex-only parsing still works for standard strategy patterns. |
| `ALPHA_VANTAGE_API_KEY` | Real-time price fetching is disabled. The dashboard falls back to the latest yfinance close (~15 min delay). |

The dashboard will **always** start — missing keys degrade gracefully rather
than crashing.

---

## Local Development

For local development outside Replit, export the keys in your terminal:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."
export ALPHA_VANTAGE_API_KEY="YOUR_AV_KEY"

# Windows (cmd)
set ANTHROPIC_API_KEY=sk-ant-...
set ALPHA_VANTAGE_API_KEY=YOUR_AV_KEY

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:ALPHA_VANTAGE_API_KEY = "YOUR_AV_KEY"
```

Or create a `.env` file in the project root (already in `.gitignore`):

```env
ANTHROPIC_API_KEY=sk-ant-...
ALPHA_VANTAGE_API_KEY=YOUR_AV_KEY
```
