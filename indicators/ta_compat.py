"""
Import shim for ``pandas_ta``.

The ``pandas-ta`` distribution on PyPI currently publishes builds that require
Python 3.12 or newer. For Python 3.11 and below, this project falls back to
``pandas-ta-classic``, which exposes the same functional API (``rsi``, ``ema``,
``macd``, ``bbands``, ...).
"""

from __future__ import annotations

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - exercised on Python <3.12 in CI
    import pandas_ta_classic as ta

__all__ = ["ta"]
