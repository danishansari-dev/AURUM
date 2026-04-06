"""SQLite persistence for OHLCV bars and precomputed indicators."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class PriceStorage:
    """
    SQLite-backed storage for gold price bars and cached indicator columns.

    The database contains:

    - ``price_data``: daily (or bar) OHLCV keyed by date.
    - ``indicator_cache``: aligned technical indicators keyed by date.
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Open (or create) a SQLite database at the given path.

        Args:
            db_path: Filesystem path to the ``.sqlite`` (or similar) database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS price_data (
                date TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indicator_cache (
                date TEXT PRIMARY KEY,
                rsi REAL,
                ema20 REAL,
                ema50 REAL,
                sma200 REAL,
                macd_line REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_lower REAL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_data_date ON price_data(date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_indicator_cache_date ON indicator_cache(date)"
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "PriceStorage":
        """Support context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Support context manager exit by closing the connection."""
        self.close()

    def save_prices(self, df: pd.DataFrame) -> None:
        """
        Upsert OHLCV rows from ``df`` into ``price_data``.

        Args:
            df: DataFrame with DatetimeIndex (or ``date`` column) and columns
                ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        frame = df.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
            date_col = frame.columns[0]
            frame = frame.rename(columns={date_col: "date"})
        elif "date" not in frame.columns:
            raise ValueError("DataFrame must have DatetimeIndex or a 'date' column.")

        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.strftime("%Y-%m-%d")
        cols = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"save_prices missing columns: {missing}")

        rows = list(frame[cols].itertuples(index=False, name=None))
        self._conn.executemany(
            """
            INSERT INTO price_data (date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
            """,
            rows,
        )
        self._conn.commit()

    def load_prices(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load stored OHLCV bars, optionally bounded by ``start`` / ``end``.

        Args:
            start: Inclusive lower date bound (string or timestamp).
            end: Inclusive upper date bound (string or timestamp).

        Returns:
            DataFrame with UTC DatetimeIndex named ``date`` and OHLCV columns.
        """
        query = "SELECT date, open, high, low, close, volume FROM price_data WHERE 1=1"
        params: list[Any] = []
        if start is not None:
            query += " AND date >= ?"
            params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
        if end is not None:
            query += " AND date <= ?"
            params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
        query += " ORDER BY date ASC"

        cur = self._conn.cursor()
        cur.execute(query, params)
        fetched = cur.fetchall()
        if not fetched:
            empty_idx = pd.DatetimeIndex([], name="date", tz="UTC")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=empty_idx,
            )

        rows = [dict(r) for r in fetched]
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.set_index("date").sort_index()
        return frame

    def save_indicators(self, df: pd.DataFrame) -> None:
        """
        Upsert indicator rows into ``indicator_cache``.

        Args:
            df: DataFrame with DatetimeIndex (or ``date`` column) and indicator
                columns matching the table schema (sparse NaNs allowed).
        """
        frame = df.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
            first = frame.columns[0]
            frame = frame.rename(columns={first: "date"})
        elif "date" not in frame.columns:
            raise ValueError("DataFrame must have DatetimeIndex or a 'date' column.")

        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.strftime("%Y-%m-%d")

        expected = [
            "date",
            "rsi",
            "ema20",
            "ema50",
            "sma200",
            "macd_line",
            "macd_signal",
            "bb_upper",
            "bb_lower",
        ]
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"save_indicators missing columns: {missing}")

        tuples = list(frame[expected].itertuples(index=False, name=None))
        self._conn.executemany(
            """
            INSERT INTO indicator_cache (
                date, rsi, ema20, ema50, sma200,
                macd_line, macd_signal, bb_upper, bb_lower
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                rsi=excluded.rsi,
                ema20=excluded.ema20,
                ema50=excluded.ema50,
                sma200=excluded.sma200,
                macd_line=excluded.macd_line,
                macd_signal=excluded.macd_signal,
                bb_upper=excluded.bb_upper,
                bb_lower=excluded.bb_lower
            """,
            tuples,
        )
        self._conn.commit()

    def load_indicators(
        self,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Load cached indicators, optionally bounded by ``start`` / ``end``.

        Args:
            start: Inclusive lower date bound.
            end: Inclusive upper date bound.

        Returns:
            DataFrame with UTC DatetimeIndex and indicator columns.
        """
        query = (
            "SELECT date, rsi, ema20, ema50, sma200, macd_line, macd_signal, "
            "bb_upper, bb_lower FROM indicator_cache WHERE 1=1"
        )
        params: list[Any] = []
        if start is not None:
            query += " AND date >= ?"
            params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
        if end is not None:
            query += " AND date <= ?"
            params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
        query += " ORDER BY date ASC"

        cur = self._conn.cursor()
        cur.execute(query, params)
        fetched = cur.fetchall()
        if not fetched:
            empty_idx = pd.DatetimeIndex([], name="date", tz="UTC")
            return pd.DataFrame(
                columns=[
                    "rsi",
                    "ema20",
                    "ema50",
                    "sma200",
                    "macd_line",
                    "macd_signal",
                    "bb_upper",
                    "bb_lower",
                ],
                index=empty_idx,
            )

        rows = [dict(r) for r in fetched]
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.set_index("date").sort_index()
        return frame


__all__ = ["PriceStorage"]
