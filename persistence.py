#!/usr/bin/env python3
"""
persistence.py — SQLite-backed trade tracking for Hermes Trading Bot.
Stores trades, orders, and performance metrics persistently.
Inspired by Freqtrade's persistence layer.
"""
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("hermes.persistence")


class TradeDatabase:
    """
    SQLite-backed trade database with thread-safe access.
    Stores open trades, closed trades, orders, and performance metrics.
    """

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL DEFAULT 'long',
                    status TEXT NOT NULL DEFAULT 'open',
                    amount REAL NOT NULL,
                    rate REAL NOT NULL,
                    stake_amount REAL NOT NULL,
                    open_date TEXT NOT NULL,
                    close_date TEXT,
                    close_rate REAL,
                    profit REAL,
                    profit_ratio REAL,
                    stoploss REAL,
                    final_balance REAL,
                    sell_reason TEXT,
                    strategy TEXT,
                    timeframe TEXT,
                    exchange TEXT,
                    is_dry_run INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL,
                    order_id TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL,
                    amount REAL,
                    filled REAL,
                    status TEXT DEFAULT 'open',
                    order_date TEXT NOT NULL,
                    FOREIGN KEY (trade_id) REFERENCES trades(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0,
                    total_profit_ratio REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    balance REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    score REAL NOT NULL,
                    price REAL NOT NULL,
                    reasons TEXT,
                    timestamp TEXT NOT NULL,
                    acted INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_coin ON trades(coin_id);
                CREATE INDEX IF NOT EXISTS idx_orders_trade ON orders(trade_id);
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            """)
            conn.commit()

    # ─── Trade CRUD ───────────────────────────────────────────────

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return {}
        return dict(row)

    def add_trade(self, coin_id: str, pair: str, amount: float, rate: float,
                  stake_amount: float, side: str = "long",
                  strategy: str = "", timeframe: str = "1d",
                  exchange: str = "paper", stoploss: float = -0.05,
                  is_dry_run: bool = True) -> int:
        """Record a new trade. Returns the trade ID."""
        with self._lock:
            conn = self._get_conn()
            now = datetime.utcnow().isoformat()
            cur = conn.execute(
                """INSERT INTO trades
                   (coin_id, pair, side, status, amount, rate, stake_amount,
                    open_date, stoploss, strategy, timeframe, exchange, is_dry_run)
                   VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (coin_id, pair, side, amount, rate, stake_amount,
                 now, stoploss, strategy, timeframe, exchange, 1 if is_dry_run else 0),
            )
            conn.commit()
            trade_id = cur.lastrowid
            log.info("Trade #%d opened: %s %s %.4f @ %.2f", trade_id, side.upper(), pair, amount, rate)
            return trade_id

    def close_trade(self, trade_id: int, close_rate: float, sell_reason: str = "signal",
                    final_balance: Optional[float] = None):
        """Close an open trade with profit calculations."""
        with self._lock:
            conn = self._get_conn()
            trade = self.get_trade(trade_id)
            if not trade or trade["status"] != "open":
                log.warning("Trade #%d not open or not found", trade_id)
                return False

            now = datetime.utcnow().isoformat()
            amount = trade["amount"]
            open_rate = trade["rate"]
            is_long = trade["side"] == "long"

            # Profit calculation
            if is_long:
                profit_ratio = (close_rate - open_rate) / open_rate
            else:
                profit_ratio = (open_rate - close_rate) / open_rate

            profit = profit_ratio * trade["stake_amount"]
            balance = final_balance or (trade["stake_amount"] + profit)

            conn.execute(
                """UPDATE trades SET
                   status='closed', close_date=?, close_rate=?,
                   profit=?, profit_ratio=?, final_balance=?,
                   sell_reason=?, updated_at=datetime('now')
                   WHERE id=?""",
                (now, close_rate, round(profit, 2), round(profit_ratio, 6),
                 round(balance, 2), sell_reason, trade_id),
            )
            conn.commit()
            log.info("Trade #%d closed: profit=%.2f (%.2f%%) reason=%s",
                     trade_id, profit, profit_ratio * 100, sell_reason)
            return True

    def update_trade(self, trade_id: int, **kwargs):
        """Update arbitrary fields on a trade."""
        allowed = {"stoploss", "amount", "rate"}
        sets = []
        values = []
        for key, val in kwargs.items():
            if key in allowed:
                sets.append(f"{key}=?")
                values.append(val)
        if not sets:
            return
        sets.append("updated_at=datetime('now')")
        values.append(trade_id)
        with self._lock:
            conn = self._get_conn()
            conn.execute(f"UPDATE trades SET {', '.join(sets)} WHERE id=?", values)
            conn.commit()

    def get_trade(self, trade_id: int) -> dict:
        """Get a single trade by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        return self._row_to_dict(row)

    def get_open_trades(self, coin_id: Optional[str] = None) -> list[dict]:
        """Get all open trades, optionally filtered by coin."""
        conn = self._get_conn()
        if coin_id:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' AND coin_id=? ORDER BY open_date DESC",
                (coin_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY open_date DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        """Get most recent closed trades."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status='closed' ORDER BY close_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_trades(self, limit: int = 100) -> list[dict]:
        """Get all trades, most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY open_date DESC LIMIT ?", (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ─── Orders ───────────────────────────────────────────────────

    def add_order(self, trade_id: int, order_id: str, order_type: str,
                  side: str, price: float, amount: float) -> int:
        with self._lock:
            conn = self._get_conn()
            now = datetime.utcnow().isoformat()
            cur = conn.execute(
                """INSERT INTO orders (trade_id, order_id, order_type, side, price, amount, order_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, order_id, order_type, side, price, amount, now),
            )
            conn.commit()
            return cur.lastrowid

    def update_order(self, order_id: str, filled: float, status: str):
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE orders SET filled=?, status=? WHERE order_id=?",
                (filled, status, order_id),
            )
            conn.commit()

    def get_orders_for_trade(self, trade_id: int) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM orders WHERE trade_id=? ORDER BY order_date", (trade_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ─── Performance ──────────────────────────────────────────────

    def record_performance(self, balance: float, max_dd: float = 0.0,
                           sharpe: float = 0.0):
        """Record daily performance snapshot."""
        with self._lock:
            conn = self._get_conn()
            today = datetime.utcnow().strftime("%Y-%m-%d")

            stats = self.get_stats()
            total = stats["total_trades"]
            wins = stats["winning_trades"]
            losses = stats["losing_trades"]
            total_profit = stats["total_profit"]
            win_rate = wins / total if total > 0 else 0.0

            conn.execute(
                """INSERT OR REPLACE INTO performance
                   (date, total_trades, winning_trades, losing_trades,
                    total_profit, total_profit_ratio, win_rate,
                    max_drawdown, sharpe_ratio, balance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (today, total, wins, losses, round(total_profit, 2),
                 round(stats["total_profit_ratio"], 6), round(win_rate, 4),
                 round(max_dd, 4), round(sharpe, 4), round(balance, 2)),
            )
            conn.commit()

    def get_stats(self) -> dict:
        """Get aggregate trading statistics."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN status='closed' AND profit_ratio > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN status='closed' AND profit_ratio <= 0 THEN 1 ELSE 0 END) as losing_trades,
                COALESCE(SUM(CASE WHEN status='closed' THEN profit ELSE 0 END), 0) as total_profit,
                COALESCE(SUM(CASE WHEN status='closed' THEN profit_ratio ELSE 0 END), 0) as total_profit_ratio,
                AVG(CASE WHEN status='closed' THEN profit_ratio ELSE NULL END) as avg_profit_ratio,
                COALESCE(SUM(CASE WHEN status='closed' AND profit_ratio > 0 THEN 1 ELSE 0 END) * 1.0 /
                    NULLIF(SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END), 0), 0) as win_rate
            FROM trades
        """).fetchone()
        result = self._row_to_dict(row)
        for k in result:
            if result[k] is None:
                result[k] = 0
        return result

    def get_profit_summary(self) -> dict:
        """Summary suitable for API display."""
        stats = self.get_stats()
        trades = self.get_all_trades(limit=500)
        closed = [t for t in trades if t.get("status") == "closed"]

        profits = [t.get("profit", 0) for t in closed if t.get("profit") is not None]
        profit_ratios = [t.get("profit_ratio", 0) for t in closed if t.get("profit_ratio") is not None]

        return {
            "total_trades": stats["total_trades"],
            "open_trades": stats["open_trades"],
            "winning_trades": stats["winning_trades"],
            "losing_trades": stats["losing_trades"],
            "win_rate": round(stats["win_rate"], 4),
            "total_profit": round(stats["total_profit"], 2),
            "total_profit_ratio": round(stats["total_profit_ratio"], 6),
            "avg_profit": round(float(np.mean(profits)) if profits else 0, 2),
            "avg_profit_ratio": round(float(np.mean(profit_ratios)) if profit_ratios else 0, 6),
            "best_trade": round(float(max(profits)) if profits else 0, 2),
            "worst_trade": round(float(min(profits)) if profits else 0, 2),
            "profit_factor": round(
                abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0))
                if any(p < 0 for p in profits) else float("inf"), 2
            ),
        }

    # ─── Signals ──────────────────────────────────────────────────

    def log_signal(self, coin_id: str, signal: str, score: float,
                   price: float, reasons: list = None, acted: bool = False):
        with self._lock:
            conn = self._get_conn()
            now = datetime.utcnow().isoformat()
            conn.execute(
                """INSERT INTO signals (coin_id, signal, score, price, reasons, timestamp, acted)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (coin_id, signal, score, price,
                 json.dumps(reasons or []), now, 1 if acted else 0),
            )
            conn.commit()

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ─── Maintenance ──────────────────────────────────────────────

    def vacuum(self):
        """Vacuum the database to reclaim space."""
        with self._lock:
            self._get_conn().execute("VACUUM")
            log.info("Database vacuumed")

    def close(self):
        """Close all connections."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Import numpy for stats
try:
    import numpy as np
except ImportError:
    import math
    class _DummyNumpy:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        @staticmethod
        def max(x):
            return max(x) if x else 0
        @staticmethod
        def min(x):
            return min(x) if x else 0
    np = _DummyNumpy()
