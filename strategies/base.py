#!/usr/bin/env python3
"""
strategies/base.py — Abstract Strategy Interface (Freqtrade-compatible pattern).
Defines IStrategy that all strategies must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Unified trading signal."""
    coin_id: str
    signal: str  # "BUY" | "SELL" | "NEUTRAL"
    score: float  # -1.0 to 1.0
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    reasons: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal == "BUY" and self.score >= 0

    @property
    def is_sell(self) -> bool:
        return self.signal == "SELL" or (self.signal == "BUY" and self.score < 0)

    @property
    def is_neutral(self) -> bool:
        return self.signal == "NEUTRAL"


@dataclass
class StrategyResult:
    """Result of a full strategy analysis pass."""
    signals: list[Signal]
    timestamp: datetime = field(default_factory=datetime.now)
    total_analyzed: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    neutral_signals: int = 0
    market_regime: str = "unknown"
    metadata: dict = field(default_factory=dict)


class IStrategy(ABC):
    """
    Abstract base strategy — all strategies must inherit from this.
    Freqtrade-compatible interface pattern.

    Lifecycle:
        1. __init__() — set strategy metadata
        2. populate_indicators(df) — add indicator columns
        3. populate_entry_trend(df) — define buy signals
        4. populate_exit_trend(df) — define sell signals
        5. analyze(df) — full pipeline (convenience wrapper)
    """

    # Strategy metadata — override in subclasses
    name: str = "BaseStrategy"
    timeframe: str = "1d"
    min_data_period: int = 30
    max_open_trades: int = 3

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the DataFrame.
        Must return the same DataFrame with added columns.
        """
        ...

    @abstractmethod
    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define buy / entry conditions.
        Add 'enter_long' or 'enter_short' boolean columns (1/0).
        """
        ...

    @abstractmethod
    def populate_exit_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Define sell / exit conditions.
        Add 'exit_long' or 'exit_short' boolean columns (1/0).
        """
        ...

    def populate_any_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional: populate both entry and exit trends at once."""
        df = self.populate_entry_trend(df)
        df = self.populate_exit_trend(df)
        return df

    def analyze(self, df: pd.DataFrame) -> dict:
        """Full analysis pipeline: indicators → entry → exit.
        Returns a dict with signal info for the last candle.
        """
        if df.empty or len(df) < self.min_data_period:
            return {"signal": "NEUTRAL", "score": 0.0, "confidence": "LOW",
                    "reasons": ["Insufficient data"]}

        df = self.populate_indicators(df.copy())
        df = self.populate_any_trends(df)

        last = df.iloc[-1]
        score = 0.0
        reasons = []

        # Check entry signals
        if last.get("enter_long", 0) == 1:
            score = last.get("buy_score", 0.5)
            signal = "BUY"
            reasons.append("Entry long signal triggered")
        elif last.get("enter_short", 0) == 1:
            score = last.get("sell_score", -0.5)
            signal = "SELL"
            reasons.append("Entry short signal triggered")
        elif last.get("exit_long", 0) == 1:
            score = last.get("sell_score", -0.5)
            signal = "SELL"
            reasons.append("Exit long signal triggered")
        elif last.get("exit_short", 0) == 1:
            score = last.get("buy_score", 0.5)
            signal = "BUY"
            reasons.append("Exit short signal triggered")
        else:
            signal = "NEUTRAL"

        confidence = "HIGH" if abs(score) > 0.5 else "MEDIUM" if abs(score) > 0.25 else "LOW"

        return {
            "signal": signal,
            "score": round(float(score), 4),
            "confidence": confidence,
            "reasons": reasons[:5],
            "indicators": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in last.items()
                          if k not in ("enter_long", "enter_short", "exit_long", "exit_short")},
        }

    def get_signal(self, df: pd.DataFrame) -> Signal:
        """Return a Signal dataclass from analysis."""
        result = self.analyze(df)
        current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0
        return Signal(
            coin_id=self.name,
            signal=result["signal"],
            score=result["score"],
            confidence=result["confidence"],
            price=current_price,
            reasons=result.get("reasons", []),
            metadata=result.get("indicators", {}),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} [{self.timeframe}]>"
