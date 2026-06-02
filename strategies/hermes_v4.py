#!/usr/bin/env python3
"""
strategies/hermes_v4.py — Hermes V4 Trading Strategy.
Ports the existing scoring logic from bot.py into a proper IStrategy.
Uses 8 technical indicators with dynamic weights via LearningEngine.
"""
import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import IStrategy

log = logging.getLogger("hermes.strategy.v4")

# Try to import learning engine
try:
    from learner import LearningEngine
    HAS_LEARNER = True
except ImportError:
    HAS_LEARNER = False


class HermesV4Strategy(IStrategy):
    """
    Hermes V4 Strategy — the refined version from bot.py.

    8 indicators weighted and combined into a single buy/sell score:
      RSI, MACD, Bollinger Bands, ADX, Stochastic, MFI, Volume, Trend

    Config options (passed via config dict):
      - min_score (float): min absolute score to trigger signal (default: 0.25)
      - learning_enabled (bool): use LearningEngine for dynamic weights (default: True)
      - rsi_oversold (int): RSI oversold threshold (default: 30)
      - rsi_overbought (int): RSI overbought threshold (default: 70)
    """

    name = "HermesV4"
    timeframe = "1d"
    min_data_period = 30
    max_open_trades = 3

    # Default weights (used if LearningEngine unavailable)
    DEFAULT_WEIGHTS = {
        "rsi": 2.0,
        "macd": 2.0,
        "bollinger": 1.5,
        "adx": 1.5,
        "stoch": 1.0,
        "mfi": 1.0,
        "volume": 1.5,
        "trend": 2.0,
        "divergences": 2.0,
    }

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.min_score = float(self.config.get("min_score", 0.25))
        self.learning_enabled = bool(self.config.get("learning_enabled", True))
        self.rsi_oversold = int(self.config.get("rsi_oversold", 30))
        self.rsi_overbought = int(self.config.get("rsi_overbought", 70))

        # Learning engine for dynamic weights
        self.learning_engine = None
        if self.learning_enabled and HAS_LEARNER:
            try:
                data_dir = self.config.get("data_dir", "data")
                self.learning_engine = LearningEngine(data_dir=data_dir)
                log.info("LearningEngine loaded for dynamic weights")
            except Exception as e:
                log.warning("LearningEngine init failed: %s", e)

    def _get_weight(self, indicator: str, default: float = 1.0) -> float:
        """Get dynamic weight from LearningEngine, or default."""
        if self.learning_engine is not None:
            try:
                return self.learning_engine.get_weight(indicator)
            except Exception:
                pass
        return self.DEFAULT_WEIGHTS.get(indicator, default)

    # ─── Indicators (100% vectorised, ported from bot.py) ────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_g = gain.ewm(span=period, adjust=False).mean()
        avg_l = loss.ewm(span=period, adjust=False).mean()
        rs = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period, min_periods=period).mean()

    @staticmethod
    def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ef = HermesV4Strategy._ema(close, fast)
        es = HermesV4Strategy._ema(close, slow)
        line = ef - es
        sig = HermesV4Strategy._ema(line, signal)
        hist = line - sig
        return line, sig, hist

    @staticmethod
    def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
        mid = HermesV4Strategy._sma(close, window)
        sd = close.rolling(window, min_periods=window).std()
        upper = mid + sd * n_std
        lower = mid - sd * n_std
        pct = (close - lower) / (upper - lower + 1e-10)
        width = (upper - lower) / (mid + 1e-10)
        return upper, mid, lower, pct, width

    @staticmethod
    def _adx(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        up = high.diff()
        dn = low.diff().abs()
        plus_dm = up.where((up > dn) & (up > 0), 0.0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
        atr = tr.ewm(span=period, adjust=False).mean()
        pdi = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        ndi = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, pdi, ndi, atr

    @staticmethod
    def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, k: int = 14, d: int = 3):
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        kline = 100 * (close - ll) / (hh - ll + 1e-10)
        dline = kline.rolling(d).mean()
        return kline, dline

    @staticmethod
    def _mfi(close: pd.Series, high: pd.Series, low: pd.Series,
             volume: pd.Series, period: int = 14) -> pd.Series:
        typical = (high + low + close) / 3
        mf = typical * volume
        sign = (typical.diff() >= 0).astype(int) * 2 - 1
        pos = mf.where(sign > 0, 0).rolling(period).sum()
        neg = mf.where(sign < 0, 0).rolling(period).sum()
        ratio = pos / neg.replace(0, np.nan)
        return 100 - (100 / (1 + ratio))

    @staticmethod
    def _atr(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _trend_strength(close: pd.Series, period: int = 30) -> dict:
        if len(close) < period:
            return {"trend": "neutral", "strength": 0, "slope": 0}
        x = np.arange(period)
        y = close[-period:].values
        if np.std(y) == 0:
            return {"trend": "neutral", "strength": 0, "slope": 0}
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / y.mean() * 100
        strength = min(abs(normalized_slope) * 10, 100)
        trend = "bullish" if normalized_slope > 0.1 else "bearish" if normalized_slope < -0.1 else "neutral"
        return {"trend": trend, "strength": round(strength, 1), "slope_pct": round(normalized_slope, 3)}

    @staticmethod
    def _find_divergence(close: np.ndarray, rsi: np.ndarray, window: int = 14) -> list:
        divergences = []
        n = len(close)
        for i in range(window + 2, n - 2):
            is_local_low = (close[i] < close[i-1] and close[i] < close[i-2] and
                           close[i] < close[i+1] and close[i] < close[i+2])
            is_local_high = (close[i] > close[i-1] and close[i] > close[i-2] and
                            close[i] > close[i+1] and close[i] > close[i+2])
            price_change = abs(close[i] - close[i-window]) / close[i-window]
            if (is_local_low and close[i] < close[i-window] * 0.98 and
                rsi[i] > rsi[i-window] + 3 and price_change > 0.03):
                divergences.append({"type": "bullish", "price": float(round(close[i], 2)),
                                    "rsi": float(round(rsi[i], 1)),
                                    "strength": "strong" if price_change > 0.08 else "moderate"})
            if (is_local_high and close[i] > close[i-window] * 1.02 and
                rsi[i] < rsi[i-window] - 3 and price_change > 0.03):
                divergences.append({"type": "bearish", "price": float(round(close[i], 2)),
                                    "rsi": float(round(rsi[i], 1)),
                                    "strength": "strong" if price_change > 0.08 else "moderate"})
        return divergences[-3:] if divergences else []

    # ─── IStrategy implementation ─────────────────────────────────

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicator columns.
        Expected columns: close, volume (high/low optional, approximated from close).
        """
        close = df["close"].astype(float)
        volume = df.get("volume", pd.Series(1, index=df.index)).astype(float)
        # Approximate high/low from close dynamics
        high = close.rolling(3, min_periods=1).max()
        low = close.rolling(3, min_periods=1).min()

        # RSI
        df["rsi"] = self._rsi(close)
        # MACD
        macd_line, macd_signal, macd_hist = self._macd(close)
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        # Bollinger
        bb_u, bb_m, bb_l, bb_p, bb_w = self._bollinger(close)
        df["bb_upper"] = bb_u
        df["bb_middle"] = bb_m
        df["bb_lower"] = bb_l
        df["bb_percent"] = bb_p
        df["bb_width"] = bb_w
        # ADX
        adx_s, pdi, ndi, atr_s = self._adx(close, high, low)
        df["adx"] = adx_s
        df["pdi"] = pdi
        df["ndi"] = ndi
        df["atr"] = atr_s
        # Stochastic
        stoch_k, stoch_d = self._stoch(close, high, low)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        # MFI
        df["mfi"] = self._mfi(close, high, low, volume)
        # Volume ratio
        vol_sma = volume.rolling(20).mean().replace(0, np.nan)
        df["vol_ratio"] = (volume / vol_sma)
        # Trend strength
        trend_data = self._trend_strength(close)
        df["trend_direction"] = trend_data["trend"]
        df["trend_strength"] = trend_data["strength"]
        df["trend_slope"] = trend_data["slope_pct"]

        return df

    def populate_entry_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute buy score and set enter_long."""
        df["buy_score"] = 0.0
        df["enter_long"] = 0
        df["buy_reasons"] = ""

        close = df["close"].astype(float)

        # Score computation (vectorised where possible)
        total_score = pd.Series(0.0, index=df.index)
        max_score = pd.Series(0.0, index=df.index)

        # 1. RSI
        w = self._get_weight("rsi", 2.0)
        max_score += w
        rsi_contrib = pd.Series(0.0, index=df.index)
        rsi_contrib.loc[df["rsi"] < self.rsi_oversold] = w
        rsi_contrib.loc[(df["rsi"] >= self.rsi_oversold) & (df["rsi"] < self.rsi_oversold + 10)] = w * 0.6
        rsi_contrib.loc[df["rsi"] > self.rsi_overbought] = -w
        rsi_contrib.loc[(df["rsi"] <= self.rsi_overbought) & (df["rsi"] > self.rsi_overbought - 10)] = -w * 0.6
        total_score += rsi_contrib

        # 2. MACD
        w = self._get_weight("macd", 2.0)
        max_score += w
        macd_contrib = pd.Series(0.0, index=df.index)
        macd_contrib.loc[df["macd"] > df["macd_signal"]] = w
        macd_contrib.loc[df["macd"] <= df["macd_signal"]] = -w
        total_score += macd_contrib

        # 3. Bollinger
        w = self._get_weight("bollinger", 1.5)
        max_score += w
        bb_contrib = pd.Series(0.0, index=df.index)
        bb_contrib.loc[df["bb_percent"] < 0.05] = w
        bb_contrib.loc[(df["bb_percent"] >= 0.05) & (df["bb_percent"] < 0.2)] = w * 0.5
        bb_contrib.loc[df["bb_percent"] > 0.95] = -w
        bb_contrib.loc[(df["bb_percent"] <= 0.95) & (df["bb_percent"] > 0.8)] = -w * 0.5
        total_score += bb_contrib

        # 4. ADX + Direction
        w = self._get_weight("adx", 1.5)
        max_score += w
        adx_contrib = pd.Series(0.0, index=df.index)
        strong_up = (df["adx"] > 25) & (df["pdi"] > df["ndi"])
        strong_dn = (df["adx"] > 25) & (df["ndi"] > df["pdi"])
        weak_up = (df["adx"] > 20) & (df["pdi"] > df["ndi"]) & ~strong_up
        weak_dn = (df["adx"] > 20) & (df["ndi"] > df["pdi"]) & ~strong_dn
        adx_contrib.loc[strong_up] = w
        adx_contrib.loc[strong_dn] = -w
        adx_contrib.loc[weak_up] = w * 0.3
        adx_contrib.loc[weak_dn] = -w * 0.3
        total_score += adx_contrib

        # 5. Stochastic
        w = self._get_weight("stoch", 1.0)
        max_score += w
        stoch_contrib = pd.Series(0.0, index=df.index)
        stoch_contrib.loc[df["stoch_k"] < 20] = w
        stoch_contrib.loc[df["stoch_k"] > 80] = -w
        stoch_contrib.loc[(df["stoch_k"] >= 20) & (df["stoch_k"] < 30)] = w * 0.3
        stoch_contrib.loc[(df["stoch_k"] > 70) & (df["stoch_k"] <= 80)] = -w * 0.3
        total_score += stoch_contrib

        # 6. MFI
        w = self._get_weight("mfi", 1.0)
        max_score += w
        mfi_contrib = pd.Series(0.0, index=df.index)
        mfi_contrib.loc[df["mfi"] < 20] = w
        mfi_contrib.loc[df["mfi"] > 80] = -w
        total_score += mfi_contrib

        # 7. Volume
        w = self._get_weight("volume", 1.5)
        max_score += w
        vol_contrib = pd.Series(0.0, index=df.index)
        vol_contrib.loc[df["vol_ratio"] > 1.5] = w * 0.5
        vol_contrib.loc[df["vol_ratio"] < 0.3] = -w * 0.5
        total_score += vol_contrib

        # 8. Trend
        w = self._get_weight("trend", 2.0)
        max_score += w
        trend_contrib = pd.Series(0.0, index=df.index)
        trend_contrib.loc[df["trend_direction"] == "bullish"] = w * (df["trend_strength"] / 100)
        trend_contrib.loc[df["trend_direction"] == "bearish"] = -w * (df["trend_strength"] / 100)
        total_score += trend_contrib

        # Normalize
        normalized = total_score / max_score.replace(0, np.nan)
        normalized = normalized.fillna(0).clip(-1, 1)

        df["buy_score"] = normalized
        df["enter_long"] = (normalized >= self.min_score).astype(int)
        df["buy_score_raw"] = total_score
        df["buy_max_score"] = max_score

        return df

    def populate_exit_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sell score and set exit_long."""
        df["sell_score"] = 0.0
        df["exit_long"] = 0

        # Reuse buy_score but with reversed threshold
        if "buy_score" in df.columns:
            normalized = df["buy_score"]
            df["sell_score"] = normalized
            df["exit_long"] = (normalized <= -self.min_score).astype(int)

        return df
