#!/usr/bin/env python3
"""
Hermes Trading Bot v2 — analyse marché crypto avancée
Inspiré de Freqtrade, qtpylib, Jesse
Indicateurs: RSI, MACD, SMA/EMA, Bollinger Bands, ADX, Stochastique,
             MFI, ATR, Heikin Ashi, Support/Resistance, Volume Profile
Signaux: système de score multicritères avec pondération

Usage:
  python3 bot.py                    # Analyse BTC, ETH
  python3 bot.py --coin all         # Top 20 coins
  python3 bot.py --coin solana --days 90
  python3 bot.py --html             # Rapport HTML
  python3 bot.py --loop 60          # Boucle toutes les 60min
"""
import json, time, sys, os, argparse, math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ─── Configuration ───────────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)

REQUEST_DELAY = 6.0  # secondes entre requetes (rate limit CoinGecko: 10-30/min)

TOP_50 = [
    "bitcoin","ethereum","ripple","cardano","solana","polkadot","dogecoin",
    "avalanche","chainlink","polygon","litecoin","uniswap","stellar","monero",
    "filecoin","vechain","theta","eos","aave","maker","algorand","tezos",
    "near","hedera","cosmos","internet-computer","aptos","sui","optimism",
    "arbitrum","pepe","injective","fetch-ai","render","immutable","sei",
    "celestia","kaspa","flow","gala","fantom","kucoin-token","compound",
    "curve-dao-token","zcash","quant","bitget-token","dydx","pyth-network"
]

# Ponderation des indicateurs pour le signal final
WEIGHTS = {
    "rsi": 2.0,
    "macd": 2.0,
    "bbands": 1.5,
    "adx": 1.0,
    "stoch": 1.0,
    "mfi": 1.0,
    "ema_trend": 1.5,
    "volume": 1.0,
    "heikin_ashi": 1.0,
}

# ─── Data fetching ─────────────────────────────────────────────────

def api_get(url, retries=3):
    """Requete API avec retry et rate limit"""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  ⏳ Rate limited, attente {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return {}

def fetch_market_data(coin_id, days=60):
    """Recupere prix + volumes depuis CoinGecko (endpoint fiable)"""
    # CoinGecko free API: 10-30 req/min. market_chart est le plus stable.
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    data = api_get(url)
    if not data or "prices" not in data or not data["prices"]:
        return pd.DataFrame()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
    volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])

    df = prices.merge(volumes, on="timestamp", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Synthetise OHLC a partir des close prices
    df["open"] = df["close"].shift(1)
    df.loc[df.index[0], "open"] = df["close"].iloc[0]
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"] = df["close"].rolling(3, min_periods=1).min()

    # Resample quotidien pour avoir des bougies propres
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if len(df) > 48:  # Plus d'un jour de donnees
        df = df.resample("1D").agg(ohlc_dict).dropna()

    return df

def fetch_coin_info(coin_id):
    """Infos supplementaires"""
    url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
    return api_get(url)

# ─── Indicateurs techniques ──────────────────────────────────────────

def crossed_above(series1, series2):
    """Detection de croisement haussier"""
    if isinstance(series2, (int, float)):
        series2_val = series2
        return (series1 > series2_val) & (series1.shift(1) <= series2_val)
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def crossed_below(series1, series2):
    """Detection de croisement baissier"""
    if isinstance(series2, (int, float)):
        series2_val = series2
        return (series1 < series2_val) & (series1.shift(1) >= series2_val)
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

def calc_rsi(series, period=14):
    """RSI - Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(period, min_periods=period).mean()

def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD - Moving Average Convergence Divergence"""
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger_bands(series, window=20, std_dev=2):
    """Bollinger Bands"""
    sma = calc_sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bb_percent = (series - lower) / (upper - lower)
    bb_width = (upper - lower) / sma
    return upper, sma, lower, bb_percent, bb_width

def calc_adx(df, period=14):
    """ADX - Average Directional Index (force de tendance)"""
    high, low, close = df["high"].values, df["low"].values, df["close"].values

    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    tr = np.zeros_like(close)

    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            minus_dm[i] = down_move if down_move > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = pd.Series(tr).rolling(period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(period).mean()

    return adx, pd.Series(plus_di), pd.Series(minus_di), pd.Series(atr)

def calc_stoch(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K et %D"""
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def calc_mfi(df, period=14):
    """MFI - Money Flow Index"""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df.get("volume", pd.Series(1, index=df.index))
    sign = (typical.diff() >= 0).astype(int) * 2 - 1  # 1 pour up, -1 pour down

    pos_flow = money_flow.where(sign > 0, 0).rolling(period).sum()
    neg_flow = money_flow.where(sign < 0, 0).rolling(period).sum()
    mfr = pos_flow / neg_flow.replace(0, 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def calc_heikin_ashi(df):
    """Heikin Ashi candles"""
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["ha_open"] = ((df["open"] + df["close"]) / 2).shift(1).bfill()
    ha["ha_high"] = ha[["high", "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"] = ha[["low", "ha_open", "ha_close"]].min(axis=1)
    ha["ha_trend"] = np.where(ha["ha_close"] > ha["ha_open"], 1, -1)
    return ha

def calc_atr(df, period=14):
    """ATR - Average True Range"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_volume_profile(prices, volumes, num_zones=5):
    """Volume profile - prix les plus negociés"""
    if len(prices) < 10 or len(volumes) < 10:
        return {}
    price_min, price_max = min(prices), max(prices)
    zone_size = (price_max - price_min) / num_zones
    zones = {}
    for i in range(num_zones):
        lo = price_min + i * zone_size
        hi = lo + zone_size
        mask = (prices >= lo) & (prices < hi)
        vol = volumes[mask].sum() if volumes[mask].any() else 0
        zone_label = f"{lo:.4f}-{hi:.4f}"
        zones[zone_label] = float(vol)
    return zones

# ─── Analyse ─────────────────────────────────────────────────────────

class CoinAnalysis:
    """Analyse complete d'une crypto"""

    def __init__(self, coin_id, ohlc_df, price_df):
        self.coin_id = coin_id
        self.ohlc = ohlc_df
        self.prices = price_df
        self.info = {}
        self.indicators = {}
        self.signals = {}
        self.result = {}

    def compute_indicators(self):
        """Calcule tous les indicateurs techniques"""
        df = self.ohlc.copy()
        if df.empty or len(df) < 50:
            return False

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Prix et variation
        current_price = float(close.iloc[-1])
        price_24h = float(close.iloc[-2]) if len(close) > 1 else current_price
        change_24h = ((current_price - price_24h) / price_24h) * 100

        # --- RSI ---
        rsi = calc_rsi(close, 14)

        # --- EMA ---
        ema9 = calc_ema(close, 9)
        ema21 = calc_ema(close, 21)
        ema50 = calc_ema(close, 50) if len(close) >= 50 else pd.Series(index=close.index)
        ema200 = calc_ema(close, 200) if len(close) >= 200 else pd.Series(index=close.index)

        # --- MACD ---
        macd_line, macd_signal, macd_hist = calc_macd(close)

        # --- Bollinger Bands ---
        bb_up, bb_mid, bb_low, bb_percent, bb_width = calc_bollinger_bands(close)

        # --- ADX ---
        adx_series, plus_di, minus_di, atr_series = calc_adx(df)

        # --- Stochastique ---
        stoch_k, stoch_d = calc_stoch(high, low, close)

        # --- MFI ---
        mfi = calc_mfi(df)

        # --- Heikin Ashi ---
        ha = calc_heikin_ashi(df)

        # --- Volume ---
        volumes = df.get("volume", pd.Series(1, index=df.index))
        vol_sma = volumes.rolling(20).mean()
        vol_ratio = volumes / vol_sma.replace(0, np.nan)

        # --- ATR pour volatilite ---
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else 0
        atr_pct = (atr_val / current_price * 100) if current_price > 0 else 0

        # --- Support / Resistance ---
        lookback = min(50, len(close))
        recent_low = float(low.iloc[-lookback:].min())
        recent_high = float(high.iloc[-lookback:].max())

        # Dernieres valeurs
        self.indicators = {
            "current_price": current_price,
            "change_24h": round(change_24h, 2),
            "rsi": round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else None,
            "ema9": float(ema9.iloc[-1]) if pd.notna(ema9.iloc[-1]) else None,
            "ema21": float(ema21.iloc[-1]) if pd.notna(ema21.iloc[-1]) else None,
            "ema50": float(ema50.iloc[-1]) if not ema50.empty and pd.notna(ema50.iloc[-1]) else None,
            "ema200": float(ema200.iloc[-1]) if not ema200.empty and pd.notna(ema200.iloc[-1]) else None,
            "macd": float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None,
            "macd_signal": float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else None,
            "macd_hist": float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None,
            "bb_upper": float(bb_up.iloc[-1]) if pd.notna(bb_up.iloc[-1]) else None,
            "bb_mid": float(bb_mid.iloc[-1]) if pd.notna(bb_mid.iloc[-1]) else None,
            "bb_lower": float(bb_low.iloc[-1]) if pd.notna(bb_low.iloc[-1]) else None,
            "bb_percent": round(float(bb_percent.iloc[-1]), 3) if pd.notna(bb_percent.iloc[-1]) else None,
            "bb_width": round(float(bb_width.iloc[-1]), 3) if pd.notna(bb_width.iloc[-1]) else None,
            "adx": round(float(adx_series.iloc[-1]), 2) if pd.notna(adx_series.iloc[-1]) else None,
            "plus_di": round(float(plus_di.iloc[-1]), 2) if pd.notna(plus_di.iloc[-1]) else None,
            "minus_di": round(float(minus_di.iloc[-1]), 2) if pd.notna(minus_di.iloc[-1]) else None,
            "stoch_k": round(float(stoch_k.iloc[-1]), 2) if pd.notna(stoch_k.iloc[-1]) else None,
            "stoch_d": round(float(stoch_d.iloc[-1]), 2) if pd.notna(stoch_d.iloc[-1]) else None,
            "mfi": round(float(mfi.iloc[-1]), 2) if pd.notna(mfi.iloc[-1]) else None,
            "atr": round(atr_val, 4),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio": round(float(vol_ratio.iloc[-1]), 2) if pd.notna(vol_ratio.iloc[-1]) else None,
            "support": round(recent_low, 6),
            "resistance": round(recent_high, 6),
            "ha_trend": int(ha["ha_trend"].iloc[-1]) if not ha.empty else 0,
            "ha_streak": self._count_ha_streak(ha),
        }

        return True

    def _count_ha_streak(self, ha):
        """Compte le nombre de bougies HA consecutives dans la meme direction"""
        if len(ha) < 3:
            return 0
        trend = ha["ha_trend"].values
        last = trend[-1]
        count = 0
        for i in range(len(trend) - 1, -1, -1):
            if trend[i] == last:
                count += 1
            else:
                break
        return count

    def generate_signals(self):
        """Genere les signaux avec score pondere"""
        ind = self.indicators
        score = 0.0
        max_score = 0.0
        reasons = []
        details = {}

        # ─── RSI (2 pts) ───
        w = WEIGHTS["rsi"]
        max_score += w
        if ind["rsi"] is not None:
            if ind["rsi"] < 30:
                score += w; reasons.append(f"RSI survente ({ind['rsi']})")
                details["rsi"] = {"signal": "bullish", "weight": w, "value": ind["rsi"]}
            elif ind["rsi"] < 40:
                score += w * 0.5; reasons.append(f"RSI bas ({ind['rsi']})")
                details["rsi"] = {"signal": "slight_bullish", "weight": w, "value": ind["rsi"]}
            elif ind["rsi"] > 70:
                score -= w; reasons.append(f"RSI surachat ({ind['rsi']})")
                details["rsi"] = {"signal": "bearish", "weight": w, "value": ind["rsi"]}
            elif ind["rsi"] > 60:
                score -= w * 0.5; reasons.append(f"RSI haut ({ind['rsi']})")
                details["rsi"] = {"signal": "slight_bearish", "weight": w, "value": ind["rsi"]}
            else:
                details["rsi"] = {"signal": "neutral", "weight": w, "value": ind["rsi"]}

        # ─── MACD (2 pts) ───
        w = WEIGHTS["macd"]
        max_score += w
        if ind["macd"] is not None and ind["macd_signal"] is not None:
            if ind["macd"] > ind["macd_signal"]:
                score += w; reasons.append(f"MACD > Signal (haussier)")
                details["macd"] = {"signal": "bullish", "weight": w}
            else:
                score -= w; reasons.append(f"MACD < Signal (baissier)")
                details["macd"] = {"signal": "bearish", "weight": w}
            if ind["macd_hist"] is not None:
                if ind["macd_hist"] > 0 and ind["macd_hist"] > details.get("prev_macd_hist", 0):
                    score += w * 0.3
                    reasons.append(f"Histogramme MACD croissant")

        # ─── Bollinger Bands (1.5 pts) ───
        w = WEIGHTS["bbands"]
        max_score += w
        if ind["bb_percent"] is not None:
            if ind["bb_percent"] < 0.05:
                score += w; reasons.append(f"Prix touche bande basse BB (rebond potentiel)")
                details["bbands"] = {"signal": "bullish", "weight": w}
            elif ind["bb_percent"] < 0.2:
                score += w * 0.5; reasons.append(f"Prix proche bande basse BB")
                details["bbands"] = {"signal": "slight_bullish", "weight": w}
            elif ind["bb_percent"] > 0.95:
                score -= w; reasons.append(f"Prix touche bande haute BB (revers potentiel)")
                details["bbands"] = {"signal": "bearish", "weight": w}
            elif ind["bb_percent"] > 0.8:
                score -= w * 0.5; reasons.append(f"Prix proche bande haute BB")
                details["bbands"] = {"signal": "slight_bearish", "weight": w}
            else:
                details["bbands"] = {"signal": "neutral", "weight": w}

        # ─── ADX — force de tendance (1 pt) ───
        w = WEIGHTS["adx"]
        max_score += w
        if ind["adx"] is not None:
            if ind["adx"] > 25:
                # Tendance forte
                if ind["plus_di"] is not None and ind["minus_di"] is not None:
                    if ind["plus_di"] > ind["minus_di"]:
                        score += w; reasons.append(f"ADX forte tendance HAUSSIERE ({ind['adx']})")
                        details["adx"] = {"signal": "bullish", "weight": w}
                    else:
                        score -= w; reasons.append(f"ADX forte tendance BAISSIERE ({ind['adx']})")
                        details["adx"] = {"signal": "bearish", "weight": w}
            elif ind["adx"] > 20:
                details["adx"] = {"signal": "neutral", "weight": w, "trend_strength": "faible"}
            else:
                details["adx"] = {"signal": "neutral", "weight": w, "trend_strength": "aucune"}

        # ─── Stochastique (1 pt) ───
        w = WEIGHTS["stoch"]
        max_score += w
        if ind["stoch_k"] is not None and ind["stoch_d"] is not None:
            if ind["stoch_k"] < 20 and ind["stoch_d"] < 20:
                score += w; reasons.append(f"Stochastique survente")
                details["stoch"] = {"signal": "bullish", "weight": w}
            elif ind["stoch_k"] < 30:
                score += w * 0.3; reasons.append(f"Stochastique bas")
                details["stoch"] = {"signal": "slight_bullish", "weight": w}
            elif ind["stoch_k"] > 80 and ind["stoch_d"] > 80:
                score -= w; reasons.append(f"Stochastique surachat")
                details["stoch"] = {"signal": "bearish", "weight": w}
            elif ind["stoch_k"] > 70:
                score -= w * 0.3; reasons.append(f"Stochastique haut")
                details["stoch"] = {"signal": "slight_bearish", "weight": w}
            else:
                details["stoch"] = {"signal": "neutral", "weight": w}

        # ─── MFI (1 pt) ───
        w = WEIGHTS["mfi"]
        max_score += w
        if ind["mfi"] is not None:
            if ind["mfi"] < 20:
                score += w; reasons.append(f"MFI survente ({ind['mfi']})")
                details["mfi"] = {"signal": "bullish", "weight": w}
            elif ind["mfi"] > 80:
                score -= w; reasons.append(f"MFI surachat ({ind['mfi']})")
                details["mfi"] = {"signal": "bearish", "weight": w}
            else:
                details["mfi"] = {"signal": "neutral", "weight": w}

        # ─── EMA Trend (1.5 pts) ───
        w = WEIGHTS["ema_trend"]
        max_score += w
        ema_score = 0.0
        if ind["ema9"] and ind["ema21"]:
            if ind["ema9"] > ind["ema21"]:
                ema_score += 0.4; reasons.append("EMA9 > EMA21 (court terme haussier)")
            else:
                ema_score -= 0.4; reasons.append("EMA9 < EMA21 (court terme baissier)")
        if ind.get("ema50") and ind["current_price"]:
            if ind["current_price"] > ind["ema50"]:
                ema_score += 0.3; reasons.append("Prix > EMA50")
            else:
                ema_score -= 0.3; reasons.append("Prix < EMA50")
        if ind.get("ema200") and ind["current_price"]:
            if ind["current_price"] > ind["ema200"]:
                ema_score += 0.3; reasons.append("Prix > EMA200 (tendance long terme haussiere)")
            else:
                ema_score -= 0.3; reasons.append("Prix < EMA200 (tendance long terme baissiere)")
        score += ema_score * w / 1.0
        details["ema_trend"] = {"signal": "bullish" if ema_score > 0 else "bearish", "weight": w}

        # ─── Volume (1 pt) ───
        w = WEIGHTS["volume"]
        max_score += w
        if ind["vol_ratio"] is not None:
            if ind["vol_ratio"] > 1.5:
                score += w; reasons.append(f"Volume eleve ({ind['vol_ratio']}x moyenne)")
                details["volume"] = {"signal": "confirmation", "weight": w}
            elif ind["vol_ratio"] < 0.5:
                score -= w * 0.5; reasons.append(f"Volume faible ({ind['vol_ratio']}x moyenne)")
                details["volume"] = {"signal": "weak", "weight": w}
            else:
                details["volume"] = {"signal": "neutral", "weight": w}

        # ─── Heikin Ashi (1 pt) ───
        w = WEIGHTS["heikin_ashi"]
        max_score += w
        if ind["ha_trend"] == 1:
            score += w * 0.5; reasons.append(f"Bougies Heikin Ashi haussiere (x{ind['ha_streak']})")
            details["heikin_ashi"] = {"signal": "bullish", "weight": w}
        elif ind["ha_trend"] == -1:
            score -= w * 0.5; reasons.append(f"Bougies Heikin Ashi baissiere (x{ind['ha_streak']})")
            details["heikin_ashi"] = {"signal": "bearish", "weight": w}

        # ─── Signal final ───
        normalized_score = score / max_score if max_score > 0 else 0  # -1 à +1

        if normalized_score >= 0.4:
            signal = "ACHAT"
            niveau = "FORT" if normalized_score >= 0.6 else "MOYEN"
        elif normalized_score <= -0.4:
            signal = "VENTE"
            niveau = "FORT" if normalized_score <= -0.6 else "MOYEN"
        else:
            signal = "NEUTRE"
            niveau = "FAIBLE"

        self.signals = {
            "signal": signal,
            "niveau": niveau,
            "raw_score": round(score, 2),
            "max_score": round(max_score, 2),
            "normalized_score": round(normalized_score, 4),
            "reasons": reasons[:10],
            "details": details,
        }
        return True

    def build_result(self):
        """Construit le dictionnaire de resultat final"""
        ind = self.indicators
        sig = self.signals

        # Detection de patterns supplementaires
        patterns = []
        if ind.get("rsi") is not None and ind["rsi"] < 30 and sig.get("signal") == "ACHAT":
            patterns.append("Overshoot RS (divergence haussiere potentielle)")
        if ind.get("bb_width") is not None and ind["bb_width"] > 0.3:
            patterns.append("Bollinger Squeeze (volatilite imminente)")
        if ind.get("ha_trend") == 1 and ind.get("ha_streak", 0) >= 5:
            patterns.append(f"Tendance HA confirmee ({ind['ha_streak']} bougies)")

        self.result = {
            "coin": self.coin_id,
            "name": self.coin_id.replace("-", " ").title(),
            "price": ind["current_price"],
            "change_24h": ind.get("change_24h", 0),
            "indicators": {k: v for k, v in ind.items() if k not in ("current_price", "change_24h")},
            "signal": sig.get("signal", "NEUTRE"),
            "signal_niveau": sig.get("niveau", "FAIBLE"),
            "signal_score": sig.get("raw_score", 0),
            "signal_max": sig.get("max_score", 1),
            "normalized_score": sig.get("normalized_score", 0),
            "reasons": sig.get("reasons", []),
            "details": sig.get("details", {}),
            "patterns": patterns,
            "timestamp": datetime.now().isoformat(),
        }
        return self.result


# ─── Analyseur principal ───────────────────────────────────────────

class MarketAnalyzer:
    """Analyseur multi-coins avec reporting"""

    def __init__(self):
        self.results = []

    def analyze_coin(self, coin_id):
        """Analyse complete d'une crypto"""
        display_name = coin_id.replace("-", " ").title()[:25]
        print(f"\n  {display_name}...", end=" ", flush=True)

        try:
            df = fetch_market_data(coin_id, days=60)
            time.sleep(REQUEST_DELAY)

            if df.empty:
                print("pas de donnees")
                return None

            analysis = CoinAnalysis(coin_id, df, df)
            if not analysis.compute_indicators():
                print("calcul impossible")
                return None
            analysis.generate_signals()
            result = analysis.build_result()

            # Affiche resume
            sig = result["signal"]
            if sig == "ACHAT":
                sig_icon = "🟢"
            elif sig == "VENTE":
                sig_icon = "🔴"
            else:
                sig_icon = "⚪"
            price_str = self._fmt_price(result["price"])
            print(f"{sig_icon} {sig} ({result['signal_niveau']}) {price_str}")
            return result

        except Exception as e:
            print(f"ERREUR: {e}")
            return None

    def _fmt_price(self, p):
        if p < 0.001: return f"${p:.8f}"
        if p < 1: return f"${p:.6f}"
        if p < 100: return f"${p:.4f}"
        if p < 10000: return f"${p:.2f}"
        return f"${p:,.0f}"

    def analyze_multiple(self, coins, show_progress=True):
        """Analyse plusieurs coins"""
        self.results = []
        for coin in coins:
            result = self.analyze_coin(coin)
            if result:
                self.results.append(result)
            time.sleep(REQUEST_DELAY * 0.5)
        return self.results

    def get_market_summary(self):
        """Resume du marche"""
        valid = [r for r in self.results if r is not None]
        if not valid or len(valid) == 0:
            return {"total": 0, "achat": 0, "vente": 0, "neutre": 0,
                    "achat_fort": 0, "vente_fort": 0, "best_score": 0, "worst_score": 0}

        buys = [r for r in valid if r["signal"] == "ACHAT"]
        sells = [r for r in valid if r["signal"] == "VENTE"]
        neutrals = [r for r in valid if r["signal"] == "NEUTRE"]

        strong_buys = [r for r in buys if r["signal_niveau"] == "FORT"]
        strong_sells = [r for r in sells if r["signal_niveau"] == "FORT"]

        return {
            "timestamp": datetime.now().isoformat(),
            "total": len(valid),
            "achat": len(buys),
            "vente": len(sells),
            "neutre": len(neutrals),
            "achat_fort": len(strong_buys),
            "vente_fort": len(strong_sells),
            "best_score": max([r.get("normalized_score", -1) for r in valid], default=0),
            "worst_score": min([r.get("normalized_score", 1) for r in valid], default=0),
        }

    def print_report(self):
        """Affiche le rapport complet dans le terminal"""
        valid = [r for r in self.results if r is not None]
        print(f"\n{'='*60}")
        print(f"  HERMES TRADING BOT v2 — Rapport {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        print(f"{'='*60}")

        summary = self.get_market_summary()
        if summary["total"] == 0:
            print(f"\n  ⚠ Aucun actif analyse — verifier les limites API")
            return

        summary = self.get_market_summary()
        print(f"\n  📊 Marche: {summary['total']} actifs | "
              f"🟢 {summary['achat']} ACHAT ({summary['achat_fort']} fort) | "
              f"🔴 {summary['vente']} VENTE ({summary['vente_fort']} fort) | "
              f"⚪ {summary['neutre']} NEUTRE")

        # Tri par score
        sorted_results = sorted(valid, key=lambda r: r.get("normalized_score", 0), reverse=True)

        # Top BUY
        top_buys = [r for r in sorted_results if r["signal"] == "ACHAT"][:5]
        if top_buys:
            print(f"\n  🟢 TOP SIGNAL ACHAT")
            print(f"  {'Coin':<18} {'Prix':<12} {'Score':<8} {'Niveau':<8} {'RSI':<8} {'Ratio Vol':<10}")
            print(f"  {'-'*60}")
            for r in top_buys:
                score = f"{r['normalized_score']:+.2f}"
                rsi = str(r["indicators"].get("rsi", "-"))
                vol = str(r["indicators"].get("vol_ratio", "-"))
                print(f"  {r['name']:<18} {self._fmt_price(r['price']):<12} {score:<8} "
                      f"{r['signal_niveau']:<8} {rsi:<8} {vol:<10}")

        # Top SELL
        top_sells = [r for r in sorted_results if r["signal"] == "VENTE"][:3]
        if top_sells:
            print(f"\n  🔴 TOP SIGNAL VENTE")
            print(f"  {'Coin':<18} {'Prix':<12} {'Score':<8} {'Niveau':<8} {'RSI':<8}")
            print(f"  {'-'*60}")
            for r in top_sells:
                score = f"{r['normalized_score']:+.2f}"
                rsi = str(r["indicators"].get("rsi", "-"))
                print(f"  {r['name']:<18} {self._fmt_price(r['price']):<12} {score:<8} "
                      f"{r['signal_niveau']:<8} {rsi:<8}")

        # Details de chaque coin
        for r in sorted_results:
            sig = r["signal"]
            if sig == "ACHAT": ico = "🟢"
            elif sig == "VENTE": ico = "🔴"
            else: ico = "⚪"
            i = r["indicators"]
            
            print(f"\n  {ico} {r['name']:<20} {self._fmt_price(r['price']):<12} "
                  f"{sig} ({r['signal_niveau']}) score {r['normalized_score']:+.2f}")
            
            details = []
            if i.get("rsi") is not None: details.append(f"RSI {i['rsi']}")
            if i.get("macd_hist") is not None: details.append(f"MACDh {i['macd_hist']:+.2f}")
            if i.get("adx") is not None: details.append(f"ADX {i['adx']}")
            if i.get("bb_percent") is not None: details.append(f"BB% {i['bb_percent']:.2f}")
            if i.get("mfi") is not None: details.append(f"MFI {i['mfi']}")
            if i.get("vol_ratio") is not None: details.append(f"Vol {i['vol_ratio']}x")
            if i.get("atr_pct") is not None: details.append(f"ATR {i['atr_pct']}%")
            if i.get("ha_trend") is not None: details.append(f"HA {'HAUSS' if i['ha_trend']==1 else 'BAISS'}")
            if details:
                print(f"     {' | '.join(details)}")
            
            for reason in r["reasons"][:3]:
                print(f"     → {reason}")
            if r["patterns"]:
                for p in r["patterns"][:2]:
                    print(f"     ★ {p}")

        print(f"\n{'='*60}\n")

    def save_report(self, html=False):
        """Sauvegarde le rapport"""
        valid = [r for r in self.results if r is not None]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # JSON
        json_path = DATA_DIR / f"analyse_{timestamp}.json"
        report = {
            "generated_at": datetime.now().isoformat(),
            "market_summary": self.get_market_summary(),
            "results": valid,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  📁 Rapport JSON: {json_path}")

        # History
        hist_path = DATA_DIR / "history" / f"history_{timestamp}.json"
        # Garde seulement les donnees essentielles
        hist = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_market_summary(),
            "coins": {r["coin"]: {
                "price": r["price"],
                "signal": r["signal"],
                "score": r["normalized_score"],
                "rsi": r["indicators"].get("rsi"),
            } for r in valid},
        }
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)

        if html:
            self._save_html(timestamp)

        return json_path

    def _save_html(self, timestamp):
        """Genere un rapport HTML"""
        valid = [r for r in self.results if r is not None]
        summary = self.get_market_summary()
        sorted_results = sorted(valid, key=lambda r: r.get("normalized_score", 0), reverse=True)

        def color_for_signal(sig):
            return {"ACHAT": "#00c853", "VENTE": "#ff1744", "NEUTRE": "#ffc107"}.get(sig, "#999")

        def score_color(s):
            if s >= 0.3: return "#00c853"
            if s <= -0.3: return "#ff1744"
            return "#ffc107"

        rows = ""
        for r in sorted_results:
            i = r["indicators"]
            reasons = "<br>".join([f"• {re}" for re in r["reasons"][:5]])
            patterns_html = ""
            for p in r["patterns"][:3]:
                patterns_html += f'<div class="pattern">{p}</div>'

            sc = score_color(r["normalized_score"])
            rows += f"""
            <tr>
                <td><strong>{r['name']}</strong></td>
                <td>{self._fmt_price(r['price'])}</td>
                <td><span style="color: {sc};">{r['normalized_score']:+.2f}</span></td>
                <td><span class="signal" style="background: {color_for_signal(r['signal'])};">{r['signal']}</span> <small>{r['signal_niveau']}</small></td>
                <td>{i.get('rsi', '-')}</td>
                <td>{i.get('macd_hist', '-')}</td>
                <td>{i.get('adx', '-')}</td>
                <td>{i.get('vol_ratio', '-')}x</td>
                <td>{i.get('atr_pct', '-')}%</td>
                <td><small>{reasons}</small>{patterns_html}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hermes Trading Bot — Rapport {datetime.now().strftime('%d/%m/%Y %H:%M')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17; color: #e0e0e0; padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00bcd4; font-size: 1.4rem; margin-bottom: 5px; }}
        .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 20px; }}
        .summary {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }}
        .stat-card {{ background: #111827; border-radius: 10px; padding: 15px 20px; flex: 1; min-width: 120px; }}
        .stat-card h3 {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-card .value {{ font-size: 1.5rem; font-weight: 700; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #111827; padding: 10px; text-align: left; font-size: 0.75rem; color: #888; text-transform: uppercase; position: sticky; top: 0; }}
        td {{ padding: 10px; border-bottom: 1px solid #1e293b; font-size: 0.85rem; }}
        tr:hover td {{ background: #111827; }}
        .signal {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #000; font-weight: 700; font-size: 0.75rem; }}
        .pattern {{ display: inline-block; background: #1e3a5f; padding: 2px 6px; border-radius: 3px; font-size: 0.7rem; margin: 1px; }}
        .footer {{ margin-top: 20px; padding: 10px; background: #111827; border-radius: 8px; color: #555; font-size: 0.8rem; text-align: center; }}
        @media (max-width: 600px) {{ .stat-card .value {{ font-size: 1.2rem; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hermes Trading Bot v2</h1>
        <div class="subtitle">Analyse marche — {datetime.now().strftime('%d %B %Y %H:%M UTC')}</div>
        <div class="summary">
            <div class="stat-card"><h3>Actifs</h3><div class="value">{summary['total']}</div></div>
            <div class="stat-card" style="border-left: 3px solid #00c853;"><h3>Achat</h3><div class="value" style="color:#00c853;">{summary['achat']} <small style="color:#666;">({summary['achat_fort']} fort)</small></div></div>
            <div class="stat-card" style="border-left: 3px solid #ff1744;"><h3>Vente</h3><div class="value" style="color:#ff1744;">{summary['vente']} <small style="color:#666;">({summary['vente_fort']} fort)</small></div></div>
            <div class="stat-card" style="border-left: 3px solid #ffc107;"><h3>Neutre</h3><div class="value" style="color:#ffc107;">{summary['neutre']}</div></div>
            <div class="stat-card"><h3>Meilleur score</h3><div class="value" style="color:#00c853;">{summary['best_score']:+.2f}</div></div>
            <div class="stat-card"><h3>Pire score</h3><div class="value" style="color:#ff1744;">{summary['worst_score']:+.2f}</div></div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Coin</th><th>Prix</th><th>Score</th><th>Signal</th><th>RSI</th><th>MACDh</th><th>ADX</th><th>Vol</th><th>ATR</th><th>Raisons</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        <div class="footer">
            Hermes Trading Bot v2 — Donnees CoinGecko — Analyse technique en temps reel
        </div>
    </div>
</body>
</html>"""

        html_path = DATA_DIR / f"analyse_{timestamp}.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"  📁 Rapport HTML: {html_path}")
        return html_path


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Bot v2")
    parser.add_argument("--coin", default="bitcoin,ethereum",
                        help="Coin(s) (virgules, 'all'=top 20, 'top5'=top 5)")
    parser.add_argument("--days", type=int, default=60, help="Jours d'historique")
    parser.add_argument("--save", action="store_true", help="Sauvegarder rapport JSON")
    parser.add_argument("--html", action="store_true", help="Generer rapport HTML")
    parser.add_argument("--loop", type=int, help="Boucle toutes les N minutes")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    args = parser.parse_args()

    # Selection des coins
    if args.coin == "all":
        coins = TOP_50[:20]
    elif args.coin == "top5":
        coins = TOP_50[:5]
    else:
        coins = [c.strip() for c in args.coin.split(",")]

    analyzer = MarketAnalyzer()

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'#'*60}")
        print(f"  # HERMES TRADING BOT v2 — analyse {len(coins)} actifs")
        print(f"  # Periode: {args.days}j | Iteration #{iteration}")
        if args.loop:
            print(f"  # Boucle: toutes les {args.loop} min")
        print(f"{'#'*60}")

        analyzer.analyze_multiple(coins, show_progress=not args.quiet)

        if not args.quiet:
            analyzer.print_report()

        if args.save or args.html:
            analyzer.save_report(html=args.html)

        if not args.loop:
            break

        # Attente avant prochaine iteration
        if iteration == 1:
            wait = args.loop
        else:
            wait = args.loop
        print(f"\n⏳ Prochaine analyse dans {wait} min...")
        time.sleep(wait * 60)


if __name__ == "__main__":
    main()
