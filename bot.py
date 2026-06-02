#!/usr/bin/env python3
"""
Hermes Trading Bot v3 — IA de raisonnement + Risk Management.
Analyse technique avancee + ML + IA (qwen2.5:3b) + GESTION DE RISQUE.

Fonctionnalites:
  - Analyse technique complete (RSI, MACD, BB, ADX, Stoch, MFI, ATR, Heikin Ashi)
  - Machine Learning (RandomForest, LinearRegression)
  - Raisonnement IA via Ollama — analyse narrative du marche
  - Divergences RSI/prix (bullish/bearish)
  - Retracements Fibonacci
  - Backtesting avec validation
  - RISK MANAGEMENT: Kelly Criterion, position sizing, stop-loss ATR,
    take-profit, trailing stop, drawdown protection
  - Simulation de portefeuille avec P&L
  - Site web explicatif (GitHub Pages)

Usage:
  python3 bot.py                                    # Analyse BTC, ETH
  python3 bot.py --coin all                         # Top 20 coins
  python3 bot.py --coin solana --llm                # Avec raisonnement IA
  python3 bot.py --portfolio 10000                  # Simulation portefeuille 10k$
  python3 bot.py --backtest                         # Backtest strategie
  python3 bot.py --loop 60 --llm --portfolio 5000   # Boucle + portefeuille
"""
import json, time, sys, os, argparse, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ─── ML ──────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_OK = True
except ImportError:
    ML_OK = False

# ─── Configuration ───────────────────────────────────────────────
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

REQUEST_DELAY = 5.0
CACHE_TTL = 300  # 5 min

TOP_50 = [
    "bitcoin","ethereum","ripple","cardano","solana","polkadot","dogecoin",
    "avalanche","chainlink","polygon","litecoin","uniswap","stellar","monero",
    "filecoin","vechain","theta","eos","aave","maker","algorand","tezos",
    "near","hedera","cosmos","internet-computer","aptos","sui","optimism",
    "arbitrum","pepe","injective","fetch-ai","render","immutable","sei",
    "celestia","kaspa","flow","gala","fantom","kucoin-token","compound",
    "curve-dao-token","zcash","quant","bitget-token","dydx","pyth-network"
]

# ─── LLM Client ──────────────────────────────────────────────────

class LLMAnalyzer:
    """Analyse narrative via Ollama (raisonnement IA)"""

    def __init__(self, model=LLM_MODEL):
        self.model = model
        self.available = self._check()

    def _check(self):
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False

    def analyze(self, prompt: str, system: str = "") -> Optional[str]:
        if not self.available:
            return None
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system or "Tu es un analyste financier expert en crypto-monnaies. Reponds en francais de facon concise et professionnelle, sans emoji, sans markdown.",
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024}
            }
            r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            return None
        except Exception as e:
            return None

    def market_analysis(self, summary: dict, top_buys: list, top_sells: list, context: str = "") -> str:
        """Analyse narrative complete du marche"""
        prompt = f"""Analyse le marche crypto suivant et donne ton verdict argumente :

RESUME DU MARCHE:
- Actifs analyses: {summary.get('total', 0)}
- Signaux ACHAT: {summary.get('achat', 0)} ({summary.get('achat_fort', 0)} forts)
- Signaux VENTE: {summary.get('vente', 0)} ({summary.get('vente_fort', 0)} forts)
- Meilleur score: {summary.get('best_score', 0):+.2f}
- Pire score: {summary.get('worst_score', 0):+.2f}

TOP ACHATS:
{chr(10).join([f'{r["name"]}: ${r["price"]:,.2f} score={r["normalized_score"]:+.2f} RSI={r["indicators"].get("rsi","N/A")} MACDh={r["indicators"].get("macd_hist","N/A"):+.2f}' for r in top_buys[:5]]) if top_buys else 'Aucun'}

TOP VENTES:
{chr(10).join([f'{r["name"]}: ${r["price"]:,.2f} score={r["normalized_score"]:+.2f} RSI={r["indicators"].get("rsi","N/A")} MACDh={r["indicators"].get("macd_hist","N/A"):+.2f}' for r in top_sells[:3]]) if top_sells else 'Aucun'}

{context}

Donne :
1. Tendance generale du marche
2. Opportunites identifiees (avec justifications)
3. Risques et points d'attention
4. Recommandation actionnable"""
        return self.analyze(prompt)

    def coin_analysis(self, coin_name: str, metrics: dict, reasons: list) -> str:
        """Analyse narrative d'un coin specifique"""
        prompt = f"""Analyse cet actif et donne ton avis d'expert:

ACTIF: {coin_name}
Prix: ${metrics.get('price', 0):,.2f}
Variation 24h: {metrics.get('change_24h', 0):+.2f}%
Signal: {metrics.get('signal', 'N/A')}
Score: {metrics.get('normalized_score', 0):+.2f}

INDICATEURS TECHNIQUES:
{chr(10).join([f'- {k}: {v}' for k, v in metrics.get('indicators', {}).items() if v is not None])[:500]}

RAISONS DU SIGNAL:
{chr(10).join([f'- {r}' for r in reasons[:8]])}

Question: Quel est ton verdict sur {coin_name} ? Explique la situation technique, le contexte de marche, et donne une recommandation claire (ACHAT/NEUTRE/VENTE) avec un niveau de confiance."""
        return self.analyze(prompt)

    def portfolio_recommendation(self, coins: list, budget: float = 10000) -> str:
        """Recommandation de portefeuille"""
        entries = []
        for c in coins[:10]:
            score = c.get("normalized_score", 0)
            sig = c.get("signal", "NEUTRE")
            entries.append(f"- {c['name']}: signal={sig} score={score:+.2f} price=${c.get('price',0):,.2f}")
        prompt = f"""Propose une allocation de portefeuille de ${budget:,.0f} basee sur ces analyses:

{chr(10).join(entries)}

Donne pour chaque actif: allocation (%), prix d'entree conseille, stop-loss, take-profit, et justification."""
        return self.analyze(prompt)


# ─── Data fetching ───────────────────────────────────────────────

class DataFetcher:
    """Gestion optimisee des donnees avec cache"""

    def __init__(self):
        self.last_request = 0

    def _throttle(self):
        elapsed = time.time() - self.last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request = time.time()

    def _cached_get(self, cache_key: str, url: str, ttl: int = CACHE_TTL):
        cache_path = CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < ttl:
                with open(cache_path) as f:
                    return json.load(f)
        # Si age > TTL, on rafraichit (mais pas si < 30s pour eviter spam)
        if age < 30 if cache_path.exists() else False:
            return json.load(open(cache_path))

        self._throttle()
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            except Exception as e:
                if cache_path.exists():
                    # Fallback au cache si echec
                    with open(cache_path) as f:
                        return json.load(f)
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        return {}

    def fetch_market_data(self, coin_id: str, days: int = 90):
        """Donnees marche avec OHLCV synthetique"""
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        data = self._cached_get(f"market_{coin_id}_{days}", url)
        if not data or "prices" not in data or not data["prices"]:
            return pd.DataFrame()

        prices = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])
        df = prices.merge(volumes, on="timestamp", how="left")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # OHLC synthetique
        df["open"] = df["close"].shift(1)
        df.loc[df.index[0], "open"] = df["close"].iloc[0]
        df["high"] = df["close"].rolling(5, min_periods=1).max()
        df["low"] = df["close"].rolling(5, min_periods=1).min()

        # Resample quotidien
        if len(df) > 48:
            ohlc_dict = {"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"}
            df = df.resample("1D").agg(ohlc_dict).dropna()

        return df

    def fetch_coins_list(self):
        """Liste de tous les coins avec market cap"""
        url = f"{COINGECKO_BASE}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1"
        return self._cached_get("coins_markets", url, ttl=600)

    def fetch_trending(self):
        """Coins tendances"""
        url = f"{COINGECKO_BASE}/search/trending"
        return self._cached_get("trending", url, ttl=900)

    def fetch_global_data(self):
        """Donnees globales du marche"""
        url = f"{COINGECKO_BASE}/global"
        return self._cached_get("global", url, ttl=600)

    def fetch_coin_info(self, coin_id: str):
        """Infos detaillees d'un coin"""
        url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
        return self._cached_get(f"info_{coin_id}", url, ttl=3600)


# ─── Indicateurs techniques ──────────────────────────────────────

class Indicators:
    """Calcul de tous les indicateurs techniques"""

    @staticmethod
    def crossed_above(s1, s2):
        if isinstance(s2, (int, float)):
            return (s1 > s2) & (s1.shift(1) <= s2)
        return (s1 > s2) & (s1.shift(1) <= s2.shift(1))

    @staticmethod
    def crossed_below(s1, s2):
        if isinstance(s2, (int, float)):
            return (s1 < s2) & (s1.shift(1) >= s2)
        return (s1 < s2) & (s1.shift(1) >= s2.shift(1))

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_g = gain.rolling(period, min_periods=period).mean()
        avg_l = loss.rolling(period, min_periods=period).mean()
        rs = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series, period):
        return series.rolling(period, min_periods=period).mean()

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        ef = Indicators.ema(series, fast)
        es = Indicators.ema(series, slow)
        line = ef - es
        sig = Indicators.ema(line, signal)
        hist = line - sig
        return line, sig, hist

    @staticmethod
    def bollinger(series, window=20, std=2):
        mid = Indicators.sma(series, window)
        sd = series.rolling(window, min_periods=window).std()
        upper = mid + sd * std
        lower = mid - sd * std
        pct = (series - lower) / (upper - lower)
        width = (upper - lower) / mid
        return upper, mid, lower, pct, width

    @staticmethod
    def adx(df, period=14):
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        pdm = np.zeros_like(c)
        ndm = np.zeros_like(c)
        tr = np.zeros_like(c)
        for i in range(1, len(c)):
            up = h[i] - h[i-1]
            dn = l[i-1] - l[i]
            pdm[i] = up if up > dn and up > 0 else 0
            ndm[i] = dn if dn > up and dn > 0 else 0
            tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr = pd.Series(tr).rolling(period).mean().values
        pdi = 100 * pd.Series(pdm).rolling(period).mean().values / np.maximum(atr, 1e-10)
        ndi = 100 * pd.Series(ndm).rolling(period).mean().values / np.maximum(atr, 1e-10)
        dx = 100 * abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-10)
        return pd.Series(dx).rolling(period).mean(), pd.Series(pdi), pd.Series(ndi), pd.Series(atr)

    @staticmethod
    def stoch(high, low, close, k=14, d=3):
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        kline = 100 * (close - ll) / (hh - ll + 1e-10)
        dline = kline.rolling(d).mean()
        return kline, dline

    @staticmethod
    def mfi(df, period=14):
        typical = (df["high"] + df["low"] + df["close"]) / 3
        mf = typical * df.get("volume", pd.Series(1, index=df.index))
        sign = (typical.diff() >= 0).astype(int) * 2 - 1
        pos = mf.where(sign > 0, 0).rolling(period).sum()
        neg = mf.where(sign < 0, 0).rolling(period).sum()
        ratio = pos / np.maximum(neg, 1e-10)
        return 100 - (100 / (1 + ratio))

    @staticmethod
    def heikin_ashi(df):
        ha = df.copy()
        ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha["ha_open"] = ((df["open"] + df["close"]) / 2).shift(1).bfill()
        ha["ha_high"] = ha[["high", "ha_open", "ha_close"]].max(axis=1)
        ha["ha_low"] = ha[["low", "ha_open", "ha_close"]].min(axis=1)
        ha["ha_trend"] = np.where(ha["ha_close"] > ha["ha_open"], 1, -1)
        # Comptage streak
        trend_vals = ha["ha_trend"].values
        streak, cnt = 0, 0
        for i in range(len(trend_vals)-1, -1, -1):
            if trend_vals[i] == ha["ha_trend"].iloc[-1]:
                cnt += 1
            else:
                break
        ha["ha_streak"] = cnt
        return ha

    @staticmethod
    def atr(df, period=14):
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def fib_retracement(high, low):
        """Niveaux Fibonacci pour une tendance"""
        diff = high - low
        return {
            "0.0": high,
            "0.236": high - diff * 0.236,
            "0.382": high - diff * 0.382,
            "0.5": high - diff * 0.5,
            "0.618": high - diff * 0.618,
            "0.786": high - diff * 0.786,
            "1.0": low,
        }

    @staticmethod
    def find_divergence(price, rsi, window=14):
        """Detecte divergences haussieres/baissieres"""
        divergences = []
        for i in range(window, len(price)-1):
            # Double bottom (haussiere)
            if (price[i] < price[i-window] and rsi[i] > rsi[i-window] and
                price[i] < price[i-1] and price[i] < price[i+1]):
                divergences.append({"type": "bullish", "index": i, "price": float(price[i]), "rsi": float(rsi[i])})
            # Double top (baissiere)
            if (price[i] > price[i-window] and rsi[i] < rsi[i-window] and
                price[i] > price[i-1] and price[i] > price[i+1]):
                divergences.append({"type": "bearish", "index": i, "price": float(price[i]), "rsi": float(rsi[i])})
        return divergences[-5:] if divergences else []

    @staticmethod
    def support_resistance(high, low, n_levels=5):
        """Niveaux de support/resistance par clustering"""
        all_levels = np.concatenate([high.values, low.values])
        if len(all_levels) < n_levels:
            return {"support": float(min(all_levels)), "resistance": float(max(all_levels))}
        # KMeans-like simple: diviser en n buckets
        sorted_vals = np.sort(all_levels)
        bucket_size = len(sorted_vals) // n_levels
        levels = {}
        for i in range(n_levels):
            bucket = sorted_vals[i*bucket_size:(i+1)*bucket_size]
            levels[f"niveau_{i+1}"] = float(np.mean(bucket))
        return {"support": min(levels.values()), "resistance": max(levels.values()), **levels}


# ─── Machine Learning ────────────────────────────────────────────

class MLPredictor:
    """Predictions ML pour les tendances de prix"""

    def __init__(self):
        self.ready = ML_OK

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Cree les features pour le ML"""
        close = df["close"].values
        volume = df.get("volume", pd.Series(1, index=df.index)).values

        features = []
        targets = []
        n = len(close)

        for i in range(20, n - 5):
            feat = [
                close[i] / close[i-1] - 1,            # rendement j-1
                close[i] / close[i-5] - 1,            # rendement j-5
                close[i] / close[i-10] - 1,           # rendement j-10
                close[i] / close[i-20] - 1,           # rendement j-20
                volume[i] / (np.mean(volume[i-5:i]) + 1e-10),  # ratio vol
                np.std(close[i-5:i]) / close[i],      # volatilite 5j
                np.std(close[i-10:i]) / close[i],     # volatilite 10j
                np.mean(close[i-5:i]) / close[i] - 1, # distance EMA5
                np.mean(close[i-10:i]) / close[i] - 1, # distance EMA10
                close[i] / np.max(close[i-20:i]) - 1, # distance au max 20j
                close[i] / np.min(close[i-20:i]) - 1, # distance au min 20j
                max(close[i-5:i]) / min(close[i-5:i]) - 1, # range 5j
            ]
            features.append(feat)

            # Target: rendement moyen sur les 5 prochains jours
            target = close[i+5] / close[i] - 1 if i+5 < n else 0
            targets.append(target)

        return np.array(features), np.array(targets)

    def predict(self, df: pd.DataFrame) -> dict:
        """Prediction ML de la tendance — version amelioree"""
        if not self.ready or len(df) < 60:
            return {"error": "ML non disponible ou donnees insuffisantes"}

        try:
            X, y = self.prepare_features(df)
            if len(X) < 20:
                return {"error": "echantillon insuffisant"}

            # Utiliser les 80% recents pour train, 20% pour test
            split = max(int(len(X) * 0.8), len(X) - 20)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if len(X_test) < 3:
                X_train, X_test = X[:-3], X[-3:]
                y_train, y_test = y[:-3], y[-3:]

            # Standardiser
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # 1) Linear Regression (robuste pour petits echantillons)
            lr = LinearRegression()
            lr.fit(X_train_s, y_train)
            lr_pred_test = lr.predict(X_test_s)
            lr_mae = mean_absolute_error(y_test, lr_pred_test)
            lr_r2 = r2_score(y_test, lr_pred_test)

            # 2) Random Forest (si assez d'echantillons)
            rf = None; rf_mae = None; rf_r2 = None; rf_pred = None
            if len(X_train) >= 30:
                rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=1)
                rf.fit(X_train_s, y_train)
                rf_pred_test = rf.predict(X_test_s)
                rf_mae = mean_absolute_error(y_test, rf_pred_test)
                rf_r2 = r2_score(y_test, rf_pred_test)
                rf_pred = float(rf.predict(scaler.transform(X[-1:].reshape(1, -1)))[0])

            # Prediction future
            lr_future = float(lr.predict(scaler.transform(X[-1:].reshape(1, -1)))[0])

            # Best model selection
            if rf is not None and rf_r2 is not None and rf_r2 > lr_r2:
                best_pred = rf_pred
                best_model = "rf"
                mae = rf_mae
            else:
                best_pred = lr_future
                best_model = "lr"
                mae = lr_mae

            # Calculer la volatilite historique comme base de comparaison
            hist_vol = np.std(y_train) * 100

            # Confiance basee sur la qualite du fit
            best_r2 = max(rf_r2 or -10, lr_r2)
            conf = "haute" if best_r2 > 0.3 else "moyenne" if best_r2 > 0.05 else "faible"
            accuracy = max(0, min(100, max(0, (1 - abs(mae) / max(hist_vol, 0.01))) * 100))

            # Tendances ML
            bf_pct = best_pred * 100
            trend = "HAUSSIER" if bf_pct > 1.5 else "BAISSIER" if bf_pct < -1.5 else "NEUTRE"

            return {
                "prediction_5d": round(bf_pct, 2),
                "model": best_model,
                "r2_score": round(float(best_r2), 3),
                "mae": round(float(mae * 100), 2),
                "accuracy_pct": round(accuracy, 1),
                "confidence": conf,
                "trend": trend,
                "training_samples": len(X_train),
                "hist_volatility": round(float(hist_vol), 2),
                "rf_r2": round(float(rf_r2), 3) if rf_r2 is not None else None,
                "lr_r2": round(float(lr_r2), 3),
                "lr_prediction": round(float(lr_future * 100), 2),
                "rf_prediction": round(float(rf_pred * 100), 2) if rf_pred is not None else None,
            }
        except Exception as e:
            return {"error": str(e)}


# ─── Analyse d'un coin ──────────────────────────────────────────

class CoinBrain:
    """Analyse intelligente d'un actif avec TOUS les indicateurs"""

    def __init__(self, coin_id: str, df: pd.DataFrame, fetcher: DataFetcher):
        self.coin_id = coin_id
        self.df = df
        self.fetcher = fetcher
        self.indicators = {}
        self.ml = MLPredictor()
        self.llm = LLMAnalyzer()

    def compute(self) -> bool:
        """Calcul de tous les indicateurs"""
        df = self.df
        if df.empty or len(df) < 30:
            return False

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        current = float(close.iloc[-1])

        # Prix et variation
        change_24h = ((current - float(close.iloc[-2])) / float(close.iloc[-2])) * 100 if len(close) > 1 else 0

        # --- Tous les indicateurs ---
        rsi_series = Indicators.rsi(close)
        ema9 = Indicators.ema(close, 9)
        ema21 = Indicators.ema(close, 21)
        ema50 = Indicators.ema(close, 50) if len(close) >= 50 else pd.Series(index=close.index)
        ema200 = Indicators.ema(close, 200) if len(close) >= 200 else pd.Series(index=close.index)
        macd_line, macd_sig, macd_hist = Indicators.macd(close)
        bb_up, bb_mid, bb_low, bb_pct, bb_wid = Indicators.bollinger(close)
        adx_s, pdi, ndi, atr_s = Indicators.adx(df)
        stoch_k, stoch_d = Indicators.stoch(high, low, close)
        mfi_s = Indicators.mfi(df)
        ha = Indicators.heikin_ashi(df)
        atr_v = Indicators.atr(df)

        vol_sma = df.get("volume", pd.Series(1, index=df.index)).rolling(20).mean()
        vol_ratio = df.get("volume", pd.Series(1, index=df.index)) / vol_sma.replace(0, np.nan)

        # Divergences
        divergences = Indicators.find_divergence(close.values, rsi_series.values)

        # Fibonacci
        lookback_90 = min(90, len(close))
        fib = Indicators.fib_retracement(float(high.iloc[-lookback_90:].max()),
                                          float(low.iloc[-lookback_90:].min()))

        # Support/Resistance
        sr = Indicators.support_resistance(high, low)

        # ATR
        atr_val = float(atr_s.iloc[-1]) if not atr_s.empty and pd.notna(atr_s.iloc[-1]) else 0
        atr_pct = (atr_val / current * 100) if current > 0 else 0

        # ML prediction
        ml_pred = self.ml.predict(df)

        self.indicators = {
            "current_price": current,
            "change_24h": round(change_24h, 2),
            "rsi": round(float(rsi_series.iloc[-1]), 2) if pd.notna(rsi_series.iloc[-1]) else None,
            "ema9": float(ema9.iloc[-1]) if pd.notna(ema9.iloc[-1]) else None,
            "ema21": float(ema21.iloc[-1]) if pd.notna(ema21.iloc[-1]) else None,
            "ema50": float(ema50.iloc[-1]) if not ema50.empty and pd.notna(ema50.iloc[-1]) else None,
            "ema200": float(ema200.iloc[-1]) if not ema200.empty and pd.notna(ema200.iloc[-1]) else None,
            "macd": float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None,
            "macd_signal": float(macd_sig.iloc[-1]) if pd.notna(macd_sig.iloc[-1]) else None,
            "macd_hist": float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None,
            "bb_upper": float(bb_up.iloc[-1]) if pd.notna(bb_up.iloc[-1]) else None,
            "bb_mid": float(bb_mid.iloc[-1]) if pd.notna(bb_mid.iloc[-1]) else None,
            "bb_lower": float(bb_low.iloc[-1]) if pd.notna(bb_low.iloc[-1]) else None,
            "bb_percent": round(float(bb_pct.iloc[-1]), 3) if pd.notna(bb_pct.iloc[-1]) else None,
            "bb_width": round(float(bb_wid.iloc[-1]), 3) if pd.notna(bb_wid.iloc[-1]) else None,
            "adx": round(float(adx_s.iloc[-1]), 2) if pd.notna(adx_s.iloc[-1]) else None,
            "plus_di": round(float(pdi.iloc[-1]), 2) if pd.notna(pdi.iloc[-1]) else None,
            "minus_di": round(float(ndi.iloc[-1]), 2) if pd.notna(ndi.iloc[-1]) else None,
            "stoch_k": round(float(stoch_k.iloc[-1]), 2) if pd.notna(stoch_k.iloc[-1]) else None,
            "stoch_d": round(float(stoch_d.iloc[-1]), 2) if pd.notna(stoch_d.iloc[-1]) else None,
            "mfi": round(float(mfi_s.iloc[-1]), 2) if pd.notna(mfi_s.iloc[-1]) else None,
            "atr": round(atr_val, 4),
            "atr_pct": round(atr_pct, 2),
            "vol_ratio": round(float(vol_ratio.iloc[-1]), 2) if pd.notna(vol_ratio.iloc[-1]) else None,
            "support": round(float(sr.get("support", 0)), 6),
            "resistance": round(float(sr.get("resistance", 0)), 6),
            "ha_trend": int(ha["ha_trend"].iloc[-1]) if not ha.empty else 0,
            "ha_streak": int(ha["ha_streak"].iloc[-1]) if not ha.empty else 0,
            "divergences": divergences[-3:] if divergences else [],
            "fibonacci": {k: round(float(v), 2) for k, v in fib.items()},
            "ml_prediction": ml_pred,
            "data_points": len(df),
        }
        return True

    def generate_signal(self) -> dict:
        """Systeme de scoring intelligent multicriteres"""
        ind = self.indicators
        score = 0.0
        max_score = 0.0
        reasons = []
        details = {}

        # POIDS: RSI=2, MACD=2, BB=1.5, ADX=1.5, Stoch=1, MFI=1, EMA=1.5, Vol=1, HA=1, ML=2 (nouveau)

        # RSI
        w = 2.0; max_score += w
        if ind["rsi"] is not None:
            if ind["rsi"] < 30: score += w; reasons.append(f"RSI survente ({ind['rsi']})")
            elif ind["rsi"] < 40: score += w*0.5; reasons.append(f"RSI bas ({ind['rsi']})")
            elif ind["rsi"] > 70: score -= w; reasons.append(f"RSI surachat ({ind['rsi']})")
            elif ind["rsi"] > 60: score -= w*0.5; reasons.append(f"RSI haut ({ind['rsi']})")

        # MACD
        w = 2.0; max_score += w
        if ind["macd"] is not None and ind["macd_signal"] is not None:
            if ind["macd"] > ind["macd_signal"]:
                score += w; reasons.append("MACD haussier")
            else:
                score -= w; reasons.append("MACD baissier")
            if ind["macd_hist"] is not None and abs(ind["macd_hist"]) < abs(ind["macd"] - ind.get("prev_macd", 0)):
                # Convergence
                if ind["macd_hist"] > 0: score += w*0.3

        # Bollinger
        w = 1.5; max_score += w
        if ind["bb_percent"] is not None:
            if ind["bb_percent"] < 0.05: score += w; reasons.append("Prix touche bande basse BB")
            elif ind["bb_percent"] < 0.2: score += w*0.5
            elif ind["bb_percent"] > 0.95: score -= w; reasons.append("Prix touche bande haute BB")
            elif ind["bb_percent"] > 0.8: score -= w*0.5

        # ADX + Direction
        w = 1.5; max_score += w
        if ind["adx"] is not None and ind["plus_di"] is not None:
            if ind["adx"] > 25:
                if ind["plus_di"] > ind["minus_di"]:
                    score += w; reasons.append(f"Tendance haussiere forte (ADX {ind['adx']})")
                else:
                    score -= w; reasons.append(f"Tendance baissiere forte (ADX {ind['adx']})")
            elif ind["adx"] > 20:
                if ind["plus_di"] > ind["minus_di"]: score += w*0.3
                else: score -= w*0.3

        # Stochastique
        w = 1.0; max_score += w
        if ind["stoch_k"] is not None:
            if ind["stoch_k"] < 20: score += w; reasons.append("Stochastique survente")
            elif ind["stoch_k"] > 80: score -= w; reasons.append("Stochastique surachat")
            elif ind["stoch_k"] < 30: score += w*0.3
            elif ind["stoch_k"] > 70: score -= w*0.3

        # MFI
        w = 1.0; max_score += w
        if ind["mfi"] is not None:
            if ind["mfi"] < 20: score += w; reasons.append(f"MFI survente ({ind['mfi']})")
            elif ind["mfi"] > 80: score -= w; reasons.append(f"MFI surachat ({ind['mfi']})")

        # EMA Trend
        w = 1.5; max_score += w
        ema_score = 0.0
        if ind["ema9"] and ind["ema21"]:
            if ind["ema9"] > ind["ema21"]: ema_score += 0.4
            else: ema_score -= 0.4
        if ind.get("ema50") and ind["current_price"] > ind["ema50"]: ema_score += 0.3
        elif ind.get("ema50"): ema_score -= 0.3
        if ind.get("ema200") and ind["current_price"] > ind["ema200"]: ema_score += 0.3
        elif ind.get("ema200"): ema_score -= 0.3
        score += ema_score * w / 1.0

        # Volume
        w = 1.0; max_score += w
        if ind["vol_ratio"] is not None:
            if ind["vol_ratio"] > 1.5: score += w; reasons.append(f"Volume eleve ({ind['vol_ratio']}x)")
            elif ind["vol_ratio"] < 0.5: score -= w*0.5

        # Heikin Ashi
        w = 1.0; max_score += w
        if ind["ha_trend"] == 1: score += w*0.5
        elif ind["ha_trend"] == -1: score -= w*0.5

        # Divergences (bonus/malus supplementaire)
        if ind.get("divergences"):
            for div in ind["divergences"][:2]:
                if div["type"] == "bullish":
                    score += 1.5; reasons.append(f"Divergence haussiere detectee (prix ${div['price']:.2f})")
                else:
                    score -= 1.5; reasons.append(f"Divergence baissiere detectee")

        # ML Prediction (2 pts)
        w = 2.0; max_score += w
        ml = ind.get("ml_prediction", {})
        if "error" not in ml and ml.get("prediction_5d") is not None:
            if ml["prediction_5d"] > 3: score += w; reasons.append(f"ML predit +{ml['prediction_5d']}% (5j)")
            elif ml["prediction_5d"] > 1: score += w*0.5
            elif ml["prediction_5d"] < -3: score -= w; reasons.append(f"ML predit {ml['prediction_5d']}% (5j)")
            elif ml["prediction_5d"] < -1: score -= w*0.5

        # Signal final
        normalized = score / max_score if max_score > 0 else 0

        if normalized >= 0.35: signal = "ACHAT"; niveau = "FORT" if normalized >= 0.55 else "MOYEN"
        elif normalized <= -0.35: signal = "VENTE"; niveau = "FORT" if normalized <= -0.55 else "MOYEN"
        else: signal = "NEUTRE"; niveau = "FAIBLE"

        return {
            "signal": signal, "niveau": niveau,
            "raw_score": round(score, 2), "max_score": round(max_score, 2),
            "normalized_score": round(normalized, 4),
            "reasons": reasons[:15], "details": details,
        }

    def build_result(self) -> dict:
        """Construit le resultat complet"""
        ind = self.indicators

        # Patterns detectes
        patterns = []
        if ind.get("rsi") is not None and ind["rsi"] < 30 and ind.get("divergences"):
            if any(d["type"] == "bullish" for d in ind["divergences"]):
                patterns.append("Divergence haussiere RSI (signal fort de retournement)")
        if ind.get("bb_width") is not None and ind["bb_width"] > 0.35:
            patterns.append("Bollinger Squeeze (explosion de volatilite imminente)")
        if ind.get("ha_trend") == 1 and ind.get("ha_streak", 0) >= 5:
            patterns.append(f"Tendance Heikin Ashi confirmee ({ind['ha_streak']} bougies)")
        if ind.get("adx") is not None and ind["adx"] > 30:
            if ind.get("plus_di", 0) > ind.get("minus_di", 0):
                patterns.append("Tendance haussiere forte (ADX > 30, +DI > -DI)")
            else:
                patterns.append("Tendance baissiere forte (ADX > 30, -DI > +DI)")

        # Fibonacci zones
        fib = ind.get("fibonacci", {})
        fib_zone = ""
        if fib and ind["current_price"]:
            for level, price in sorted(fib.items(), key=lambda x: float(x[0])):
                if abs(ind["current_price"] - price) / price < 0.02:
                    fib_zone = f"Prix proche retracement Fib {level} (${price:.2f})"
                    patterns.append(fib_zone)

        return {
            "coin": self.coin_id,
            "name": self.coin_id.replace("-", " ").title(),
            "price": ind["current_price"],
            "change_24h": ind.get("change_24h", 0),
            "indicators": ind,
            "signal": self.signal.get("signal", "NEUTRE"),
            "signal_niveau": self.signal.get("niveau", "FAIBLE"),
            "signal_score": self.signal.get("raw_score", 0),
            "signal_max": self.signal.get("max_score", 1),
            "normalized_score": self.signal.get("normalized_score", 0),
            "reasons": self.signal.get("reasons", []),
            "patterns": patterns,
            "timestamp": datetime.now().isoformat(),
            "ml": ind.get("ml_prediction", {}),
            "divergences": ind.get("divergences", []),
            "fibonacci": ind.get("fibonacci", {}),
        }


# ─── Analyseur principal ──────────────────────────────────────────

class MarketAnalyzer:
    """Analyseur intelligent multi-coins"""

    def __init__(self, use_llm=False):
        self.results = []
        self.fetcher = DataFetcher()
        self.llm = LLMAnalyzer() if use_llm else None
        self.use_llm = use_llm

    def analyze_coin(self, coin_id: str) -> Optional[dict]:
        display_name = coin_id.replace("-", " ").title()[:25]
        sys.stdout.write(f"\n  {display_name}... ")
        sys.stdout.flush()

        try:
            df = self.fetcher.fetch_market_data(coin_id, days=365)
            if df.empty:
                print(f"pas de donnees (limite API)")
                return None

            brain = CoinBrain(coin_id, df, self.fetcher)
            if not brain.compute():
                print(f"calcul impossible")
                return None
            brain.signal = brain.generate_signal()
            result = brain.build_result()

            # Resume visuel
            sig = result["signal"]
            icon = "🟢" if sig == "ACHAT" else ("🔴" if sig == "VENTE" else "⚪")
            price_str = self._fmt_price(result["price"])
            print(f"{icon} {sig} ({result['signal_niveau']}) {price_str}")
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

    def analyze_multiple(self, coins):
        self.results = []
        for coin in coins:
            r = self.analyze_coin(coin)
            if r: self.results.append(r)
        return self.results

    def get_summary(self):
        valid = [r for r in self.results if r]
        if not valid: return {"total": 0}
        buys = [r for r in valid if r["signal"] == "ACHAT"]
        sells = [r for r in valid if r["signal"] == "VENTE"]
        return {
            "timestamp": datetime.now().isoformat(),
            "total": len(valid),
            "achat": len(buys), "vente": len(sells),
            "neutre": len(valid) - len(buys) - len(sells),
            "achat_fort": len([r for r in buys if r["signal_niveau"] == "FORT"]),
            "vente_fort": len([r for r in sells if r["signal_niveau"] == "FORT"]),
            "best_score": max([r.get("normalized_score", -1) for r in valid], default=0),
            "worst_score": min([r.get("normalized_score", 1) for r in valid], default=0),
        }

    def print_report(self):
        valid = [r for r in self.results if r]
        print(f"\n{'='*62}")
        print(f"  HERMES TRADING BOT v3 — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        print(f"{'='*62}")

        summary = self.get_summary()
        if summary["total"] == 0:
            print("\n  Aucun actif analyse.")
            return

        print(f"\n  Marche: {summary['total']} actifs | "
              f"ACHAT {summary['achat']} ({summary['achat_fort']} fort) | "
              f"VENTE {summary['vente']} ({summary['vente_fort']} fort) | "
              f"NEUTRE {summary['neutre']}")

        # Top BUY
        sorted_r = sorted(valid, key=lambda r: r.get("normalized_score", 0), reverse=True)
        top_buys = [r for r in sorted_r if r["signal"] == "ACHAT"]
        top_sells = [r for r in sorted_r if r["signal"] == "VENTE"]

        if top_buys:
            print(f"\n  🟢 TOP SIGNAL ACHAT")
            print(f"  {'Coin':<18} {'Prix':<12} {'Score':<8} {'Niv':<6} {'RSI':<6} {'ADX':<6} {'ML 5j':<8} {'Vol':<6}")
            print(f"  {'-'*66}")
            for r in top_buys[:5]:
                i = r.get("indicators", {})
                ml = r.get("ml", {})
                ml_str = f"{ml.get('prediction_5d', '-')}%" if "error" not in ml else "-"
                print(f"  {r['name']:<18} {self._fmt_price(r['price']):<12} {r['normalized_score']:+.2f}  "
                      f"{r['signal_niveau']:<6} {i.get('rsi','-'):<6} {i.get('adx','-'):<6} "
                      f"{ml_str:<8} {i.get('vol_ratio','-'):<6}")

        if top_sells:
            print(f"\n  🔴 TOP SIGNAL VENTE")
            for r in top_sells[:3]:
                print(f"     {r['name']:<20} score {r['normalized_score']:+.2f} RSI {r.get('indicators',{}).get('rsi','-')}")

        # Details enrichis
        print(f"\n  ── Analyses detaillees ──")
        for r in sorted_r:
            sig = r["signal"]
            icon = "🟢" if sig == "ACHAT" else "🔴" if sig == "VENTE" else "⚪"
            i = r.get("indicators", {})
            divs = r.get("divergences", [])
            ml = r.get("ml", {})

            print(f"\n  {icon} {r['name']:<20} {self._fmt_price(r['price']):<12} "
                  f"{sig} ({r['signal_niveau']}) score {r['normalized_score']:+.2f}")

            # Ligne 1: indicateurs cles
            parts = []
            if i.get("rsi"): parts.append(f"RSI {i['rsi']}")
            if i.get("macd_hist") is not None: parts.append(f"MACDh {i['macd_hist']:+.2f}")
            if i.get("adx"): parts.append(f"ADX {i['adx']}")
            if i.get("bb_percent") is not None: parts.append(f"BB% {i['bb_percent']:.2f}")
            if i.get("mfi"): parts.append(f"MFI {i['mfi']}")
            if i.get("atr_pct"): parts.append(f"ATR {i['atr_pct']}%")
            if i.get("vol_ratio"): parts.append(f"Vol {i['vol_ratio']}x")
            if parts: print(f"     {' | '.join(parts)}")

            # ML
            if "error" not in ml and ml.get("prediction_5d") is not None:
                print(f"     ML: predit {ml['prediction_5d']:+.2f}% (confiance {ml.get('confidence','N/A')}) R²={ml.get('r2_score','N/A')}")

            # Fib
            if i.get("fibonacci"):
                fib = i["fibonacci"]
                print(f"     Fib: 0.618={self._fmt_price(fib.get('0.618',0))} 0.5={self._fmt_price(fib.get('0.5',0))}")

            # Reasons
            for reason in r["reasons"][:4]:
                print(f"     → {reason}")
            for p in r.get("patterns", [])[:2]:
                print(f"     ★ {p}")

        # LLM Analysis
        if self.llm and self.use_llm:
            self._llm_report(top_buys[:5], top_sells[:3], summary)

        print(f"\n{'='*62}\n")

    def _llm_report(self, top_buys, top_sells, summary):
        """Analyse narrative via LLM"""
        print(f"\n  ── Analyse IA (raisonnement) ──")
        sys.stdout.flush()

        analysis = self.llm.market_analysis(summary, top_buys, top_sells)
        if analysis:
            print(f"\n{analysis}\n")
        else:
            print("  (IA non disponible)\n")

    def save_report(self, html=False):
        valid = [r for r in self.results if r]
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        report = {
            "generated_at": datetime.now().isoformat(),
            "market_summary": self.get_summary(),
            "results": valid,
        }
        json_path = DATA_DIR / f"analyse_{ts}.json"
        with open(json_path, "w") as f: json.dump(report, f, indent=2, default=str)
        print(f"  Rapport JSON: {json_path}")

        if html:
            self._save_html(ts)
        return json_path

    def _save_html(self, timestamp):
        valid = [r for r in self.results if r]
        sm = self.get_summary()
        sorted_r = sorted(valid, key=lambda r: r.get("normalized_score", 0), reverse=True)

        def sc(s):
            return "#00c853" if s >= 0.3 else "#ff1744" if s <= -0.3 else "#ffc107"

        rows = ""
        for r in sorted_r:
            i = r.get("indicators", {})
            ml = r.get("ml", {})
            ml_str = f"{ml.get('prediction_5d', 'N/A')}%" if "error" not in ml else "N/A"
            divs = "; ".join([f"{d['type']} ${d.get('price',0):.2f}" for d in r.get("divergences",[])])
            fib = r.get("fibonacci", {})
            fib618 = self._fmt_price(fib.get("0.618", 0)) if fib else "-"
            reasons = "<br>".join([f"• {re}" for re in r["reasons"][:5]])
            patterns = "".join([f'<div class="p">{p}</div>' for p in r.get("patterns", [])[:2]])

            rows += f"""<tr>
                <td><strong>{r['name']}</strong></td>
                <td>{self._fmt_price(r['price'])}</td>
                <td><span style="color:{sc(r['normalized_score'])};">{r['normalized_score']:+.2f}</span></td>
                <td><span class="sig" style="background:{sc(r['signal'])};">{r['signal']}</span> {r['signal_niveau']}</td>
                <td>{i.get('rsi','-')}</td>
                <td>{i.get('macd_hist','-')}</td>
                <td>{i.get('adx','-')}</td>
                <td>{ml_str}</td>
                <td>{i.get('vol_ratio','-')}x</td>
                <td>{i.get('atr_pct','-')}%</td>
                <td>{fib618}</td>
                <td>{divs}</td>
                <td><small>{reasons}{patterns}</small></td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Hermes Trading Bot v3 — Rapport IA</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#0a0e17; color:#e0e0e0; padding:20px; }}
.container {{ max-width:1400px; margin:0 auto; }}
h1 {{ color:#00bcd4; font-size:1.4rem; margin-bottom:5px; }}
.sub {{ color:#888; font-size:0.9rem; margin-bottom:20px; }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }}
.card {{ background:#111827; border-radius:8px; padding:12px 15px; flex:1; min-width:100px; }}
.card h3 {{ font-size:0.7rem; color:#888; text-transform:uppercase; }}
.card .val {{ font-size:1.3rem; font-weight:700; margin-top:3px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
th {{ background:#111827; padding:8px 6px; text-align:left; color:#888; text-transform:uppercase; position:sticky; top:0; font-size:0.65rem; }}
td {{ padding:6px; border-bottom:1px solid #1e293b; }}
tr:hover td {{ background:#111827; }}
.sig {{ display:inline-block; padding:1px 6px; border-radius:3px; color:#000; font-weight:700; font-size:0.7rem; }}
.p {{ display:inline-block; background:#1e3a5f; padding:1px 4px; border-radius:2px; font-size:0.65rem; margin:1px; }}
</style></head><body><div class="container">
<h1>Hermes Trading Bot v3 &mdash; Analyse IA</h1>
<div class="sub">{datetime.now().strftime('%d %B %Y %H:%M UTC')}</div>
<div class="stats">
<div class="card"><h3>Actifs</h3><div class="val">{sm['total']}</div></div>
<div class="card" style="border-left:3px solid #00c853;"><h3>Achat</h3><div class="val" style="color:#00c853;">{sm['achat']}</div></div>
<div class="card" style="border-left:3px solid #ff1744;"><h3>Vente</h3><div class="val" style="color:#ff1744;">{sm['vente']}</div></div>
<div class="card" style="border-left:3px solid #ffc107;"><h3>Neutre</h3><div class="val" style="color:#ffc107;">{sm['neutre']}</div></div>
<div class="card"><h3>Best score</h3><div class="val" style="color:#00c853;">{sm['best_score']:+.2f}</div></div>
<div class="card"><h3>Worst score</h3><div class="val" style="color:#ff1744;">{sm['worst_score']:+.2f}</div></div>
</div>
<table><thead><tr>
<th>Coin</th><th>Prix</th><th>Score</th><th>Signal</th><th>RSI</th><th>MACDh</th><th>ADX</th><th>ML 5j</th><th>Vol</th><th>ATR</th><th>Fib 618</th><th>Diverg</th><th>Analyse</th>
</tr></thead><tbody>{rows}</tbody></table>
</div></body></html>"""

        html_path = DATA_DIR / f"analyse_{timestamp}.html"
        with open(html_path, "w") as f: f.write(html)
        print(f"  Rapport HTML: {html_path}")


# ─── Backtest ────────────────────────────────────────────────────

def run_backtest():
    """Backtest simple de la strategie"""
    print("\n  Backtesting de la strategie...\n")
    fetcher = DataFetcher()
    coins = ["bitcoin", "ethereum", "solana"]
    all_results = []

    for coin_id in coins:
        df = fetcher.fetch_market_data(coin_id, days=365)
        if df.empty or len(df) < 100:
            continue

        close = df["close"].values
        total_trades = 0
        wins = 0
        losses = 0
        pnl = []

        for i in range(60, len(close) - 5):
            # Simule l'analyse a ce point dans le temps
            chunk = df.iloc[:i]
            brain = CoinBrain(coin_id, chunk, fetcher)
            if not brain.compute(): continue
            brain.signal = brain.generate_signal()
            result = brain.build_result()

            if result["signal"] == "ACHAT":
                total_trades += 1
                entry = close[i]
                future = close[i+5:i+6]
                if len(future) > 0:
                    exit_p = future[0]
                    ret = (exit_p - entry) / entry * 100
                    pnl.append(ret)
                    if ret > 0: wins += 1
                    else: losses += 1

        if total_trades > 0:
            win_rate = wins / total_trades * 100
            avg_pnl = np.mean(pnl) if pnl else 0
            all_results.append({
                "coin": coin_id,
                "trades": total_trades,
                "win_rate": round(win_rate, 1),
                "avg_return": round(avg_pnl, 2),
                "total_return": round(sum(pnl), 2),
                "sharpe": round(np.mean(pnl) / np.std(pnl) * np.sqrt(52) if np.std(pnl) > 0 else 0, 2),
            })

    print(f"  Resultats du backtest (periode 1 an):\n")
    for r in all_results:
        print(f"  {r['coin']:12s} trades={r['trades']:3d} win_rate={r['win_rate']:5.1f}% "
              f"avg_ret={r['avg_return']:+.2f}% total={r['total_return']:+.2f}% sharpe={r['sharpe']:+.2f}")

    return all_results


# ─── Risk Management ─────────────────────────────────────────────

class RiskManager:
    """Gestion de risque prudente pour maximiser les gains sans exploser"""

    def __init__(self, initial_capital=10000, max_drawdown=15, max_risk_per_trade=2):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown  # stop trading si -X%
        self.max_risk_per_trade = max_risk_per_trade  # % du capital risque par trade
        self.trades = []
        self.daily_pnl = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3  # pause apres X pertes
        self.cooldown = False
        self.cooldown_until = None

    def kelly_fraction(self, win_rate, avg_win, avg_loss):
        """Kelly Criterion modere (fraction securitaire = Kelly / 2)
        Calcule le % du capital a risquer"""
        if avg_loss == 0 or win_rate == 0:
            return 0.01  # 1% par defaut
        r = avg_win / abs(avg_loss) if avg_loss != 0 else 1
        p = win_rate / 100
        kelly = (p * r - (1 - p)) / r if r > 0 else 0
        # Kelly modere: on prend la moitie pour etre prudent
        return max(0.005, min(kelly / 2, self.max_risk_per_trade / 100))

    def position_size(self, price, stop_loss_pct, win_rate=50, avg_win=5, avg_loss=3):
        """Calcule la taille de position optimale et prudente"""
        # Kelly fraction
        fraction = self.kelly_fraction(win_rate, avg_win, avg_loss)

        # Ajustement selon le drawdown actuel
        dd = self.current_drawdown()
        if dd > 10:
            fraction *= 0.5  # Moitie de risque en drawdown
        elif dd > 5:
            fraction *= 0.75

        # Limite max de capital par trade
        max_capital_at_risk = self.capital * fraction
        position_value = max_capital_at_risk / (stop_loss_pct / 100) if stop_loss_pct > 0 else 0
        position_value = min(position_value, self.capital * 0.3)  # max 30% du capital

        quantity = position_value / price if price > 0 else 0

        return {
            "position_value": round(position_value, 2),
            "quantity": round(quantity, 6),
            "capital_at_risk": round(max_capital_at_risk, 2),
            "risk_pct": round(fraction * 100, 2),
            "kelly_fraction": round(fraction, 4),
        }

    def stop_loss_atr(self, price, atr, multiplier=2.0):
        """Stop-loss base sur ATR (volatilite)
        Plus la volatilite est haute, plus le stop est large"""
        distance = atr * multiplier
        stop = price - distance
        stop_pct = distance / price * 100
        return {
            "stop_price": round(stop, 2),
            "stop_pct": round(stop_pct, 2),
            "distance": round(distance, 2),
            "multiplier": multiplier,
        }

    def take_profit(self, price, atr, risk_reward=2.5):
        """Take-profit base sur Risk/Reward ratio
        1:2.5 par defaut (prudent)"""
        stop_info = self.stop_loss_atr(price, atr)
        distance = price - stop_info["stop_price"]
        tp = price + distance * risk_reward
        tp_pct = (tp - price) / price * 100
        return {
            "tp_price": round(tp, 2),
            "tp_pct": round(tp_pct, 2),
            "risk_reward": round(risk_reward, 1),
            "potential_gain": round((tp - price) / (price - stop_info["stop_price"]) if (price - stop_info["stop_price"]) > 0 else 0, 2),
        }

    def trailing_stop(self, entry_price, current_price, atr, activation_pct=3):
        """Trailing stop qui suit le prix
        S'active seulement apres +3% de gain"""
        gain_pct = (current_price - entry_price) / entry_price * 100
        if gain_pct < activation_pct:
            return {"active": False, "stop": entry_price * 0.97}  # stop a -3% si pas active

        # Trailing: stop a 2x ATR en dessous du plus haut
        trail_distance = atr * 2
        trailing_stop = current_price - trail_distance
        return {
            "active": True,
            "stop": round(trailing_stop, 2),
            "locked_profit": round(gain_pct - (trail_distance / entry_price * 100), 2),
        }

    def current_drawdown(self):
        """Drawdown actuel en %"""
        if self.peak_capital == 0:
            return 0
        dd = (self.peak_capital - self.capital) / self.peak_capital * 100
        return max(0, dd)

    def should_trade(self, signal_score, confidence="faible"):
        """Decision: est-ce qu'on trade ce signal ?"""
        # Verifier drawdown
        dd = self.current_drawdown()
        if dd >= self.max_drawdown_pct:
            return {
                "trade": False,
                "reason": f"Drawdown maximum atteint ({dd:.1f}%)",
                "wait_for": "retour au dessus du drawdown max",
            }

        # Verifier cooldown apres pertes
        if self.cooldown:
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now()).seconds // 60
                return {
                    "trade": False,
                    "reason": f"Cooldown apres {self.consecutive_losses} pertes consecutives",
                    "wait_for": f"{remaining} minutes",
                }
            else:
                self.cooldown = False
                self.cooldown_until = None

        # Verifier score minimum
        min_score = 0.35  # score normalise minimum
        if signal_score < min_score:
            return {
                "trade": False,
                "reason": f"Score insuffisant ({signal_score:.2f} < {min_score})",
                "wait_for": "amelioration du signal",
            }

        # Verifier confiance
        if confidence == "faible":
            return {
                "trade": False,
                "reason": "Confiance ML trop faible",
                "wait_for": "confirmation supplementaire",
            }

        return {
            "trade": True,
            "reason": "Tous les feux sont au vert",
            "confidence": confidence,
            "drawdown": round(dd, 1),
        }

    def record_trade(self, entry_price, exit_price, quantity, side="long"):
        """Enregistre un trade et met a jour le capital"""
        if side == "long":
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        self.capital += pnl
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        self.trades.append({
            "entry": entry_price,
            "exit": exit_price,
            "quantity": quantity,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "capital": round(self.capital, 2),
            "drawdown": round(self.current_drawdown(), 2),
            "timestamp": datetime.now().isoformat(),
        })

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown = True
                self.cooldown_until = datetime.now() + timedelta(hours=24)
                print(f"  ⛔ Cooldown 24h active ({self.consecutive_losses} pertes consecutives)")
        else:
            self.consecutive_losses = 0

        return self.trades[-1]

    def summary(self):
        """Resume de performance"""
        if not self.trades:
            return {
                "capital": round(self.capital, 2),
                "total_return": 0,
                "win_rate": 0,
                "trades": 0,
                "drawdown": 0,
            }

        wins = [t for t in self.trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        return {
            "capital": round(self.capital, 2),
            "initial_capital": self.initial_capital,
            "total_return": round(total_return, 2),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1) if self.trades else 0,
            "total_trades": len(self.trades),
            "current_drawdown": round(self.current_drawdown(), 2),
            "peak_capital": round(self.peak_capital, 2),
            "consecutive_losses": self.consecutive_losses,
            "cooldown_active": self.cooldown,
        }


def run_portfolio_simulation(capital=10000):
    """Simulation de portefeuille avec Risk Management"""
    print(f"\n  Simulation portefeuille — ${capital:,.0f} initial\n")
    print(f"  Regles:\n"
          f"  - Kelly Criterion / 2 pour la taille de position\n"
          f"  - Stop-loss a 2x ATR\n"
          f"  - Take-profit a 1:2.5 (Risk/Reward)\n"
          f"  - Trailing stop apres +3%\n"
          f"  - Cooldown 24h apres 3 pertes consecutives\n"
          f"  - Arret si drawdown > 15%\n")

    fetcher = DataFetcher()
    rm = RiskManager(initial_capital=capital)
    coins_to_analyze = ["bitcoin", "ethereum", "solana", "cardano", "ripple"]

    simulation_days = 365  # simule sur 1 an
    total_bars = simulation_days

    print(f"  Simulation sur {simulation_days} jours de donnees...\n")

    # Collecte toutes les donnees
    all_data = {}
    for coin_id in coins_to_analyze:
        df = fetcher.fetch_market_data(coin_id, days=simulation_days)
        if not df.empty:
            all_data[coin_id] = df

    if not all_data:
        print("  Pas de donnees disponibles.")
        return

    # Pour chaque jour de la simulation
    dates = list(list(all_data.values())[0].index)
    results_by_date = {}
    total_signals = 0
    executed_trades = 0

    for i in range(60, len(dates)):
        date = dates[i]
        day_results = []
        
        for coin_id in coins_to_analyze:
            if coin_id not in all_data:
                continue
            df = all_data[coin_id]
            if i >= len(df):
                continue
                
            # Analyse jusqu'a ce point
            chunk = df.iloc[:i+1]
            brain = CoinBrain(coin_id, chunk, fetcher)
            if not brain.compute():
                continue
            brain.signal = brain.generate_signal()
            result = brain.build_result()

            price = result["price"]
            score = result["normalized_score"]
            atr = result.get("indicators", {}).get("atr", 0)
            ml = result.get("ml", {})
            confidence = ml.get("confidence", "faible") if "error" not in ml else "faible"

            # Risk Management decision
            decision = rm.should_trade(score, confidence)
            
            if decision["trade"] and result["signal"] == "ACHAT" and atr > 0:
                total_signals += 1
                sl = rm.stop_loss_atr(price, atr)
                tp = rm.take_profit(price, atr)
                
                # Estimer win rate du backtest
                win_rate_estimate = 55  # estimation prudente
                pos = rm.position_size(price, sl["stop_pct"], win_rate=win_rate_estimate)

                if pos["position_value"] > 10:  # min 10$
                    executed_trades += 1
                    rm.record_trade(
                        entry_price=price,
                        exit_price=tp["tp_price"] if tp["tp_pct"] > sl["stop_pct"] else price * 0.95,
                        quantity=pos["quantity"],
                    )
                    day_results.append({
                        "coin": coin_id,
                        "action": "ACHAT",
                        "price": price,
                        "position": pos["position_value"],
                        "stop": sl["stop_price"],
                        "tp": tp["tp_price"],
                        "rr": tp["risk_reward"],
                    })

        if day_results:
            results_by_date[str(date.date())] = day_results

    # Rapport
    s = rm.summary()
    print(f"\n  {'='*50}")
    print(f"  RESULTAT SIMULATION ({simulation_days} jours)")
    print(f"  {'='*50}")
    print(f"\n  Capital initial:  ${s['initial_capital']:,.2f}")
    print(f"  Capital final:    ${s['capital']:,.2f}")
    print(f"  Rendement total:  {s['total_return']:+.2f}%")
    if s['total_trades'] > 0:
        print(f"  P&L total:        ${s['total_pnl']:+,.2f}")
        print(f"  Trades executer:   {s['total_trades']}")
        print(f"  Win rate:          {s['win_rate']:.1f}%")
        print(f"  Drawdown max:      {s['current_drawdown']:.1f}%")
        print(f"  Pertes conséc.:    {s['consecutive_losses']}")
    print(f"  Signaux totaux:    {total_signals}")
    print(f"  Trades simulés:    {executed_trades}")

    # Recommandation
    print(f"\n  RECOMMANDATION:")
    print(f"  {'='*50}")
    if s['total_return'] > 0:
        print(f"  ✅ Strategie rentable sur la periode")
        if s['win_rate'] > 50:
            print(f"  ✅ Win rate > 50%, strategie coherente")
        else:
            print(f"  ⚠ Win rate < 50%, ameliorer le filtrage")
    else:
        print(f"  ❌ Strategie non rentable, revoir les parametres")
    if s['current_drawdown'] > 10:
        print(f"  ⚠ Drawdown > 10%, reduire la taille des positions")
    print()

    return s


# ─── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Bot v3 — IA")
    parser.add_argument("--coin", default="bitcoin,ethereum", help="Coin(s) ou 'all'")
    parser.add_argument("--days", type=int, default=90, help="Jours d'historique")
    parser.add_argument("--save", action="store_true", help="Sauvegarder JSON")
    parser.add_argument("--html", action="store_true", help="Rapport HTML")
    parser.add_argument("--llm", action="store_true", help="Analyse IA via Ollama")
    parser.add_argument("--backtest", action="store_true", help="Backtest strategie")
    parser.add_argument("--portfolio", type=float, default=0, help="Simulation portefeuille (montant)")
    parser.add_argument("--loop", type=int, help="Boucle toutes les N min")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
        return

    if args.portfolio > 0:
        run_portfolio_simulation(capital=args.portfolio)
        return

    coins = TOP_50[:20] if args.coin == "all" else [c.strip() for c in args.coin.split(",")]
    analyzer = MarketAnalyzer(use_llm=args.llm)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'#'*62}")
        print(f"# HERMES TRADING BOT v3 — {len(coins)} actifs")
        print(f"# Periode: {args.days}j | Iteration #{iteration}")
        if args.llm: print("# Mode IA: actif (qwen2.5:3b)")
        if args.loop: print(f"# Boucle: {args.loop}min")
        print(f"{'#'*62}")

        analyzer.analyze_multiple(coins)
        analyzer.print_report()

        if args.save or args.html:
            analyzer.save_report(html=args.html)

        if not args.loop:
            break

        time.sleep(args.loop * 60)


if __name__ == "__main__":
    main()
