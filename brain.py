#!/usr/bin/env python3
"""
brain.py — Module d'intelligence pour Hermes Trading Bot v4.
3 features avancées :
  1. Market Regime Detection (régime de marché adaptatif)
  2. Analyse de sentiment via Tavily API
  3. Matrice de corrélation + suggestions de diversification

Importé par bot.py pour enrichir l'analyse.
"""

import json
import time
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger("hermes.brain")

# ─── Chemin de la clé Tavily ──────────────────────────────────────
TAVILY_KEY_PATH = Path("/opt/data/scripts/.tavily_key")
TAVILY_URL = "https://api.tavily.com/search"

# ─── Feature 1: Market Regime Detection ────────────────────────────

class MarketRegimeDetector:
    """
    Détecte le régime de marché actuel parmi 4 états :
      - Trending haussier / baissier  → favoriser trades directionnels
      - Ranging (latéral)             → favoriser trades de range
      - Volatile                      → réduire taille positions, stops larges
      - Calm (faible volatilité)      → arrêter de trader, attendre

    Utilise ADX, Bollinger Width, et pente de régression linéaire.
    """

    REGIMES = ["trending_bullish", "trending_bearish", "ranging", "volatile", "calm"]

    @staticmethod
    def detect(close: pd.Series,
               adx_val: Optional[float] = None,
               bb_width_val: Optional[float] = None) -> dict:
        """
        Analyse le régime de marché à partir d'une série de prix close
        et des valeurs ADX / Bollinger Width si fournies.

        Retourne un dict avec :
          - regime: str (nom du régime)
          - label_fr: str (libellé français)
          - confidence: float (0.0 à 1.0)
          - details: dict (scores intermédiaires)
          - recommandation: str
        """
        if close.empty or len(close) < 30:
            return {
                "regime": "inconnu",
                "label_fr": "Inconnu (données insuffisantes)",
                "confidence": 0.0,
                "details": {},
                "recommandation": "Attendre plus de données"
            }

        # 1. ADX → force de tendance
        if adx_val is not None and pd.notna(adx_val):
            adx_score = adx_val
        else:
            # Calcul ADX simplifié si non fourni
            delta = close.diff()
            up = delta.where(delta > 0, 0.0)
            dn = (-delta.where(delta < 0, 0.0))
            avg_up = up.rolling(14).mean()
            avg_dn = dn.rolling(14).mean()
            rs = avg_up / avg_dn.replace(0, np.nan)
            adx_series = 100 - (100 / (1 + rs))
            adx_score = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 20

        # 2. Bollinger Width → volatilité
        if bb_width_val is not None and pd.notna(bb_width_val):
            bbw = bb_width_val
        else:
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bbw = float(((sma20 + 2 * std20) - (sma20 - 2 * std20)).iloc[-1] / sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) and sma20.iloc[-1] > 0 else 0.0

        # 3. Pente de régression linéaire 30j → direction
        period = min(30, len(close))
        x = np.arange(period)
        y = close[-period:].values
        slope = 0.0
        if np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
        slope_pct = slope / y.mean() * 100 if y.mean() > 0 else 0

        # ─── Scoring ─────────────────────────────────────────────
        is_volatile = bbw > 0.3           # Bollinger très large
        is_calm = bbw < 0.1               # Bollinger très étroit
        is_trending = adx_score > 25       # Tendance forte
        is_ranging = adx_score < 20        # Pas de tendance
        is_strong_bullish = slope_pct > 0.15
        is_strong_bearish = slope_pct < -0.15

        # Détermination du régime
        if is_volatile:
            regime = "volatile"
            confidence = min(bbw * 2.0, 1.0)
            recommandation = (
                "Volatilité élevée — réduire la taille des positions de 50%, "
                "élargir les stops, privilégier les actifs défensifs"
            )
        elif is_calm:
            regime = "calm"
            confidence = min(1.0 - bbw * 5, 0.95)
            recommandation = (
                "Marché calme — faible volatilité, attendre un signal fort "
                "avant d'entrer, réduire l'exposition"
            )
        elif is_trending and is_strong_bullish:
            regime = "trending_bullish"
            confidence = min(adx_score / 50 + abs(slope_pct) * 3, 0.98)
            recommandation = (
                "Tendance haussière forte — favoriser les trades directionnels "
                "longs, utiliser des trailing stops"
            )
        elif is_trending and is_strong_bearish:
            regime = "trending_bearish"
            confidence = min(adx_score / 50 + abs(slope_pct) * 3, 0.98)
            recommandation = (
                "Tendance baissière forte — favoriser les shorts ou rester en cash, "
                "stops serrés, pas de buy the dip"
            )
        elif is_ranging:
            regime = "ranging"
            confidence = min((20 - adx_score) / 15, 0.9)
            recommandation = (
                "Marché range / latéral — stratégie d'achat près du support, "
                "vente près de la résistance, taille modérée"
            )
        else:
            # Zone grise: ADX entre 20 et 25
            if slope_pct > 0:
                regime = "trending_bullish"
                confidence = 0.4
                recommandation = "Tendance naissante haussière — vigilance, entrées progressives"
            elif slope_pct < 0:
                regime = "trending_bearish"
                confidence = 0.4
                recommandation = "Tendance naissante baissière — vigilance, réduire les longs"
            else:
                regime = "ranging"
                confidence = 0.3
                recommandation = "Marché indécis — attendre un signal plus clair"

        # Labels français
        labels = {
            "trending_bullish": "Tendance haussière 📈",
            "trending_bearish": "Tendance baissière 📉",
            "ranging": "Range / Latéral ↔️",
            "volatile": "Volatile ⚡",
            "calm": "Calme 😴",
        }

        # Ajustement du facteur de position selon le régime
        position_factor = {
            "trending_bullish": 1.0,
            "trending_bearish": 0.7,
            "ranging": 0.5,
            "volatile": 0.5,
            "calm": 0.2,
        }.get(regime, 0.5)

        return {
            "regime": regime,
            "label_fr": labels.get(regime, regime),
            "confidence": round(confidence, 3),
            "position_factor": position_factor,
            "recommandation": recommandation,
            "details": {
                "adx": round(adx_score, 1),
                "bb_width": round(bbw, 4),
                "slope_pct": round(slope_pct, 4),
                "is_volatile": is_volatile,
                "is_calm": is_calm,
                "is_trending": is_trending,
                "is_ranging": is_ranging,
            }
        }

    @staticmethod
    def adapt_strategy(regime_data: dict, current_score: float,
                       current_position_size: float) -> dict:
        """
        Adapte la stratégie de trading en fonction du régime détecté.
        """
        regime = regime_data.get("regime", "inconnu")
        factor = regime_data.get("position_factor", 0.5)
        confidence = regime_data.get("confidence", 0.0)

        new_size = current_position_size * factor

        if regime == "calm" and confidence > 0.5:
            action = "ATTENDRE"
        elif regime == "volatile":
            action = "REDUIRE_RISQUE"
        elif regime in ("trending_bullish", "trending_bearish") and confidence > 0.6:
            action = "TRADER_DIRECTIONNEL"
        elif regime == "ranging":
            action = "TRADER_RANGE"
        else:
            action = "PRUDENCE"

        return {
            "action": action,
            "taille_adaptee": round(new_size, 4),
            "facteur_regime": factor,
            "confiance_regime": confidence,
        }


# ─── Feature 2: Analyse de sentiment via Tavily ─────────────────────

class SentimentAnalyzer:
    """
    Analyse le sentiment du marché crypto via Tavily Search API.
    Cherche les news du jour et attribue un score de sentiment (-1 à +1).
    """

    def __init__(self):
        self.api_key = self._load_key()
        self.cache = {}
        self.cache_ttl = 3600  # 1h

    def _load_key(self) -> Optional[str]:
        """Charge la clé API Tavily depuis le fichier."""
        try:
            if TAVILY_KEY_PATH.exists():
                key = TAVILY_KEY_PATH.read_text().strip()
                if key:
                    return key
            log.warning("Clé Tavily introuvable dans %s", TAVILY_KEY_PATH)
            return None
        except Exception as e:
            log.error("Erreur lecture clé Tavily: %s", e)
            return None

    def _search(self, query: str, max_results: int = 5) -> list:
        """Requête Tavily Search API."""
        if not self.api_key:
            return []

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            }
            r = requests.post(TAVILY_URL, json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                return data.get("results", [])
            else:
                log.warning("Tavily API error %s: %s", r.status_code, r.text[:200])
                return []
        except requests.Timeout:
            log.warning("Tavily API timeout pour query: %s", query[:50])
            return []
        except Exception as e:
            log.error("Erreur Tavily API: %s", e)
            return []

    def _analyze_title_sentiment(self, title: str) -> float:
        """
        Analyse le sentiment d'un titre de news.
        Retourne un score entre -1 (très négatif) et +1 (très positif).
        """
        title_lower = title.lower()

        # Mots positifs
        positive_words = [
            "bullish", "rally", "surge", "surges", "soar", "soars", "soaring",
            "gain", "gains", "green", "moon", "pump", "breakout", "up",
            "positive", "optimistic", "adoption", "institutional", "approval",
            "launch", "partnership", "upgrade", "growth", "boom", "strong",
            "hausse", "haussier", "vert", "bond", "rebond", "explosion",
            "record", "新高", "上昇", "上昇中",
        ]
        # Mots négatifs
        negative_words = [
            "bearish", "crash", "dump", "drop", "drops", "dropping", "plunge",
            "plunges", "fall", "falls", "falling", "decline", "declines",
            "red", "loss", "losses", "fear", "panic", "sell-off", "selloff",
            "negative", "ban", "crackdown", "regulation", "hack", "exploit",
            "liquidation", "recession", "inflation", "war", "crisis",
            "baissier", "baisse", "rouge", "effondrement", "chute",
            "暴跌", "下跌", "下げ", "弱気",
        ]
        # Intensificateurs
        intensifiers = [
            "major", "huge", "massive", "extreme", "significant", "strong",
            "violent", "brutal", "énorme", "massif", "significatif",
        ]

        pos_count = sum(1 for w in positive_words if w in title_lower)
        neg_count = sum(1 for w in negative_words if w in title_lower)
        int_count = sum(1 for w in intensifiers if w in title_lower)

        # Score brut
        score = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        # Application des intensificateurs
        if score > 0 and int_count > 0:
            score *= min(1.0 + int_count * 0.2, 1.5)
        elif score < 0 and int_count > 0:
            score *= min(1.0 + int_count * 0.2, 1.5)

        # Normalisation dans [-1, 1]
        return max(-1.0, min(1.0, score))

    def get_sentiment(self, force_refresh: bool = False) -> dict:
        """
        Analyse le sentiment général du marché crypto.
        Retourne un dict avec score global, détails et news.

        Cache les résultats 1 heure.
        """
        cache_key = "sentiment_global"
        now = time.time()

        if not force_refresh and cache_key in self.cache:
            cached = self.cache[cache_key]
            if now - cached["timestamp"] < self.cache_ttl:
                return cached["data"]

        if not self.api_key:
            return {
                "available": False,
                "score": 0.0,
                "label": "Non disponible (pas de clé Tavily)",
                "news_count": 0,
                "news": [],
                "detail": {},
            }

        # Requêtes Tavily
        queries = [
            "crypto market news today",
            "bitcoin sentiment today",
        ]

        all_results = []
        for query in queries:
            results = self._search(query, max_results=5)
            all_results.extend(results)
            time.sleep(0.5)  # Polite delay entre requêtes

        # Déduplication simple par titre
        seen_titles = set()
        unique_news = []
        for r in all_results:
            title = r.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(r)

        # Limiter à 10 news max
        unique_news = unique_news[:10]

        if not unique_news:
            return {
                "available": True,
                "score": 0.0,
                "label": "Neutre (pas de news récentes)",
                "news_count": 0,
                "news": [],
                "detail": {"note": "Aucune news trouvée via Tavily"},
            }

        # Analyse de sentiment de chaque titre
        sentiments = []
        news_data = []
        for article in unique_news:
            title = article.get("title", "")
            s = self._analyze_title_sentiment(title)
            sentiments.append(s)
            news_data.append({
                "title": title,
                "url": article.get("url", ""),
                "sentiment": round(s, 3),
                "sentiment_label": "positif" if s > 0.2 else "negatif" if s < -0.2 else "neutre",
            })

        # Score global (moyenne pondérée: les plus extrêmes comptent plus)
        if sentiments:
            weights = [abs(s) + 0.5 for s in sentiments]
            total_w = sum(weights)
            global_score = sum(s * w for s, w in zip(sentiments, weights)) / total_w if total_w > 0 else 0
        else:
            global_score = 0.0

        # Label
        if global_score > 0.3:
            label = "Positif 😀"
        elif global_score > 0.1:
            label = "Légèrement positif 🙂"
        elif global_score < -0.3:
            label = "Négatif 😟"
        elif global_score < -0.1:
            label = "Légèrement négatif 🙁"
        else:
            label = "Neutre 😐"

        # Compter positifs/négatifs
        pos_count = sum(1 for s in sentiments if s > 0.1)
        neg_count = sum(1 for s in sentiments if s < -0.1)
        neut_count = len(sentiments) - pos_count - neg_count

        result = {
            "available": True,
            "score": round(global_score, 3),
            "label": label,
            "news_count": len(news_data),
            "news": news_data[:8],  # Top 8 pour lisibilité
            "detail": {
                "positifs": pos_count,
                "negatifs": neg_count,
                "neutres": neut_count,
                "total_analyses": len(sentiments),
            }
        }

        # Mise en cache
        self.cache[cache_key] = {"timestamp": now, "data": result}
        return result

    def get_sentiment_score_adjustment(self, sentiment_score: float) -> float:
        """
        Convertit le score de sentiment en ajustement pour le scoring.
        Retourne un multiplicateur entre 0.8 et 1.2.
        """
        # Sentiment très positif → boost léger
        # Sentiment très négatif → pénalité légère
        adjustment = 1.0 + sentiment_score * 0.15
        return round(max(0.8, min(1.2, adjustment)), 3)


# ─── Feature 3: Matrice de corrélation + diversification ────────────

class CorrelationMatrix:
    """
    Calcule la matrice de corrélation des rendements entre actifs
    et suggère des allocations diversifiées.
    """

    def __init__(self):
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[float] = None

    def compute(self, price_data: dict[str, pd.DataFrame],
                days: int = 30, force: bool = False) -> Optional[pd.DataFrame]:
        """
        Calcule la matrice de corrélation des rendements journaliers
        sur `days` jours pour tous les coins dans price_data.

        price_data: dict {coin_id: DataFrame avec colonne 'close'}
        Retourne un DataFrame (coins x coins) avec les corrélations.
        """
        if not price_data:
            return None

        # Extraire les rendements journaliers pour chaque coin
        returns_dict = {}
        for coin_id, df in price_data.items():
            if df.empty or len(df) < days + 5:
                continue
            close = df["close"].astype(float)
            rets = close.pct_change().dropna().tail(days)
            if len(rets) >= 10:  # Minimum 10 points pour une corrélation utile
                returns_dict[coin_id] = rets

        if len(returns_dict) < 2:
            log.info("Corrélation: besoin d'au moins 2 actifs avec données")
            return None

        # DataFrame des rendements
        rets_df = pd.DataFrame(returns_dict)

        # Matrice de corrélation (Pearson)
        corr = rets_df.corr(method="pearson")

        self.correlation_matrix = corr
        self.last_update = time.time()
        return corr

    def get_summary(self) -> dict:
        """
        Retourne un résumé de la matrice de corrélation :
        - corrélation moyenne
        - paires les plus corrélées
        - paires les moins corrélées
        - suggestions de diversification
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {"disponible": False, "message": "Pas encore de matrice calculée"}

        corr = self.correlation_matrix
        coins = corr.columns.tolist()
        n = len(coins)

        if n < 2:
            return {"disponible": False, "message": "Moins de 2 actifs disponibles"}

        # Corrélation moyenne (hors diagonale)
        mask = np.ones(corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr.values[mask].mean()

        # Paires les plus / moins corrélées
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "coin1": coins[i],
                    "coin2": coins[j],
                    "correlation": round(corr.iloc[i, j], 3),
                })

        pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
        top_correlated = pairs_sorted[:5]
        least_correlated = sorted(pairs, key=lambda x: abs(x["correlation"]))[:5]

        # Suggestions de diversification
        suggestions = []
        if least_correlated:
            # Les paires les moins corrélées sont bonnes pour la diversification
            for p in least_correlated[:3]:
                if abs(p["correlation"]) < 0.5:
                    suggestions.append(
                        f"{p['coin1']} + {p['coin2']} (corr {p['correlation']:+.3f}) — bonne diversification"
                    )

        if not suggestions and n >= 3:
            # Trouver le coin le moins corrélé en moyenne
            avg_corr_per_coin = {}
            for coin in coins:
                others = [c for c in coins if c != coin]
                avg_c = corr.loc[coin, others].mean()
                avg_corr_per_coin[coin] = avg_c

            if avg_corr_per_coin:
                best_div = min(avg_corr_per_coin, key=avg_corr_per_coin.get)
                suggestions.append(
                    f"{best_div} est le moins corrélé aux autres (moy. {avg_corr_per_coin[best_div]:.3f})"
                )

        # Portfolio équipondéré suggéré
        if n >= 3:
            # Sélectionner les actifs les moins corrélés entre eux (naïf: garder les 3-4
            # avec la corrélation croisée la plus faible)
            selected = self._select_diversified(coins, corr, max_count=min(5, n))
            allocation = {c: round(1.0 / len(selected), 3) for c in selected}
        else:
            allocation = {c: 1.0 / n for c in coins}

        return {
            "disponible": True,
            "actifs": len(coins),
            "correlation_moyenne": round(avg_corr, 3),
            "top_correlees": top_correlated,
            "moins_correlees": least_correlated[:3],
            "suggestions_diversification": suggestions,
            "allocation_suggeree": allocation,
            "matrice": corr.round(3).to_dict(),  # Pour sérialisation JSON
        }

    def _select_diversified(self, coins: list[str], corr: pd.DataFrame,
                            max_count: int = 4) -> list[str]:
        """
        Algorithme glouton simple pour sélectionner un sous-ensemble
        d'actifs faiblement corrélés.
        """
        if len(coins) <= max_count:
            return coins

        # Commencer avec l'actif ayant la plus faible corrélation moyenne
        avg_corrs = {c: corr.loc[c, [x for x in coins if x != c]].mean() for c in coins}
        selected = [min(avg_corrs, key=avg_corrs.get)]

        while len(selected) < max_count:
            best_coin = None
            best_avg_corr = float("inf")
            for c in coins:
                if c in selected:
                    continue
                # Corrélation moyenne de c avec les déjà sélectionnés
                avg_c = corr.loc[c, selected].mean()
                if avg_c < best_avg_corr:
                    best_avg_corr = avg_c
                    best_coin = c
            if best_coin:
                selected.append(best_coin)
            else:
                break

        return selected

    def portfolio_diversification_score(self, allocation: dict[str, float]) -> float:
        """
        Calcule un score de diversification (0 à 1) pour une allocation donnée.
        Plus le score est élevé, mieux diversifié.
        """
        if self.correlation_matrix is None or len(allocation) < 2:
            return 0.0

        coins = list(allocation.keys())
        weights = np.array([allocation[c] for c in coins])

        # Matrice de corrélation pour ces coins
        sub_corr = self.correlation_matrix.loc[coins, coins].values

        # Variance du portefeuille = w^T * Sigma * w
        # où Sigma = corr (approximé, en ignorant les volatilités individuelles)
        port_var = weights @ sub_corr @ weights

        # Variance moyenne pondérée = sum(w_i^2) (si toutes corr=1, port_var=1)
        # Score = 1 - (port_var - min_var) / (max_var - min_var) normalisé
        min_var = weights @ np.eye(len(coins)) @ weights  # Si corr=0
        max_var = weights @ np.ones((len(coins), len(coins))) @ weights  # Si corr=1

        if max_var == min_var:
            return 0.5

        score = 1.0 - (port_var - min_var) / (max_var - min_var)
        return round(max(0.0, min(1.0, score)), 3)

    def to_json_serializable(self) -> dict:
        """Convertit la matrice en format JSON-serializable."""
        if self.correlation_matrix is None:
            return {}
        result = {}
        for col in self.correlation_matrix.columns:
            result[col] = {
                k: round(float(v), 3)
                for k, v in self.correlation_matrix[col].items()
            }
        return result


# ─── Feature 4: Multi-timeframe Analysis (CT/MT/LT) ────────────────

class TimeframeAnalyzer:
    """
    Analyse multi-timeframe : Court Terme (7j), Moyen Terme (30j), Long Terme (365j).
    Utilise les mêmes données quotidiennes mais avec des périodes différentes.
    Le signal final est la moyenne pondérée des 3 timeframes.

    Poids :
      - CT (Court Terme)  poids 1 → réaction rapide
      - MT (Moyen Terme)  poids 2 → tendance principale
      - LT (Long Terme)   poids 3 → tendance de fond
    """

    PERIODS = {
        "CT": {"days": 7,  "label": "Court Terme",  "weight": 1},
        "MT": {"days": 30, "label": "Moyen Terme",  "weight": 2},
        "LT": {"days": 365,"label": "Long Terme",   "weight": 3},
    }

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        """RSI simple sur une série, retourne la dernière valeur."""
        if len(series) < period + 1:
            return 50.0
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_g = gain.ewm(span=period, adjust=False).mean()
        avg_l = loss.ewm(span=period, adjust=False).mean()
        rs = avg_g / avg_l.replace(0, np.nan)
        rsi_s = 100 - (100 / (1 + rs))
        val = float(rsi_s.iloc[-1]) if pd.notna(rsi_s.iloc[-1]) else 50.0
        return val

    @staticmethod
    def _compute_trend(close: pd.Series) -> dict:
        """Tendance simple par régression linéaire."""
        n = len(close)
        if n < 5:
            return {"trend": "neutral", "strength": 0, "slope_pct": 0}
        x = np.arange(n)
        y = close.values
        if np.std(y) == 0:
            return {"trend": "neutral", "strength": 0, "slope_pct": 0}
        slope = np.polyfit(x, y, 1)[0]
        norm_slope = slope / y.mean() * 100
        strength = min(abs(norm_slope) * 10, 100)
        trend = "bullish" if norm_slope > 0.1 else "bearish" if norm_slope < -0.1 else "neutral"
        return {"trend": trend, "strength": round(strength, 1), "slope_pct": round(norm_slope, 4)}

    @staticmethod
    def analyze(close: pd.Series, volume: pd.Series) -> dict:
        """
        Analyse les 3 timeframes et retourne un dict structuré.

        Retourne :
          - timeframes: dict {CT: {...}, MT: {...}, LT: {...}}
            chaque timeframe contient : score, signal, niveau, rsi, trend, pct_change
          - weighted_score: float (combiné pondéré)
          - signal / niveau : signal final
        """
        results = {}
        for tf_key, cfg in TimeframeAnalyzer.PERIODS.items():
            period = min(cfg["days"], len(close))
            if period < 5:
                results[tf_key] = {
                    "score": 0.0, "signal": "NEUTRE", "niveau": "FAIBLE",
                    "rsi": None, "trend": "neutral", "pct_change": 0.0,
                    "label": cfg["label"], "weight": cfg["weight"],
                    "error": "données insuffisantes",
                }
                continue

            sub_close = close.iloc[-period:]
            sub_volume = volume.iloc[-period:] if len(volume) >= period else volume

            # RSI
            rsi_val = TimeframeAnalyzer._compute_rsi(sub_close)

            # Tendance
            trend = TimeframeAnalyzer._compute_trend(sub_close)

            # MACD simplifié (EMA12 - EMA26)
            ema12 = sub_close.ewm(span=12, adjust=False).mean()
            ema26 = sub_close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            macd_bullish = float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1]) if len(macd_line) > 0 else False

            # Variation de prix sur la période
            pct_change = (float(sub_close.iloc[-1]) - float(sub_close.iloc[0])) / float(sub_close.iloc[0]) * 100

            # ── Scoring sur ce timeframe ──
            score = 0.0
            max_score = 0.0

            # RSI (poids 2)
            w = 2.0; max_score += w
            if rsi_val < 30: score += w
            elif rsi_val < 40: score += w * 0.5
            elif rsi_val > 70: score -= w
            elif rsi_val > 60: score -= w * 0.5

            # MACD (poids 2)
            w = 2.0; max_score += w
            if macd_bullish: score += w
            else: score -= w

            # Tendance (poids 2)
            w = 2.0; max_score += w
            if trend["trend"] == "bullish": score += w * min(trend["strength"] / 100, 1.0)
            elif trend["trend"] == "bearish": score -= w * min(trend["strength"] / 100, 1.0)

            # Momentum prix (poids 1)
            w = 1.0; max_score += w
            if pct_change > 5: score += w
            elif pct_change > 2: score += w * 0.5
            elif pct_change < -5: score -= w
            elif pct_change < -2: score -= w * 0.5

            normalized = score / max_score if max_score > 0 else 0

            # Signal
            if normalized >= 0.25:
                signal = "ACHAT"; niveau = "FORT" if normalized >= 0.45 else "MOYEN"
            elif normalized <= -0.25:
                signal = "VENTE"; niveau = "FORT" if normalized <= -0.45 else "MOYEN"
            else:
                signal = "NEUTRE"; niveau = "FAIBLE"

            results[tf_key] = {
                "score": round(normalized, 4),
                "signal": signal,
                "niveau": niveau,
                "rsi": round(rsi_val, 1),
                "trend": trend["trend"],
                "trend_strength": trend["strength"],
                "pct_change": round(pct_change, 2),
                "label": cfg["label"],
                "weight": cfg["weight"],
                "macd_bullish": macd_bullish,
            }

        # Score combiné pondéré
        total_weight = sum(cfg["weight"] for cfg in TimeframeAnalyzer.PERIODS.values())
        weighted_sum = sum(
            results[tf_key]["score"] * TimeframeAnalyzer.PERIODS[tf_key]["weight"]
            for tf_key in results
        )
        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Signal final
        if weighted_score >= 0.25:
            final_signal = "ACHAT"; final_niveau = "FORT" if weighted_score >= 0.45 else "MOYEN"
        elif weighted_score <= -0.25:
            final_signal = "VENTE"; final_niveau = "FORT" if weighted_score <= -0.45 else "MOYEN"
        else:
            final_signal = "NEUTRE"; final_niveau = "FAIBLE"

        return {
            "timeframes": results,
            "weighted_score": round(weighted_score, 4),
            "signal": final_signal,
            "niveau": final_niveau,
        }


# ─── Feature 5: Alert Manager intelligent ───────────────────────────

class AlertManager:
    """
    Gère les alertes intelligentes du bot.
    Détecte et remonte les événements importants :
      - Signal FORT sur un coin
      - Divergence haussière confirmée
      - Bollinger Squeeze (volatilité imminente)
      - Score normalisé extrême (> +0.5 ou < -0.5)
      - Changement de régime de marché

    Les alertes sont dédoublonnées (pas de répétition du même type+coin).
    """

    def __init__(self):
        self.alerts: list[dict] = []
        self._seen: set[tuple[str, str]] = set()  # (type, coin) → dédoublonnage
        self._regime_history: dict[str, str] = {}  # coin_id → dernier régime connu

    def add_alert(self, alert_type: str, coin: str, message: str,
                  score: float = 0.0, timestamp: Optional[str] = None):
        """
        Ajoute une alerte si elle n'existe pas déjà (dédoublonnage type+coin).

        Args:
            alert_type: Type d'alerte (SIGNAL_FORT, DIVERGENCE, SQUEEZE, SCORE_EXTREME, REGIME_CHANGE)
            coin: Identifiant du coin concerné
            message: Message descriptif
            score: Score associé (optionnel)
            timestamp: Horodatage (auto si None)
        """
        # Dédoublonnage
        dedup_key = (alert_type, coin)
        if dedup_key in self._seen:
            return

        alert = {
            "type": alert_type,
            "coin": coin,
            "message": message,
            "score": round(score, 4),
            "timestamp": timestamp or datetime.now().strftime("%H:%M:%S"),
        }
        self.alerts.append(alert)
        self._seen.add(dedup_key)
        log.info("ALERTE [%s] %s: %s", alert_type, coin, message)

    def check_result(self, result: Any, coin_id: str):
        """
        Analyse un CoinResult et génère les alertes appropriées.
        """
        # Importé ici pour éviter circular import (le type hint est Any)
        # 1. Signal FORT
        if hasattr(result, 'signal') and hasattr(result, 'signal_niveau'):
            if result.signal in ("ACHAT", "VENTE") and result.signal_niveau == "FORT":
                self.add_alert(
                    "SIGNAL_FORT", coin_id,
                    f"Signal {result.signal} FORT (score {result.normalized_score:+.2f})",
                    score=result.normalized_score,
                )

        # 2. Divergence haussière confirmée
        if hasattr(result, 'divergences') and result.divergences:
            for div in result.divergences:
                if div.get("type") == "bullish" and div.get("strength") == "strong":
                    self.add_alert(
                        "DIVERGENCE", coin_id,
                        f"Divergence haussière confirmée à ${div.get('price', 0):.2f} "
                        f"(RSI {div.get('rsi', 0):.1f})",
                        score=result.normalized_score,
                    )

        # 3. Bollinger Squeeze
        has_squeeze = False
        if hasattr(result, 'indicators') and result.indicators:
            bb_p = result.indicators.get("bb_percent")
            # Squeeze détecté si BB% est très bas ou si le pattern le mentionne
        if hasattr(result, 'patterns'):
            for p in result.patterns:
                if "Squeeze" in p or "volatilite" in p.lower():
                    has_squeeze = True
                    break
        if has_squeeze:
            self.add_alert(
                "SQUEEZE", coin_id,
                "Bollinger Squeeze détecté — volatilité imminente",
                score=result.normalized_score,
            )

        # 4. Score extrême
        if hasattr(result, 'normalized_score'):
            ns = result.normalized_score
            if ns > 0.5:
                self.add_alert(
                    "SCORE_EXTREME", coin_id,
                    f"Score très haussier ({ns:+.2f}) — signal fort à surveiller",
                    score=ns,
                )
            elif ns < -0.5:
                self.add_alert(
                    "SCORE_EXTREME", coin_id,
                    f"Score très baissier ({ns:+.2f}) — risque de baisse",
                    score=ns,
                )

        # 5. Changement de régime
        if hasattr(result, 'regime') and result.regime:
            current_regime = result.regime.get("regime", "")
            prev_regime = self._regime_history.get(coin_id)
            if prev_regime and current_regime and current_regime != prev_regime:
                self.add_alert(
                    "REGIME_CHANGE", coin_id,
                    f"Changement de régime : {prev_regime} → {current_regime}",
                    score=result.normalized_score,
                )
            if current_regime:
                self._regime_history[coin_id] = current_regime

    def clear(self):
        """Réinitialise toutes les alertes."""
        self.alerts.clear()
        self._seen.clear()

    def get_alerts(self, alert_type: Optional[str] = None,
                   min_score: float = 0.0) -> list[dict]:
        """
        Récupère les alertes, filtrées optionnellement.

        Args:
            alert_type: Filtrer par type (None = tous)
            min_score: Score minimum (abs)
        """
        filtered = self.alerts
        if alert_type:
            filtered = [a for a in filtered if a["type"] == alert_type]
        if min_score > 0:
            filtered = [a for a in filtered if abs(a["score"]) >= min_score]
        return filtered

    def count(self) -> int:
        return len(self.alerts)

    def summary(self) -> dict:
        """Retourne un résumé des alertes par type."""
        by_type = {}
        for a in self.alerts:
            by_type.setdefault(a["type"], []).append(a)
        return {
            "total": len(self.alerts),
            "by_type": {t: len(v) for t, v in by_type.items()},
            "types": list(by_type.keys()),
        }

    def format_report(self) -> str:
        """Formatte les alertes pour affichage dans le rapport."""
        if not self.alerts:
            return ""

        lines = []
        lines.append(f"  ⚠ ALERTES DU JOUR ({self.count()})")
        lines.append(f"  {'─' * 60}")

        # Grouper par type pour un affichage clair
        type_icons = {
            "SIGNAL_FORT": "🔴🔵",
            "DIVERGENCE": "⚡",
            "SQUEEZE": "💥",
            "SCORE_EXTREME": "🔥",
            "REGIME_CHANGE": "🔄",
        }

        for alert in self.alerts:
            icon = type_icons.get(alert["type"], "⚠")
            lines.append(
                f"  {icon} [{alert['type']}] {alert['coin']:<15} "
                f"{alert['message']} "
                f"(score {alert['score']:+.2f})"
            )

        lines.append(f"  {'─' * 60}")
        return "\n".join(lines)


# ─── Analyseur global intelligent ──────────────────────────────────

class BrainAnalyzer:
    """
    Orchestrateur des 3 features intelligentes.
    Utilisé par bot.py pour enrichir l'analyse.
    """

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.sentiment = SentimentAnalyzer()
        self.correlation = CorrelationMatrix()
        self.timeframe = TimeframeAnalyzer()
        self.alerts = AlertManager()
        self.price_data: dict[str, pd.DataFrame] = {}
        self.regime_cache: Optional[dict] = None
        self.sentiment_cache: Optional[dict] = None

    def feed_price_data(self, coin_id: str, df: pd.DataFrame):
        """Stocke les données prix pour le calcul de corrélation."""
        if not df.empty:
            self.price_data[coin_id] = df

    def analyze_global_regime(self, close: pd.Series,
                               adx_val: Optional[float] = None,
                               bb_width_val: Optional[float] = None) -> dict:
        """Analyse le régime de marché global."""
        self.regime_cache = MarketRegimeDetector.detect(close, adx_val, bb_width_val)
        return self.regime_cache

    def analyze_sentiment(self, force_refresh: bool = False) -> dict:
        """Analyse le sentiment via Tavily (avec cache)."""
        self.sentiment_cache = self.sentiment.get_sentiment(force_refresh)
        return self.sentiment_cache

    def analyze_correlation(self, force: bool = False) -> Optional[dict]:
        """Calcule et retourne le résumé de corrélation."""
        corr_matrix = self.correlation.compute(self.price_data, force=force)
        if corr_matrix is not None:
            return self.correlation.get_summary()
        return None

    def analyze_timeframe(self, close: pd.Series, volume: pd.Series) -> dict:
        """Analyse multi-timeframe CT/MT/LT."""
        return TimeframeAnalyzer.analyze(close, volume)

    def get_alerts(self) -> 'AlertManager':
        """Retourne le gestionnaire d'alertes."""
        return self.alerts

    def reset_alerts(self):
        """Réinitialise les alertes pour une nouvelle session d'analyse."""
        self.alerts.clear()

    def process_alerts(self, result: Any, coin_id: str):
        """Analyse un résultat et génère les alertes appropriées."""
        self.alerts.check_result(result, coin_id)

    def get_full_analysis(self, close: pd.Series,
                           adx_val: Optional[float] = None,
                           bb_width_val: Optional[float] = None,
                           force_sentiment: bool = False) -> dict:
        """Analyse complète: régime + sentiment + corrélation."""
        regime = self.analyze_global_regime(close, adx_val, bb_width_val)
        sentiment = self.analyze_sentiment(force_sentiment)
        correlation = self.analyze_correlation()

        return {
            "regime": regime,
            "sentiment": sentiment,
            "correlation": correlation,
            "timestamp": datetime.now().isoformat(),
        }
