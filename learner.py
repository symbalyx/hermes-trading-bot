#!/usr/bin/env python3
"""
Hermes Learning Engine v1 — Apprentissage des trades passés
Ajuste les poids des indicateurs, les seuils, et adapte la stratégie
au régime de marché. Autonome (json/datetime/os/pathlib seulement).
"""
import json
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Structure des données par défaut ────────────────────────────────

DEFAULT_DATA = {
    "version": 1,
    "last_updated": "",
    "total_trades_tracked": 0,
    "total_wins": 0,
    "total_losses": 0,
    "indicator_weights": {
        "rsi":  {"weight": 2.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "macd": {"weight": 2.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "bollinger": {"weight": 1.5, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "adx":  {"weight": 1.5, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "stoch": {"weight": 1.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "mfi":  {"weight": 1.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "volume": {"weight": 1.5, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "trend": {"weight": 2.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
        "divergences": {"weight": 2.0, "win_rate": 0.0, "trades": 0, "avg_return": 0.0},
    },
    "regime_stats": {
        "trending_bullish": {"trades": 0, "wins": 0, "avg_return": 0.0, "total_return": 0.0},
        "trending_bearish": {"trades": 0, "wins": 0, "avg_return": 0.0, "total_return": 0.0},
        "ranging":         {"trades": 0, "wins": 0, "avg_return": 0.0, "total_return": 0.0},
        "volatile":        {"trades": 0, "wins": 0, "avg_return": 0.0, "total_return": 0.0},
        "calm":            {"trades": 0, "wins": 0, "avg_return": 0.0, "total_return": 0.0},
    },
    "thresholds": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_score": 0.25,
        "bb_oversold": 0.05,
        "bb_overbought": 0.95,
    },
    "regime_thresholds": {},
    "trade_history": [],
    "daily_adjustments": {},  # date -> {indicator -> old_weight}
}


# ─── Learning Engine ─────────────────────────────────────────────────

class LearningEngine:
    """Moteur d'apprentissage qui ajuste poids et seuils selon les trades."""

    # Variation max par jour (conservateur)
    MAX_DAILY_CHANGE = 0.20  # 20%
    # Pas d'ajustement des seuils
    THRESHOLD_STEP = 0.5
    # Poids min/max
    MIN_WEIGHT = 0.3
    MAX_WEIGHT = 4.0

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_path = self.data_dir / "learning_data.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._today = datetime.now().strftime("%Y-%m-%d")

    # ── Persistance ────────────────────────────────────────────

    def _load(self) -> dict:
        """Charge les données depuis le fichier JSON."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                # Assurer la compatibilité des clés
                for key in DEFAULT_DATA:
                    if key not in data:
                        data[key] = DEFAULT_DATA[key]
                # Assurer tous les indicateurs
                for ind_key, ind_val in DEFAULT_DATA["indicator_weights"].items():
                    if ind_key not in data.get("indicator_weights", {}):
                        data.setdefault("indicator_weights", {})[ind_key] = dict(ind_val)
                # Assurer tous les régimes
                for reg_key, reg_val in DEFAULT_DATA["regime_stats"].items():
                    if reg_key not in data.get("regime_stats", {}):
                        data.setdefault("regime_stats", {})[reg_key] = dict(reg_val)
                return data
            except (json.JSONDecodeError, OSError):
                pass
        # Copie profonde
        return json.loads(json.dumps(DEFAULT_DATA))

    def _save(self):
        """Sauvegarde les données dans le fichier JSON."""
        self.data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.data_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError as e:
            print(f"  ⚠ Sauvegarde learning_data impossible: {e}")

    # ── Enregistrement d'un trade ──────────────────────────────

    def record_trade(
        self,
        coin: str,
        entry: float,
        exit_price: float,
        pnl_pct: float,
        regime: str = "unknown",
        score: float = 0.0,
        indicator_wins: Optional[dict] = None,
    ):
        """Enregistre un trade et met à jour les stats.

        Args:
            coin: Identifiant de la crypto
            entry: Prix d'entrée
            exit_price: Prix de sortie
            pnl_pct: Pourcentage de gain/perte
            regime: Régime de marché au moment du trade
            score: Score de confiance de l'analyse
            indicator_wins: Dict {indicator_name: True/False} si l'indicateur
                           a correctement prédit la direction
        """
        trade = {
            "coin": coin,
            "entry": round(entry, 4),
            "exit": round(exit_price, 4),
            "pnl_pct": round(pnl_pct, 2),
            "regime": regime,
            "score": round(score, 4),
            "timestamp": datetime.now().isoformat(),
        }
        self.data["trade_history"].append(trade)
        self.data["total_trades_tracked"] += 1

        is_win = pnl_pct > 0
        if is_win:
            self.data["total_wins"] += 1
        else:
            self.data["total_losses"] += 1

        # Mise à jour des stats par indicateur
        if indicator_wins:
            for ind_name, won in indicator_wins.items():
                ind = self.data["indicator_weights"].get(ind_name)
                if ind is None:
                    continue
                ind["trades"] += 1
                old_wr = ind["win_rate"]
                n = ind["trades"]
                # Moyenne glissante du win rate
                ind["win_rate"] = ((old_wr * (n - 1)) + (100.0 if won else 0.0)) / n
                # Moyenne glissante du return
                ind["avg_return"] = ((ind["avg_return"] * (n - 1)) + pnl_pct) / n

        # Mise à jour des stats par régime
        reg = self.data["regime_stats"].get(regime)
        if reg is not None:
            reg["trades"] += 1
            if is_win:
                reg["wins"] += 1
            reg["total_return"] = round(reg.get("total_return", 0.0) + pnl_pct, 2)
            reg["avg_return"] = round(reg["total_return"] / reg["trades"], 4)

        # Limiter l'historique des trades (garder les 500 derniers)
        if len(self.data["trade_history"]) > 500:
            self.data["trade_history"] = self.data["trade_history"][-500:]

        self._save()

    # ── Récupération des poids ajustés ────────────────────────

    def get_weight(self, indicator_name: str, regime: str = "") -> float:
        """Retourne le poids ajusté d'un indicateur.

        Si un poids spécifique au régime existe, l'utilise. Sinon,
        retourne le poids global ajusté par le win rate.
        """
        ind = self.data["indicator_weights"].get(indicator_name)
        if ind is None:
            return 1.0

        base_weight = ind["weight"]
        win_rate = ind["win_rate"]
        trades = ind["trades"]

        # Si pas assez de trades, utiliser le poids de base
        if trades < 5:
            return base_weight

        # Ajustement basé sur le win rate
        if win_rate > 60.0:
            # Bon indicateur: +10% de poids (max +50%)
            factor = min(1 + (win_rate - 60) / 100, 1.50)
        elif win_rate < 40.0:
            # Mauvais indicateur: -10% de poids (min -50%)
            factor = max(1 - (40 - win_rate) / 100, 0.50)
        else:
            factor = 1.0

        adjusted = base_weight * factor
        return max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, round(adjusted, 3)))

    def set_weight(self, indicator_name: str, new_weight: float):
        """Définit le poids de base d'un indicateur (avec limite conservatrice)."""
        ind = self.data["indicator_weights"].get(indicator_name)
        if ind is None:
            return
        clamped = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
        ind["weight"] = round(clamped, 3)

    # ── Seuils ajustés ─────────────────────────────────────────

    def get_threshold(self, param_name: str, regime: str = "") -> float:
        """Retourne le seuil ajusté pour un paramètre.

        Si un seuil spécifique au régime existe, il est prioritaire.
        """
        base = self.data["thresholds"].get(param_name)
        if base is None:
            return 0.0

        # Vérifier les seuils spécifiques au régime
        regime_key = f"{param_name}_{regime}" if regime else ""
        reg_thresholds = self.data.get("regime_thresholds", {})
        if regime_key in reg_thresholds:
            return reg_thresholds[regime_key]

        return base

    def _adjust_threshold(self, param_name: str, direction: str):
        """Ajuste un seuil dans une direction (+ ou -) par pas."""
        current = self.data["thresholds"].get(param_name)
        if current is None:
            return

        step = self.THRESHOLD_STEP
        if direction == "lower":
            new_val = current - step
        elif direction == "raise":
            new_val = current + step
        else:
            return

        # Contraintes de sécurité
        if param_name == "rsi_oversold":
            new_val = max(15, min(40, new_val))
        elif param_name == "rsi_overbought":
            new_val = max(60, min(85, new_val))
        elif param_name == "min_score":
            new_val = max(0.05, min(0.50, new_val))
        elif param_name == "bb_oversold":
            new_val = max(0.01, min(0.20, new_val))
        elif param_name == "bb_overbought":
            new_val = max(0.80, min(0.99, new_val))
        else:
            return

        self.data["thresholds"][param_name] = round(new_val, 3)

    # ── Optimisation complète ──────────────────────────────────

    def optimize(self):
        """Réévalue tous les poids et seuils basés sur l'historique."""
        today = self._today
        daily_adj = self.data.setdefault("daily_adjustments", {})

        # Limiter les ajustements à 1x par jour
        if today in daily_adj:
            return  # Déjà optimisé aujourd'hui

        # 1. Optimiser les poids des indicateurs
        for ind_name, ind in self.data["indicator_weights"].items():
            if ind["trades"] < 5:
                continue

            old_weight = ind["weight"]
            win_rate = ind["win_rate"]

            if win_rate > 60.0:
                new_weight = old_weight * (1 + min((win_rate - 60) / 200, 0.20))
            elif win_rate < 40.0 and win_rate > 0:
                new_weight = old_weight * (1 - min((40 - win_rate) / 200, 0.20))
            else:
                continue

            clamped = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            change_pct = abs(clamped - old_weight) / old_weight if old_weight > 0 else 0

            # Limiter à MAX_DAILY_CHANGE par jour
            if change_pct > self.MAX_DAILY_CHANGE:
                direction = 1 if clamped > old_weight else -1
                clamped = old_weight * (1 + direction * self.MAX_DAILY_CHANGE)

            ind["weight"] = round(clamped, 3)
            daily_adj.setdefault(today, {})[ind_name] = round(old_weight, 3)

        # 2. Optimiser les seuils RSI
        if self.data["total_trades_tracked"] >= 10:
            trades = self.data["trade_history"]
            # Analyse des trades près du seuil oversold
            oversold_trades = [t for t in trades if t.get("rsi_adj", {}).get("near_oversold")]
            if len(oversold_trades) >= 5:
                oversold_wr = sum(1 for t in oversold_trades if t["pnl_pct"] > 0) / len(oversold_trades)
                if oversold_wr > 0.65:
                    # Le seuil actuel est trop restrictif (trop de bons trades ratés)
                    self._adjust_threshold("rsi_oversold", "raise")
                elif oversold_wr < 0.40:
                    # Le seuil actuel est trop permissif
                    self._adjust_threshold("rsi_oversold", "lower")

            # Analyse des trades près du seuil overbought
            overbought_trades = [t for t in trades if t.get("rsi_adj", {}).get("near_overbought")]
            if len(overbought_trades) >= 5:
                overbought_wr = sum(1 for t in overbought_trades if t["pnl_pct"] > 0) / len(overbought_trades)
                if overbought_wr < 0.35:
                    # Vendre près du surachat fonctionne bien — baisser le seuil
                    self._adjust_threshold("rsi_overbought", "lower")
                elif overbought_wr > 0.55:
                    # Trop de faux positifs — monter le seuil
                    self._adjust_threshold("rsi_overbought", "raise")

        # 3. Optimiser le min_score par régime
        for regime, stats in self.data["regime_stats"].items():
            if stats["trades"] < 5:
                continue
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            regime_key = f"min_score_{regime}"
            if wr < 40:
                self.data.setdefault("regime_thresholds", {})[regime_key] = round(
                    max(0.10, self.get_threshold("min_score") + 0.05), 3
                )
            elif wr > 65:
                self.data.setdefault("regime_thresholds", {})[regime_key] = round(
                    min(0.50, self.get_threshold("min_score") - 0.03), 3
                )

        daily_adj[today] = daily_adj.get(today, {})
        self._save()

    # ── Score de confiance ─────────────────────────────────────

    def get_confidence(self, regime: str = "", score: float = 0.0) -> float:
        """Retourne un score de confiance 0-100% basé sur l'historique.

        Prend en compte:
        - Le win rate global
        - La performance dans ce régime
        - Le nombre de trades d'expérience
        """
        base = 50.0  # Confiance de base

        # 1. Win rate global
        total = self.data["total_trades_tracked"]
        if total > 0:
            global_wr = self.data["total_wins"] / total * 100
            base += (global_wr - 50) * 0.3
            # Bonus d'expérience
            exp_factor = min(total / 100, 1.0)
            base += exp_factor * 10

        # 2. Performance par régime
        if regime:
            reg = self.data["regime_stats"].get(regime)
            if reg and reg["trades"] >= 3:
                reg_wr = reg["wins"] / reg["trades"] * 100 if reg["trades"] > 0 else 50
                base += (reg_wr - 50) * 0.2
                # Prime pour trades récents dans ce régime
                reg_exp = min(reg["trades"] / 20, 1.0)
                base += reg_exp * 5

        # 3. Amplitude du score (plus le score est extrême, plus on est confiant)
        base += abs(score) * 30

        return max(0.0, min(100.0, round(base, 1)))

    # ── Résumé ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Retourne un résumé formaté des stats d'apprentissage."""
        d = self.data
        lines = []

        total = d["total_trades_tracked"]
        wins = d["total_wins"]
        losses = d["total_losses"]
        wr = round(wins / total * 100, 1) if total > 0 else 0

        lines.append(f"📚 LEARNING ENGINE — {total} trades trackés")
        lines.append(f"   Win rate: {wr}% ({wins}W / {losses}L)")

        # Indicateurs
        lines.append(f"\n   📊 POIDS DES INDICATEURS:")
        for name, ind in sorted(d["indicator_weights"].items()):
            w = ind["weight"]
            wr_ind = ind["win_rate"]
            t_ind = ind["trades"]
            bar = "█" * max(1, int(w * 5))
            wr_str = f"{wr_ind:.0f}%" if t_ind > 0 else "-"
            lines.append(f"      {name:12s} {bar} x{w:.2f} (WR: {wr_str}, {t_ind}t)")

        # Régimes
        lines.append(f"\n   🏷️  PERFORMANCE PAR RÉGIME:")
        for regime, stats in sorted(d["regime_stats"].items()):
            if stats["trades"] > 0:
                wr_r = stats["wins"] / stats["trades"] * 100
                lines.append(f"      {regime:20s} {stats['trades']:3d}t "
                             f"WR {wr_r:5.1f}% return {stats['avg_return']:+.2f}%")

        # Seuils
        th = d["thresholds"]
        lines.append(f"\n   ⚙️  SEUILS:")
        lines.append(f"      RSI survente: {th['rsi_oversold']}")
        lines.append(f"      RSI surachat: {th['rsi_overbought']}")
        lines.append(f"      Score min: {th['min_score']}")
        lines.append(f"      BB survente: {th['bb_oversold']}")
        lines.append(f"      BB surachat: {th['bb_overbought']}")

        # Confiance
        lines.append(f"\n   🎯 CONFIANCE GLOBALE: {self.get_confidence():.0f}%")

        return "\n".join(lines)

    # ── Utilitaires ────────────────────────────────────────────

    def get_weights_dict(self, regime: str = "") -> dict:
        """Retourne un dict {indicator_name: weight} pour tous les indicateurs."""
        return {name: self.get_weight(name, regime)
                for name in self.data["indicator_weights"]}

    def get_trade_count(self) -> int:
        return self.data["total_trades_tracked"]

    def get_win_rate(self) -> float:
        total = self.data["total_trades_tracked"]
        if total == 0:
            return 0.0
        return self.data["total_wins"] / total * 100
