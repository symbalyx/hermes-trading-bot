# Stratégie Hermes Trading Bot v3 

## 1. Vision générale

**Hermes Trading Bot v3** est un assistant de trading automatique pour les cryptomonnaies. Son objectif n'est pas de prédire l'avenir (personne ne peut le faire), mais **d'analyser le marché de façon méthodique** et de prendre des décisions basées sur des données objectives.

Le bot combine trois niveaux d'analyse complémentaires :

- **Analyse technique** — les indicateurs classiques (RSI, MACD, Bollinger, etc.)
- **Machine Learning** — des prédictions statistiques basées sur l'historique
- **IA narrative** — un modèle de langage (Qwen 2.5) qui analyse le contexte global

Ces trois couches sont combinées en un **score unique** qui détermine la qualité du signal. Plus le score est élevé, plus la configuration de marché est favorable.

---

## 2. Philosophie : la prudence avant tout 💎

La règle la plus importante du bot est simple :

> **Ne pas perdre d'argent est plus important que d'en gagner.**

Concrètement, ça se traduit par des règles strictes :

| Règle | Détail |
|-------|--------|
| Risque max par trade | **2% du capital** — jamais plus |
| Taille de position | Calculée via **Kelly Criterion / 2** (formule mathématique qui optimise la taille sans être trop agressive) |
| Stop-loss dynamique | Basé sur la **volatilité du marché** (ATR — Average True Range) |
| Cooldown | **24h de pause** après 3 pertes consécutives |
| Arrêt automatique | Le bot s'arrête si le drawdown dépasse **15%** du capital |

> ⚠️ **Pourquoi Kelly / 2 ?** Kelly Criterion donne la taille de position mathématiquement optimale. Mais appliquer Kelly à 100% est risqué car il suppose des probabilités parfaites. En divisant par 2, on garde l'avantage mathématique tout en étant **beaucoup plus prudent**.

---

## 3. Les 3 couches d'analyse

### 🔵 Couche 1 — Analyse technique

Le bot utilise **8 indicateurs techniques** pour évaluer le marché :

- **RSI (Relative Strength Index)** — mesure si un actif est suracheté (>70) ou survendu (<30)
- **MACD** — détecte les changements de tendance (croisement de moyennes mobiles)
- **Bollinger Bands** — mesure la volatilité et repère les extrêmes de prix
- **ADX (Average Directional Index)** — indique la force de la tendance (forte >25, faible <20)
- **Stochastic RSI** — version plus sensible du RSI, utile pour les retournements
- **MFI (Money Flow Index)** — RSI pondéré par le volume (suit l'argent qui entre et sort)
- **ATR (Average True Range)** — mesure la volatilité du marché
- **Heikin Ashi** — chandeliers lissés qui filtrent le bruit du marché

Chaque indicateur donne un vote (positif, négatif, ou neutre). L'ensemble forme un **score technique** sur 10.

Le bot détecte aussi :
- Les **divergences** RSI/prix (quand le prix monte mais que le RSI baisse → signe de faiblesse)
- Les **niveaux de support/résistance** et les retracements **Fibonacci**
- Le **volume** : un signal est plus fort quand le volume est élevé

### 🟢 Couche 2 — Machine Learning

Le bot utilise deux modèles de Machine Learning entraînés sur l'historique du prix :

1. **RandomForest Regressor** — forêt d'arbres de décision, bon pour capter des motifs complexes
2. **Linear Regression** — modèle linéaire simple, bon pour les tendances claires

Les deux modèles prédisent le prix à **3, 7 et 14 jours**. Leurs prédictions sont comparées au prix actuel pour générer des signaux d'achat ou de vente.

> **⚠️ Limitation importante :** Les cryptomonnaies sont extrêmement volatiles et influencées par des événements imprévisibles (news, régulations, tweets...). Le Machine Learning a une **confiance limitée** sur ce type de marché. Les prédictions ML sont utilisées comme **indice supplémentaire**, pas comme vérité absolue. La confiance affichée (ex: 58%) reflète cette incertitude.

### 🟣 Couche 3 — IA narrative (Ollama)

Le bot peut interroger **Qwen 2.5 (modèle 3B)** via Ollama pour une analyse qualitative du marché.

L'IA reçoit un résumé complet de la situation :
- Prix actuel, tendance, volatilité
- Résultats de l'analyse technique
- Prédictions ML
- Scores et signaux

Elle produit un **raisonnement en langage naturel** qui peut détecter des patterns ou des contextes que les indicateurs seuls ne voient pas.

> **Utile pour :** Repérer des configurations de marché particulières, des situations de doute, ou confirmer (ou infirmer) un signal technique avec un « regard humain ».

---

## 4. Gestion de risque — la partie la plus importante 🛡️

### 4.1 Taille de position (Kelly Criterion / 2)

```
Taille = (Capital × Kelly%) / 2
```

Le Kelly Criterion calcule la taille idéale à investir en fonction de :
- La probabilité de gagner (estimée par le score global)
- Le ratio gain/pertes attendu

En divisant par 2, on garde une **marge de sécurité** importante.

### 4.2 Stop-loss adaptatif (2× ATR)

Le stop-loss n'est pas fixe : il s'adapte à la **volatilité du marché**.

```
Stop-loss = Prix d'entrée - (2 × ATR)
```

- Si la volatilité augmente → le stop s'élargit (évite de se faire sortir par un simple bruit)
- Si la volatilité baisse → le stop se resserre (protège mieux les gains)

### 4.3 Take-profit (Ratio Risk/Reward 1:2.5)

Pour chaque trade, l'objectif de gain est **2,5 fois** le risque pris.

```
Take-profit = Prix d'entrée + (2,5 × distance_stop_loss)
```

Exemple : si on risque 100€, on vise 250€ de gain.

### 4.4 Trailing stop après +3%

Une fois que le trade est gagnant de **3% ou plus**, le stop-loss devient **suiveur** (trailing) :

- Il remonte automatiquement quand le prix monte
- Si le prix redescend de 1,5% depuis son plus haut, le trade est fermé
- Ça permet de **laisser courir les gains** tout en protégeant les profits déjà acquis

### 4.5 Cooldown après 3 pertes 🔄

Si le bot enchaîne **3 trades perdants consécutifs**, il s'arrête automatiquement pendant **24 heures**.

Pourquoi ? Une série de pertes peut arriver sur un marché défavorable. Au lieu de forcer et d'aggraver la situation, le bot prend du recul et attend que les conditions redeviennent favorables.

### 4.6 Protection contre le drawdown 🛑

Si la perte totale du portefeuille atteint **-15%**, le bot **s'arrête complètement**.

C'est une protection vitale : mieux vaut préserver 85% du capital et revenir plus tard que de risquer de tout perdre dans un marché baissier violent.

---

## 5. Règles d'or du bot

| # | Règle | Explication |
|---|-------|-------------|
| 1 | **Ne pas trader si le score est trop bas** | Si le score global est inférieur à un seuil minimum, le bot ne prend aucun trade — même si un indicateur isolé est positif. |
| 2 | **Ne pas forcer les trades** | Le bot ne trade pas pour trader. Il attend patiemment les meilleures configurations. |
| 3 | **Laisser le marché venir à nous** | On n'achète pas parce qu'on « pense » que ça va monter. On attend que les données disent oui. |
| 4 | **Diversification** | Le bot analyse plusieurs cryptos et compare leurs scores. Il ne met jamais tous les œufs dans le même panier. |
| 5 | **Backtester avant d'utiliser** | Toute stratégie doit être testée sur l'historique avant d'être utilisée en réel. |

---

## 6. Backtest — comment interpréter les résultats 📊

Le bot peut simuler sa stratégie sur des données historiques. Voici ce qu'il faut regarder :

### Métriques principales

- **Rendement total** — combien le portefeuille aurait gagné/perdu (%) sur la période
- **Win rate** — pourcentage de trades gagnants (un bon win rate est > 50%, mais ce n'est pas le seul critère)
- **Ratio gain/pertes** — en moyenne, combien on gagne quand on gagne vs combien on perd quand on perd
- **Profit factor** — gains totaux / pertes totales (idéalement > 1,5 — signifie qu'on gagne 1,5× plus qu'on ne perd)
- **Drawdown maximal** — la plus grande perte du portefeuille en cours de route (idéalement < 15%)
- **Nombre de trades** — assez de trades pour que les statistiques aient un sens (minimum 50-100)

### Lecture du rapport

```
=== BACKTEST Hermes Trading Bot v3 ===
Capital initial:    10 000,00 $
Capital final:      12 450,00 $
Rendement:          +24,50%
Trades total:       84
Win rate:           58,3%
Profit factor:      1,72
Drawdown max:       -8,4%
```

Dans cet exemple :
- ✅ Rendement positif (+24,5%)
- ✅ Win rate correct (58,3%)
- ✅ On gagne 1,72× plus qu'on ne perd (bon)
- ✅ Drawdown max maîtrisé (-8,4%, bien sous les 15%)

### Pièges à éviter

- ❌ Ne pas sur-optimiser les paramètres sur les données passées (overfitting — ça marche sur l'historique mais pas en réel)
- ❌ Se fier uniquement au rendement : un rendement de +200% avec seulement 3 trades n'a aucun sens statistique
- ❌ Ignorer le drawdown : un rendement de +50% avec -40% de drawdown est trop risqué pour un compte réel

---

## 7. Limitations importantes à connaître ⚠️

1. **Marché imprévisible** — Les cryptomonnaies sont parmi les actifs les plus volatils au monde. Personne, ni aucun algorithme, ne peut prédire les mouvements avec certitude.

2. **Machine Learning limité** — Le ML performe mieux sur des marchés avec des cycles prévisibles. Les cryptos sont souvent mues par l'actualité, les réseaux sociaux, et des événements impossibles à modéliser avec des données historiques seules.

3. **Pas de sentiment en temps réel** — Le bot n'analyse pas Twitter/X, les news, ou les annonces réglementaires en direct. Ces facteurs peuvent provoquer des mouvements brutaux que le bot ne verra pas venir.

4. **API CoinGecko gratuite** — Le bot utilise l'API publique gratuite de CoinGecko, qui a des limites de taux. En période de forte volatilité, les données peuvent être légèrement décalées.

5. **Outil d'aide, pas une solution magique** — Hermes Trading Bot est un **assistant d'analyse**, pas une machine à imprimer de l'argent. Il aide à prendre de meilleures décisions, mais ne remplace pas la vigilance et le bon sens.

---

## 8. En résumé

```
                        HERMES TRADING BOT v3
┌─────────────────────────────────────────────────────┐
│  📊 Analyse Technique  →  8 indicateurs + divergence  │
│  🤖 Machine Learning   →  RandomForest + Régression   │
│  🧠 IA Narrative       →  Qwen 2.5 (raisonnement)     │
│                                                       │
│         ↓  Score unique combiné ↓                     │
│                                                       │
│  🛡️ GESTION DE RISQUE (la priorité)                  │
│  ├─ Kelly / 2 → taille de position prudente           │
│  ├─ Stop-loss ATR → s'adapte à la volatilité          │
│  ├─ Take-profit 1:2,5 → ratio gain/risque sain        │
│  ├─ Trailing stop → protège les gains                 │
│  ├─ Cooldown 24h → pause après 3 pertes               │
│  └─ Drawdown max 15% → arrêt d'urgence                │
└─────────────────────────────────────────────────────┘
```

### La philosophie en une phrase

> **Analyser avec rigueur, trader avec prudence, et survivre pour trader un autre jour.** 🎯
