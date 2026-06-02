# Hermes Trading Bot

Analyse intelligente du marche crypto avec IA, Machine Learning et gestion de risque prudente.

→ **Site web :** https://symbalyx.github.io/hermes-trading-bot/
→ **Strategie :** STRATEGIE.md

## Fonctionnalites

- **Analyse technique avancee** — RSI, MACD, Bollinger Bands, ADX, Stochastique, MFI, Heikin Ashi, ATR, Fibonacci, divergences RSI/prix
- **Machine Learning** — RandomForest + LinearRegression pour predictions de tendance
- **IA narrative** — analyse raisonnee du marche via Ollama (qwen2.5:3b)
- **Risk Management** — Kelly Criterion, position sizing, stop-loss ATR, trailing stop, drawdown protection, cooldown
- **Backtesting** — validation historique de la strategie
- **Portfolio simulation** — simulation de portefeuille avec P&L
- **Rapports HTML/JSON** — tableau complet avec tous les indicateurs

## Usage

```bash
# Analyse rapide
python3 bot.py

# Analyse avec IA
python3 bot.py --coin all --llm

# Simulation portefeuille (10 000 $)
python3 bot.py --portfolio 10000

# Backtest strategie
python3 bot.py --backtest

# Rapport HTML
python3 bot.py --coin bitcoin,ethereum,solana --html --save --llm
```

## Dependances

```bash
pip install requests pandas numpy scikit-learn
```

## GitHub Pages

Le site explicatif est accessible sur :
https://symbalyx.github.io/hermes-trading-bot/

## Auteur

Pour Maxime Vaslin — Bot cree par Hermes Agent.
