# Hermes Trading Bot 🤖📈

Bot de trading analyse marche cree par **Hermes Agent** (Nous Research).

Analyse les marches crypto via CoinGecko API (gratuit, sans cle API).

## Fonctionnalites

- **Indicateurs techniques** : RSI (14), SMA (20/50/200), MACD (12/26/9)
- **Signaux trading** : ACHAT / VENTE / NEUTRE avec score pondere
- **Support/Resistance** : niveaux automatiques sur 50 periodes
- **Volume** : ratio volume recent vs moyenne 7 jours
- **Top picks** : classement des meilleurs signaux du marche

## Usage

```bash
# Analyser Bitcoin et Ethereum
python3 bot.py

# Analyser des coins specifiques
python3 bot.py --coin bitcoin,cardano,solana

# Analyser le top 20
python3 bot.py --coin all

# Sauvegarder le rapport
python3 bot.py --coin bitcoin,ethereum --save

# Analyse en boucle (toutes les 30 min)
python3 bot.py --coin bitcoin --loop 30
```

## Sortie

```
==================================================
  BITCOIN — analyse 14:30 02/06/2026
==================================================
  Prix:     $68,452
  24h:      +2.34%
  Volume:   1.2x moyenne recente
  RSI(14):  58.4
  SMA20:    $67,890  SMA50: $65,432
  MACD:     245  Signal: 198  Hist: 47
  Support:  $62,100  Resistance: $71,500
  SIGNAL:   ⚪ NEUTRE (FAIBLE) — score 1/7
    → RSI neutre (58.4)
    → Prix > SMA20 ($67,890$)
```

## Dependances

- Python 3.8+
- requests
