#!/usr/bin/env python3
"""
Hermes Trading Bot — analyse marche crypto et actions
Indicateurs : RSI, MACD, Moyennes Mobiles, Volume
Donnees : CoinGecko API (gratuit, sans cle)

Usage:
  python3 bot.py                    # Analyse BTC, ETH par defaut
  python3 bot.py --coin bitcoin,cardano  # Analyse specifique
  python3 bot.py --coin all --save  # Tous les top 50, sauvegarde
"""
import json, time, sys, os, argparse
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    os.system("pip install requests -q --break-system-packages 2>/dev/null || pip3 install requests -q --break-system-packages 2>/dev/null")
    import requests

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
DATA_DIR.mkdir(exist_ok=True)

TOP_50 = [
    "bitcoin","ethereum","tether","ripple","cardano","solana","polkadot",
    "dogecoin","avalanche","chainlink","polygon","litecoin","uniswap",
    "stellar","monero","filecoin","vechain","theta","eos","aave","maker",
    "algorand","tezos","decentraland","the-sandbox","axie-infinity",
    "near","hedera","cosmos","internet-computer","aptos","sui","optimism",
    "arbitrum","pepe","injective","fetch-ai","render","immutable",
    "sei","celestia","kaspa","flow","gala","fantom","kucoin-token",
    "compound","curve-dao-token","yearn-finance","zcash"
]

def fetch_prices(coin_ids="bitcoin,ethereum", days=30):
    """Recupere donnees historique prix + volume"""
    url = f"{COINGECKO_BASE}/coins/{coin_ids}/market_chart?vs_currency=usd&days={days}"
    r = requests.get(url, timeout=30)
    if r.status_code == 429:
        print("Rate limited, waiting 60s...")
        time.sleep(60)
        return fetch_prices(coin_ids, days)
    r.raise_for_status()
    return r.json()

def fetch_coin_info(coin_id):
    """Recupere infos supplementaires (market cap, rank, etc.)"""
    url = f"{COINGECKO_BASE}/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
    r = requests.get(url, timeout=15)
    if r.status_code == 429:
        time.sleep(30)
        return fetch_coin_info(coin_id)
    r.raise_for_status()
    return r.json()

# --- INDICATEURS TECHNIQUES ---

def calc_rsi(prices, period=14):
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    gains, losses = 0, 0
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i-1]
        if diff >= 0:
            gains += diff
        else:
            losses += abs(diff)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calc_moving_averages(prices):
    """Moyennes mobiles SMA 20, 50, 200, EMA12, EMA26"""
    if len(prices) < 200:
        return {"sma_20": None, "sma_50": None, "sma_200": None}
    sma20 = sum(prices[-20:]) / 20
    sma50 = sum(prices[-50:]) / 50
    sma200 = sum(prices[-200:]) / 200 if len(prices) >= 200 else None
    return {"sma_20": round(sma20, 2), "sma_50": round(sma50, 2), "sma_200": round(sma200, 2) if sma200 else None}

def calc_macd(prices):
    """MACD (12,26) + Signal (9) + Histogramme"""
    if len(prices) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd_line = ema12[-1] - ema26[-1]
    signal = calc_ema_values(macd_line, 9) if len(prices) > 35 else None
    hist = macd_line - signal if signal else None
    return {"macd": round(macd_line, 2), "signal": round(signal, 2) if signal else None, "histogram": round(hist, 2) if hist else None}

def calc_ema(prices, period):
    """Exponential Moving Average"""
    multiplier = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

def calc_ema_values(values, period):
    """Calcul EMA sur une serie de valeurs"""
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = values[-period]
    for v in values[-(period-1):]:
        ema = (v - ema) * multiplier + ema
    return ema

def calc_support_resistance(prices):
    """Niveaux support/resistance simples (min/mai 50 periodes)"""
    recent = prices[-50:] if len(prices) >= 50 else prices
    support = min(recent)
    resistance = max(recent)
    return {"support": round(support, 2), "resistance": round(resistance, 2)}

# --- SIGNAL ---

def generate_signal(rsi, mas, macd, current_price):
    """Genere signal trading (ACHAT/VENTE/NEUTRE) avec score"""
    score = 0
    reasons = []
    
    if rsi is not None:
        if rsi < 30:
            score += 2; reasons.append(f"RSI survente ({rsi})")
        elif rsi > 70:
            score -= 2; reasons.append(f"RSI surachat ({rsi})")
        elif 30 <= rsi <= 45:
            score += 1; reasons.append(f"RSI neutre-bas ({rsi})")
        elif 55 <= rsi <= 70:
            score -= 1; reasons.append(f"RSI neutre-haut ({rsi})")
    
    if mas.get("sma_20") and current_price:
        if current_price > mas["sma_20"]:
            score += 1; reasons.append(f"Prix > SMA20 ({mas['sma_20']}$)")
        else:
            score -= 1; reasons.append(f"Prix < SMA20 ({mas['sma_20']}$)")
        if mas.get("sma_50") and current_price > mas["sma_50"]:
            score += 1; reasons.append(f"Prix > SMA50")
        if mas.get("sma_200") and current_price > mas["sma_200"]:
            score += 1; reasons.append(f"Prix > SMA200 (tendance haussiere LT)")
        elif mas.get("sma_200") and current_price < mas["sma_200"]:
            score -= 1; reasons.append(f"Prix < SMA200 (tendance baissiere LT)")
    
    if macd.get("macd") and macd.get("signal"):
        if macd["macd"] > macd["signal"]:
            score += 1; reasons.append("MACD > Signal (hausse MT)")
        else:
            score -= 1; reasons.append("MACD < Signal (baisse MT)")
        if macd.get("histogram") and macd["histogram"] > 0:
            if macd["histogram"] > macd.get("prev_histogram", 0):
                score += 1
    
    if score >= 3:
        signal = "ACHAT"
    elif score <= -3:
        signal = "VENTE"
    else:
        signal = "NEUTRE"
    
    return {
        "signal": signal,
        "score": score,
        "max_score": 7,
        "reasons": reasons,
        "niveau": "FORT" if abs(score) >= 5 else "MOYEN" if abs(score) >= 3 else "FAIBLE"
    }

# --- RAPPORT ---

def analyze_coin(coin_id, display_name=None):
    """Analyse complete d'une crypto"""
    name = display_name or coin_id.capitalize()
    print(f"\n{'='*50}")
    print(f"  {name.upper()} — analyse {datetime.now().strftime('%H:%M %d/%m/%Y')}")
    print(f"{'='*50}")
    
    try:
        prices_raw = fetch_prices(coin_id, days=60)
    except Exception as e:
        return {"error": str(e), "coin": coin_id}
    
    prices = [p[1] for p in prices_raw.get("prices", [])]
    volumes = [v[1] for v in prices_raw.get("total_volumes", [])]
    
    if not prices:
        return {"error": "Pas de donnees", "coin": coin_id}
    
    current_price = prices[-1]
    price_24h_ago = prices[-2] if len(prices) > 1 else current_price
    change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    vol_moyen = sum(volumes[-7:]) / 7 if len(volumes) >= 7 else sum(volumes) / len(volumes)
    vol_recent = volumes[-1] if volumes else 0
    vol_ratio = vol_recent / vol_moyen if vol_moyen > 0 else 0
    
    rsi = calc_rsi(prices)
    mas = calc_moving_averages(prices)
    macd = calc_macd(prices)
    sr = calc_support_resistance(prices)
    signal_data = generate_signal(rsi, mas, macd, current_price)
    
    # Prix en format lisible
    def fmt_price(p):
        if p < 1: return f"${p:.6f}"
        if p < 100: return f"${p:.4f}"
        if p < 10000: return f"${p:.2f}"
        return f"${p:,.0f}"
    
    print(f"  Prix:     {fmt_price(current_price)}")
    print(f"  24h:      {change_24h:+.2f}%")
    print(f"  Volume:   {vol_ratio:.1f}x moyenne recente")
    if rsi: print(f"  RSI(14):  {rsi}")
    if mas["sma_20"]: print(f"  SMA20:    {fmt_price(mas['sma_20'])}  SMA50: {fmt_price(mas['sma_50'])}")
    if macd["macd"]: print(f"  MACD:     {macd['macd']}  Signal: {macd['signal']}  Hist: {macd['histogram']}")
    print(f"  Support:  {fmt_price(sr['support'])}  Resistance: {fmt_price(sr['resistance'])}")
    
    sig = signal_data["signal"]
    sig_color = "🟢" if sig == "ACHAT" else "🔴" if sig == "VENTE" else "⚪"
    print(f"  SIGNAL:   {sig_color} {sig} ({signal_data['niveau']}) — score {signal_data['score']}/{signal_data['max_score']}")
    for reason in signal_data["reasons"]:
        print(f"    → {reason}")
    
    result = {
        "coin": coin_id,
        "name": name,
        "price": current_price,
        "change_24h": round(change_24h, 2),
        "rsi": rsi,
        "sma_20": mas["sma_20"],
        "sma_50": mas["sma_50"],
        "sma_200": mas["sma_200"],
        "macd": macd["macd"],
        "signal_value": macd["signal"],
        "histogram": macd["histogram"],
        "support": sr["support"],
        "resistance": sr["resistance"],
        "vol_ratio": round(vol_ratio, 2),
        "trading_signal": signal_data["signal"],
        "signal_score": signal_data["score"],
        "signal_niveau": signal_data["niveau"],
        "reasons": signal_data["reasons"],
        "timestamp": datetime.now().isoformat()
    }
    return result

def save_report(results, filename=None):
    """Sauvegarde le rapport dans un fichier JSON"""
    if not filename:
        filename = DATA_DIR / f"analyse_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "market_summary": {
            "total_analyzed": len([r for r in results if "error" not in r]),
            "buy_signals": len([r for r in results if r.get("trading_signal") == "ACHAT"]),
            "sell_signals": len([r for r in results if r.get("trading_signal") == "VENTE"]),
            "neutral": len([r for r in results if r.get("trading_signal") == "NEUTRE"]),
            "best_score": max([r.get("signal_score", -10) for r in results], default=0),
            "worst_score": min([r.get("signal_score", 10) for r in results], default=0)
        },
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📁 Rapport sauvegarde: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Bot")
    parser.add_argument("--coin", default="bitcoin,ethereum", 
                        help="Coin(s) a analyser (separer par des virgules, ou 'all')")
    parser.add_argument("--days", type=int, default=60, help="Periode d'analyse en jours")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le rapport")
    parser.add_argument("--loop", type=int, help="Analyser en boucle toutes les N minutes")
    args = parser.parse_args()
    
    if args.coin == "all":
        coins = TOP_50[:20]  # Top 20 pour limiter les appels API
    else:
        coins = [c.strip() for c in args.coin.split(",")]
    
    while True:
        print(f"\n🚀 HERMES TRADING BOT — analyse {len(coins)} actifs")
        print(f"   Periode: {args.days} jours | Donnees: CoinGecko")
        
        results = []
        for coin in coins:
            name = coin.replace("-", " ").title()[:20]
            try:
                result = analyze_coin(coin, name)
                results.append(result)
                time.sleep(1.5)  # Rate limit CoinGecko
            except Exception as e:
                print(f"  ❌ {coin}: {e}")
                results.append({"coin": coin, "error": str(e)})
        
        # Top picks
        valid = [r for r in results if "error" not in r]
        if valid:
            buy = sorted([r for r in valid if r.get("trading_signal") == "ACHAT"], 
                        key=lambda x: x.get("signal_score", 0), reverse=True)
            sell = sorted([r for r in valid if r.get("trading_signal") == "VENTE"], 
                         key=lambda x: x.get("signal_score", 0))
            
            print(f"\n{'='*50}")
            print(f"  📊 TOP PICKS DU MARCHE")
            print(f"{'='*50}")
            if buy:
                print(f"\n  🟢 SIGNAL ACHAT ({len(buy)})")
                for b in buy[:5]:
                    print(f"     {b['coin']:15s} score: {b['signal_score']:+d}  RSI: {b.get('rsi','-'):>6}  ${b.get('price',0):,.2f}")
            if sell:
                print(f"\n  🔴 SIGNAL VENTE ({len(sell)})")
                for s in sell[:3]:
                    print(f"     {s['coin']:15s} score: {s['signal_score']:+d}  RSI: {s.get('rsi','-'):>6}  ${s.get('price',0):,.2f}")
        
        if args.save:
            save_report(results)
        
        if not args.loop:
            break
        
        print(f"\n⏳ Prochaine analyse dans {args.loop} min...")
        time.sleep(args.loop * 60)

if __name__ == "__main__":
    main()
