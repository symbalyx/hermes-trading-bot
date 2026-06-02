#!/usr/bin/env python3
"""
Exchange Connector Module — Hermes Trading Bot v4
=================================================

Stratégie de conception (Strategy Pattern):
  ExchangeConnector       → classe abstraite (interface)
  BinanceConnector        → vraie API Binance REST
  PaperTradingConnector   → simulation papier (défaut)

Utilisation:
    from exchange import get_connector
    c = get_connector(mode="paper")
    print(c.get_balance())
    c.market_buy("BTCUSDT", 1.0)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ─── Logger partagé (reprend celui de bot.py si dispo) ──────────
log = logging.getLogger("hermes.exchange")
if not log.handlers:
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("  [EXCHANGE] %(message)s"))
    log.addHandler(ch)

# ─── Chemins — compatibles avec bot.py ──────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
API_KEYS_PATH = DATA_DIR / "api_keys.json"


# ─── Data models ─────────────────────────────────────────────────

@dataclass
class Order:
    """Représentation unifiée d'un ordre."""
    order_id: str
    symbol: str
    side: str                  # "BUY" | "SELL"
    type: str                  # "MARKET" | "LIMIT"
    quantity: float
    price: float = 0.0
    status: str = "NEW"        # "NEW" | "FILLED" | "PARTIALLY_FILLED" | "CANCELED"
    executed_qty: float = 0.0
    cummulative_quote_qty: float = 0.0
    created_at: str = ""       # ISO timestamp


@dataclass
class Balance:
    """Porte-monnaie du compte."""
    asset: str
    free: float = 0.0
    locked: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.locked


# ─── Factory / helpers ───────────────────────────────────────────

def load_api_keys() -> dict:
    """Charge les clés API depuis data/api_keys.json.

    Structure attendue:
        {
            "exchange": "binance",
            "binance": {
                "api_key": "...",
                "api_secret": "..."
            }
        }

    Si le fichier n'existe pas, retourne un dict vide.
    """
    if not API_KEYS_PATH.exists():
        log.warning(f"Aucun fichier de clés API trouvé: {API_KEYS_PATH}")
        log.warning("  Créez data/api_keys.json avec le format requis.")
        return {}

    try:
        with open(API_KEYS_PATH) as f:
            keys = json.load(f)
        # Validation minimale
        if not isinstance(keys, dict):
            raise ValueError("api_keys.json doit être un dictionnaire")
        return keys
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Erreur de lecture api_keys.json: {e}")
        return {}


def save_api_keys_template():
    """Crée un fichier api_keys.json d'exemple si absent."""
    if not API_KEYS_PATH.exists():
        template = {
            "exchange": "binance",
            "binance": {
                "api_key": "VOTRE_CLE_API",
                "api_secret": "VOTRE_CLE_SECRETE"
            },
            "binance_testnet": {
                "api_key": "VOTRE_CLE_TESTNET",
                "api_secret": "VOTRE_CLE_SECRETE_TESTNET"
            }
        }
        DATA_DIR.mkdir(exist_ok=True)
        with open(API_KEYS_PATH, "w") as f:
            json.dump(template, f, indent=2)
        log.info(f"Fichier template créé: {API_KEYS_PATH}")
        log.info("  Éditez-le avec vos vraies clés Binance avant d'utiliser --live")
    else:
        log.info(f"api_keys.json existe déjà: {API_KEYS_PATH}")


# ─── Abstract Base ───────────────────────────────────────────────

class ExchangeConnector(ABC):
    """Interface commune pour tous les connecteurs d'échange.

    Toutes les méthodes lèvent NotImplementedError — les sous-classes
    doivent les implémenter.
    """

    @abstractmethod
    def get_balance(self) -> dict[str, Balance]:
        """Retourne le solde complet du compte {asset: Balance}."""
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Prix actuel d'un symbole."""
        ...

    @abstractmethod
    def market_buy(self, symbol: str, quantity: float) -> Order:
        """Achat au marché. quantity = montant de l'asset de base."""
        ...

    @abstractmethod
    def market_sell(self, symbol: str, quantity: float) -> Order:
        """Vente au marché."""
        ...

    @abstractmethod
    def limit_buy(self, symbol: str, quantity: float, price: float) -> Order:
        """Achat limité."""
        ...

    @abstractmethod
    def limit_sell(self, symbol: str, quantity: float, price: float) -> Order:
        """Vente limitée."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Statut d'un ordre par son ID."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre ouvert. Retourne True si succès."""
        ...

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Liste des ordres ouverts."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Balance:
        """Position actuelle pour un symbole donné (balance de l'asset)."""
        ...

    def get_info(self) -> dict:
        """Méta-info sur le connecteur."""
        return {
            "name": self.__class__.__name__,
            "type": "real" if "Paper" not in self.__class__.__name__ else "paper",
        }


# ─── Binance Connector (API REST native) ─────────────────────────

class BinanceConnector(ExchangeConnector):
    """Connecteur Binance via API REST native (sans dépendance python-binance)."""

    BASE_URL = "https://api.binance.com"
    BASE_URL_TESTNET = "https://testnet.binance.vision"

    def __init__(self, api_key: str = "", api_secret: str = "",
                 testnet: bool = False, recv_window: int = 5000):
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL
        self._session = requests.Session()
        self._session.headers.update({
            "X-MBX-APIKEY": self.api_key,
            "User-Agent": "HermesTradingBot/v4",
            "Content-Type": "application/json",
        })
        log.info(f"BinanceConnector initialisé {'(TESTNET)' if testnet else '(REAL)'}")

    # ─── Signatures HMAC-SHA256 ──────────────────────────────────

    def _sign(self, params: dict) -> dict:
        """Ajoute timestamp + signature HMAC aux paramètres."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, endpoint: str,
                 signed: bool = False, params: dict = None) -> Any:
        """Requête HTTP vers l'API Binance.

        Retourne le JSON décodé, ou lève une exception sur erreur.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        headers = {}
        # Les requêtes signées utilisent X-MBX-APIKEY dans le header de session
        if signed:
            params = self._sign(params)
            # La session a déjà X-MBX-APIKEY dans ses headers

        for attempt in range(3):
            try:
                if method == "GET":
                    r = self._session.get(url, params=params, timeout=15)
                elif method == "POST":
                    r = self._session.post(url, params=params, timeout=15)
                elif method == "DELETE":
                    r = self._session.delete(url, params=params, timeout=15)
                else:
                    raise ValueError(f"Méthode non supportée: {method}")

                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log.warning(f"Binance rate-limit, pause {wait}s")
                    time.sleep(wait)
                    continue

                if r.status_code in (401, 403):
                    log.error(f"Binance authentification échouée ({r.status_code}): {r.text}")
                    raise PermissionError(
                        f"Échec d'authentification Binance. Vérifiez vos clés API. "
                        f"Réponse: {r.text[:200]}"
                    )

                r.raise_for_status()
                return r.json()

            except requests.Timeout:
                log.warning(f"Timeout Binance {endpoint}, tentative {attempt+1}/3")
                time.sleep(5 * (attempt + 1))
            except requests.ConnectionError:
                log.warning(f"Connection error Binance {endpoint}, tentative {attempt+1}/3")
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Échec après 3 tentatives: {endpoint}")

    def _parse_order(self, raw: dict) -> Order:
        """Convertit une réponse Binance en Order unifié."""
        # Gestion des noms de champs (nouvelle API Binance utilise des camelCase)
        return Order(
            order_id=str(raw.get("orderId", raw.get("order_id", ""))),
            symbol=raw.get("symbol", ""),
            side=raw.get("side", ""),
            type=raw.get("type", ""),
            quantity=float(raw.get("origQty", raw.get("quantity", 0))),
            price=float(raw.get("price", 0)),
            status=raw.get("status", "NEW"),
            executed_qty=float(raw.get("executedQty", raw.get("executed_qty", 0))),
            cummulative_quote_qty=float(
                raw.get("cummulativeQuoteQty", raw.get("cummulative_quote_qty", 0))
            ),
            created_at=datetime.fromtimestamp(
                raw.get("time", raw.get("transactTime", time.time() * 1000)) / 1000
            ).isoformat(),
        )

    # ─── Implémentation de l'interface ──────────────────────────

    def get_balance(self) -> dict[str, Balance]:
        data = self._request("GET", "/api/v3/account", signed=True)
        balances: dict[str, Balance] = {}
        for bal in data.get("balances", []):
            free = float(bal["free"])
            locked = float(bal["locked"])
            if free > 0 or locked > 0:
                balances[bal["asset"]] = Balance(
                    asset=bal["asset"],
                    free=free,
                    locked=locked,
                )
        return balances

    def get_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol})
        return float(data["price"])

    def market_buy(self, symbol: str, quantity: float) -> Order:
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": quantity,
        }
        raw = self._request("POST", "/api/v3/order", signed=True, params=params)
        return self._parse_order(raw)

    def market_sell(self, symbol: str, quantity: float) -> Order:
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": quantity,
        }
        raw = self._request("POST", "/api/v3/order", signed=True, params=params)
        return self._parse_order(raw)

    def limit_buy(self, symbol: str, quantity: float, price: float) -> Order:
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
        }
        raw = self._request("POST", "/api/v3/order", signed=True, params=params)
        return self._parse_order(raw)

    def limit_sell(self, symbol: str, quantity: float, price: float) -> Order:
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
        }
        raw = self._request("POST", "/api/v3/order", signed=True, params=params)
        return self._parse_order(raw)

    def get_order_status(self, order_id: str) -> Optional[Order]:
        # L'API nécessite le symbole — on essaie de le retrouver depuis les ordres ouverts
        # Alternative: garder une map interne order_id -> symbol (simplifié ici)
        try:
            # On récupère tous les ordres ouverts et on cherche
            # Si on ne trouve pas, on peut tenter order.book
            # Pour simplifier, on utilise l'API oco ou l'order lookup avec symbol
            data = self._request("GET", "/api/v3/allOrders",
                                 signed=True,
                                 params={"limit": 5})
            for raw in data:
                if str(raw.get("orderId", "")) == order_id:
                    return self._parse_order(raw)
            return None
        except Exception:
            return None

    def cancel_order(self, order_id: str) -> bool:
        # Nécessite le symbol — tentative via openOrders puis annulation
        try:
            open_orders = self.get_open_orders()
            for o in open_orders:
                if o.order_id == order_id:
                    self._request("DELETE", "/api/v3/order",
                                  signed=True,
                                  params={
                                      "symbol": o.symbol,
                                      "orderId": int(order_id),
                                  })
                    return True
            log.warning(f"Ordre {order_id} introuvable dans les ordres ouverts")
            return False
        except Exception as e:
            log.error(f"Erreur annulation ordre {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/api/v3/openOrders", signed=True, params=params)
        return [self._parse_order(raw) for raw in data]

    def get_position(self, symbol: str) -> Balance:
        """Pour Binance spot, la position est juste le solde libre de l'asset de base."""
        # Extraire l'asset de base du symbol (ex: BTCUSDT -> BTC)
        base_asset = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        # Gérer les paires stables
        if base_asset == symbol:
            # Tenter de trouver l'asset de base par convention
            for suffix in ["USDT", "BUSD", "USDC", "DAI", "FDUSD", "TUSD"]:
                if symbol.endswith(suffix):
                    base_asset = symbol[: -len(suffix)]
                    break
        balances = self.get_balance()
        return balances.get(base_asset, Balance(asset=base_asset))


# ─── Paper Trading Connector (simulation) ─────────────────────────

class PaperTradingConnector(ExchangeConnector):
    """Simulation de trading — pas d'argent réel.

    Maintient un portefeuille virtuel et un historique des trades.
    Les prix sont récupérés depuis une source publique (CoinGecko).

    Mode par défaut (sécurité) — activé automatiquement si --paper ou aucun flag.
    """

    def __init__(self, initial_balance_usd: float = 10000.0,
                 price_source: str = "coingecko"):
        self.initial_balance = initial_balance_usd
        self.cash: float = initial_balance_usd  # USD disponibles
        self.holdings: dict[str, float] = {}     # asset -> quantity
        self.orders: dict[str, Order] = {}       # order_id -> Order
        self.trade_history: list[Order] = []
        self.fees_pct = 0.001  # 0.1% frais simulés
        self.price_source = price_source
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self._cache_ttl = 30  # secondes

        log.info(f"PaperTradingConnector — ${initial_balance_usd:,.0f} de capital virtuel")

    # ─── Prix simulés (via CoinGecko) ────────────────────────────

    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convertit un symbol Binance (BTCUSDT) en CoinGecko ID (bitcoin)."""
        mapping = {
            "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
            "SOLUSDT": "solana", "ADAUSDT": "cardano",
            "XRPUSDT": "ripple", "DOTUSDT": "polkadot",
            "DOGEUSDT": "dogecoin", "AVAXUSDT": "avalanche-2",
            "LINKUSDT": "chainlink", "MATICUSDT": "matic-network",
            "UNIUSDT": "uniswap", "ATOMUSDT": "cosmos",
            "LTCUSDT": "litecoin", "BCHUSDT": "bitcoin-cash",
            "XLMUSDT": "stellar", "FILUSDT": "filecoin",
            "NEARUSDT": "near", "APTUSDT": "aptos",
            "ARBUSDT": "arbitrum", "OPUSDT": "optimism",
            "SUIUSDT": "sui", "PEPEUSDT": "pepe",
            "INJUSDT": "injective", "FETUSDT": "fetch-ai",
            "RNDRUSDT": "render-token", "IMXUSDT": "immutable-x",
            "SEIUSDT": "sei-network", "TIAUSDT": "celestia",
            "KASUSDT": "kaspa",
        }
        upper = symbol.upper()
        if upper in mapping:
            return mapping[upper]
        # Fallback: retirer les suffixes stables
        for s in ["USDT", "BUSD", "USDC", "DAI", "FDUSD", "TUSD"]:
            if upper.endswith(s):
                base = upper[: -len(s)]
                return base.lower()
        return upper.lower()

    def _fetch_price_coingecko(self, coin_id: str) -> float:
        """Récupère le prix actuel depuis CoinGecko."""
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                price = data.get(coin_id, {}).get("usd")
                if price:
                    return float(price)
            elif r.status_code == 429:
                log.warning("CoinGecko rate-limit (PaperTrading)")
                time.sleep(5)
        except Exception as e:
            log.warning(f"Erreur prix CoinGecko pour {coin_id}: {e}")
        return 0.0

    def get_price(self, symbol: str) -> float:
        """Prix actuel du symbole (avec cache)."""
        now = time.time()
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if now - ts < self._cache_ttl:
                return price

        coin_id = self._symbol_to_coingecko_id(symbol)
        price = self._fetch_price_coingecko(coin_id)

        if price > 0:
            self._price_cache[symbol] = (price, now)
        else:
            # Fallback: utiliser le dernier prix connu ou 0
            log.warning(f"Impossible d'obtenir le prix de {symbol} (PaperTrading)")
            # Garder le cache si disponible
            if symbol in self._price_cache:
                price = self._price_cache[symbol][0]

        return price

    def _generate_order_id(self) -> str:
        return f"paper_{uuid.uuid4().hex[:12]}"

    def _apply_fees(self, value: float) -> float:
        return value * (1 - self.fees_pct)

    # ─── Implémentation de l'interface ──────────────────────────

    def get_balance(self) -> dict[str, Balance]:
        balances: dict[str, Balance] = {
            "USD": Balance(asset="USD", free=self.cash, locked=0.0),
        }
        for asset, qty in self.holdings.items():
            if qty > 0:
                balances[asset] = Balance(asset=asset, free=qty, locked=0.0)
        return balances

    def market_buy(self, symbol: str, quantity: float) -> Order:
        price = self.get_price(symbol)
        if price <= 0:
            raise ValueError(f"Impossible d'obtenir le prix pour {symbol}")

        total_cost = quantity * price
        total_cost_with_fees = total_cost / (1 - self.fees_pct)  # Les frais sont déduits de l'asset

        if total_cost_with_fees > self.cash:
            # Acheter avec le maximum disponible
            max_qty = (self.cash * (1 - self.fees_pct)) / price
            if max_qty <= 0:
                raise ValueError(f"Fonds insuffisants: ${self.cash:.2f} < ${total_cost:.2f}")
            quantity = max_qty
            total_cost = quantity * price
            total_cost_with_fees = total_cost / (1 - self.fees_pct)

        # Exécution
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        self.cash -= total_cost_with_fees
        received_qty = self._apply_fees(quantity)
        self.holdings[base_asset] = self.holdings.get(base_asset, 0) + received_qty

        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=quantity,
            price=price,
            status="FILLED",
            executed_qty=received_qty,
            cummulative_quote_qty=total_cost,
            created_at=datetime.now().isoformat(),
        )
        self.orders[order.order_id] = order
        self.trade_history.append(order)
        log.info(f"[PAPER] ACHAT {quantity:.6f} {symbol} @ ${price:.2f} | "
                 f"Coût ${total_cost:.2f} | Cash restant ${self.cash:.2f}")
        return order

    def market_sell(self, symbol: str, quantity: float) -> Order:
        price = self.get_price(symbol)
        if price <= 0:
            raise ValueError(f"Impossible d'obtenir le prix pour {symbol}")

        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        available = self.holdings.get(base_asset, 0)
        if available <= 0:
            raise ValueError(f"Aucune position {base_asset} à vendre")

        qty = min(quantity, available)
        total_revenue = qty * price
        fees = total_revenue * self.fees_pct
        net_revenue = total_revenue - fees

        # Exécution
        self.holdings[base_asset] = available - qty
        if self.holdings[base_asset] <= 1e-10:
            del self.holdings[base_asset]
        self.cash += net_revenue

        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=qty,
            price=price,
            status="FILLED",
            executed_qty=qty,
            cummulative_quote_qty=total_revenue,
            created_at=datetime.now().isoformat(),
        )
        self.orders[order.order_id] = order
        self.trade_history.append(order)
        log.info(f"[PAPER] VENTE {qty:.6f} {symbol} @ ${price:.2f} | "
                 f"Revenu ${net_revenue:.2f} | Cash ${self.cash:.2f}")
        return order

    def limit_buy(self, symbol: str, quantity: float, price: float) -> Order:
        """Place un ordre limité d'achat."""
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        total_cost = quantity * price

        if total_cost > self.cash:
            raise ValueError(
                f"Fonds insuffisants pour ordre limit: "
                f"${total_cost:.2f} > ${self.cash:.2f}"
            )

        # On bloque le cash
        self.cash -= total_cost

        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            quantity=quantity,
            price=price,
            status="NEW",
            executed_qty=0.0,
            cummulative_quote_qty=0.0,
            created_at=datetime.now().isoformat(),
        )
        self.orders[order.order_id] = order
        log.info(f"[PAPER] LIMIT ACHAT {quantity:.6f} {symbol} @ ${price:.2f} "
                 f"(en attente, ${total_cost:.2f} bloqué)")
        return order

    def limit_sell(self, symbol: str, quantity: float, price: float) -> Order:
        """Place un ordre limité de vente."""
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        available = self.holdings.get(base_asset, 0)

        if quantity > available:
            raise ValueError(
                f"Quantité insuffisante pour ordre limit: "
                f"{quantity:.6f} > {available:.6f}"
            )

        # On bloque la quantité
        self.holdings[base_asset] = available - quantity

        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side="SELL",
            type="LIMIT",
            quantity=quantity,
            price=price,
            status="NEW",
            executed_qty=0.0,
            cummulative_quote_qty=0.0,
            created_at=datetime.now().isoformat(),
        )
        self.orders[order.order_id] = order
        log.info(f"[PAPER] LIMIT VENTE {quantity:.6f} {symbol} @ ${price:.2f} "
                 f"(en attente, {quantity:.6f} bloqué)")
        return order

    def get_order_status(self, order_id: str) -> Optional[Order]:
        order = self.orders.get(order_id)
        if order and order.status in ("NEW", "PARTIALLY_FILLED"):
            # Vérifier si le prix du marché correspond
            current_price = self.get_price(order.symbol)
            if order.side == "BUY" and current_price <= order.price * 1.001:
                # Ordre limit exécuté (simulation simplifiée)
                self._execute_limit_order(order, current_price)
            elif order.side == "SELL" and current_price >= order.price * 0.999:
                self._execute_limit_order(order, current_price)
        return self.orders.get(order_id)

    def _execute_limit_order(self, order: Order, current_price: float):
        """Simule l'exécution d'un ordre limité."""
        if order.status != "NEW":
            return

        base_asset = order.symbol.replace("USDT", "").replace("BUSD", "")

        if order.side == "BUY":
            total_cost = order.quantity * order.price
            received_qty = self._apply_fees(order.quantity)
            self.holdings[base_asset] = self.holdings.get(base_asset, 0) + received_qty
            # Le cash a déjà été déduit lors du placement
            order.executed_qty = received_qty
            order.cummulative_quote_qty = total_cost
            log.info(f"[PAPER] LIMIT ACHAT EXECUTÉ {order.quantity:.6f} {order.symbol} "
                     f"@ ${order.price:.2f}")
        else:  # SELL
            total_revenue = order.quantity * order.price
            fees = total_revenue * self.fees_pct
            net_revenue = total_revenue - fees
            self.cash += net_revenue
            order.executed_qty = order.quantity
            order.cummulative_quote_qty = total_revenue
            log.info(f"[PAPER] LIMIT VENTE EXÉCUTÉE {order.quantity:.6f} {order.symbol} "
                     f"@ ${order.price:.2f}")

        order.status = "FILLED"
        self.trade_history.append(order)

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if not order:
            log.warning(f"Ordre {order_id} introuvable")
            return False
        if order.status != "NEW":
            log.warning(f"Impossible d'annuler l'ordre {order_id} (statut: {order.status})")
            return False

        # Restituer les fonds bloqués
        base_asset = order.symbol.replace("USDT", "").replace("BUSD", "")
        if order.side == "BUY":
            total_cost = order.quantity * order.price
            self.cash += total_cost
        else:  # SELL
            self.holdings[base_asset] = self.holdings.get(base_asset, 0) + order.quantity

        order.status = "CANCELED"
        log.info(f"[PAPER] ORDRE ANNULÉ {order_id}: {order.side} {order.quantity:.6f} {order.symbol}")
        return True

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        orders = [
            o for o in self.orders.values()
            if o.status in ("NEW", "PARTIALLY_FILLED")
        ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_position(self, symbol: str) -> Balance:
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        qty = self.holdings.get(base_asset, 0.0)
        return Balance(asset=base_asset, free=qty, locked=0.0)

    def get_portfolio_summary(self) -> dict:
        """Résumé complet du portefeuille papier."""
        total_value = self.cash
        positions = []
        for asset, qty in self.holdings.items():
            if qty <= 0:
                continue
            # Trouver un symbole pour le pricing
            symbol = f"{asset}USDT"
            price = self.get_price(symbol)
            value = qty * price
            total_value += value
            positions.append({
                "asset": asset,
                "quantity": round(qty, 6),
                "price": round(price, 2),
                "value": round(value, 2),
            })

        pnl = total_value - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            "initial_balance": round(self.initial_balance, 2),
            "cash": round(self.cash, 2),
            "holdings_value": round(total_value - self.cash, 2),
            "total_value": round(total_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "positions": positions,
            "open_orders": len([o for o in self.get_open_orders()]),
            "total_trades": len(self.trade_history),
        }

    def reset(self, new_balance: Optional[float] = None):
        """Réinitialise le portefeuille papier."""
        self.cash = new_balance if new_balance is not None else self.initial_balance
        self.holdings.clear()
        self.orders.clear()
        self.trade_history.clear()
        self._price_cache.clear()
        log.info(f"[PAPER] Portefeuille réinitialisé (${self.cash:,.2f})")


# ─── Factory ─────────────────────────────────────────────────────

def get_connector(mode: str = "paper",
                  initial_capital: float = 10000.0,
                  testnet: bool = False) -> ExchangeConnector:
    """Fabrique le connecteur approprié.

    Args:
        mode: "paper" (défaut) | "live" | "binance"
        initial_capital: capital initial pour le paper trading
        testnet: utiliser le testnet Binance (au lieu du mainnet)

    Retourne:
        Une instance de ExchangeConnector

    Lève:
        ValueError: si les clés API sont manquantes en mode live
    """
    mode = mode.lower().strip()

    if mode == "paper":
        return PaperTradingConnector(initial_balance_usd=initial_capital)

    if mode in ("live", "binance"):
        keys = load_api_keys()
        api_key = keys.get("binance", {}).get("api_key", "")
        api_secret = keys.get("binance", {}).get("api_secret", "")

        if testnet:
            # Essayer les clés testnet d'abord, puis les clés mainnet
            testnet_keys = keys.get("binance_testnet", {})
            api_key = testnet_keys.get("api_key", api_key)
            api_secret = testnet_keys.get("api_secret", api_secret)

        if not api_key or not api_secret:
            raise ValueError(
                "Clés API Binance manquantes. "
                "Créez data/api_keys.json avec vos clés ou utilisez --paper."
            )

        log.info("Connecteur Binance LIVE initialisé")
        return BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=testnet)

    raise ValueError(f"Mode inconnu: {mode}. Utilisez 'paper' ou 'live'.")


# ─── Exécution de signaux ────────────────────────────────────────

def execute_signals(results: list,
                    connector: ExchangeConnector,
                    risk_manager=None,
                    dry_run: bool = False,
                    max_positions: int = 5,
                    max_per_trade_usd: float = 500.0) -> list[dict]:
    """Exécute les signaux d'analyse via un connecteur d'échange.

    Args:
        results: liste d'objets CoinResult (ou ayant .coin_id, .signal, .price, etc.)
        connector: instance ExchangeConnector (paper ou live)
        risk_manager: optionnel, instance RiskManager pour le position sizing
        dry_run: si True, affiche seulement ce qui serait fait
        max_positions: nombre max de positions simultanées
        max_per_trade_usd: montant max par trade (ignoré si risk_manager fourni)

    Retourne:
        liste de dicts décrivant chaque trade exécuté (ou simulé)
    """
    executed: list[dict] = []

    # Récupérer les positions ouvertes
    open_orders = connector.get_open_orders()
    open_symbols = {o.symbol for o in open_orders}

    # Filtrer seulement les signaux forts
    signals = [r for r in results if r and r.signal in ("ACHAT", "VENTE")]

    if not signals:
        log.info("Aucun signal à exécuter")
        return executed

    # Trier par score absolu (force du signal)
    signals.sort(key=lambda r: abs(r.normalized_score), reverse=True)

    # Limiter le nombre de positions
    if len(open_symbols) >= max_positions:
        log.warning(f"Nombre max de positions atteint ({max_positions}) — "
                     f"aucun nouveau trade")
        return executed

    available_slots = max_positions - len(open_symbols)

    for i, result in enumerate(signals):
        if i >= available_slots:
            log.info(f"Slots épuisés ({available_slots} dispo), arrêt")
            break

        # Convertir le coin_id en symbole Binance
        symbol = coin_id_to_symbol(result.coin_id)

        if symbol in open_symbols:
            log.info(f"Position déjà ouverte sur {symbol}, ignoré")
            continue

        # Déterminer la quantité
        price = result.price
        if price <= 0:
            log.warning(f"Prix invalide pour {symbol}, ignoré")
            continue

        # Position sizing
        if risk_manager and hasattr(result, "atr_pct") and result.atr_pct:
            sl_info = risk_manager.stop_loss(price, price * (result.atr_pct / 100), mult=2.0)
            pos_info = risk_manager.position_size(price, sl_info["stop_pct"])
            quantity = pos_info["quantity"]
            risk_pct = pos_info["risk_pct"]
        else:
            quantity = max_per_trade_usd / price
            risk_pct = None

        # Exécution
        try:
            if result.signal == "ACHAT":
                log.info(f"Signal ACHAT: {symbol} — ${price:.2f} x {quantity:.6f}")
                if not dry_run:
                    order = connector.market_buy(symbol, quantity)
                else:
                    order = Order(
                        order_id="DRY_RUN",
                        symbol=symbol, side="BUY", type="MARKET",
                        quantity=quantity, price=price,
                        status="FILLED" if not dry_run else "NEW",
                        executed_qty=quantity,
                        cummulative_quote_qty=quantity * price,
                        created_at=datetime.now().isoformat(),
                    )

                executed.append({
                    "action": "BUY",
                    "symbol": symbol,
                    "coin": result.coin_id,
                    "price": price,
                    "quantity": quantity,
                    "value": quantity * price,
                    "risk_pct": risk_pct,
                    "order_id": order.order_id,
                    "status": order.status,
                    "dry_run": dry_run,
                })

            elif result.signal == "VENTE":
                # Vérifier qu'on a la position
                position = connector.get_position(symbol)
                if position and position.free > 0:
                    qty_sell = position.free
                    log.info(f"Signal VENTE: {symbol} — ${price:.2f} x {qty_sell:.6f}")
                    if not dry_run:
                        order = connector.market_sell(symbol, qty_sell)
                    else:
                        order = Order(
                            order_id="DRY_RUN",
                            symbol=symbol, side="SELL", type="MARKET",
                            quantity=qty_sell, price=price,
                            status="FILLED" if not dry_run else "NEW",
                            executed_qty=qty_sell,
                            cummulative_quote_qty=qty_sell * price,
                            created_at=datetime.now().isoformat(),
                        )

                    executed.append({
                        "action": "SELL",
                        "symbol": symbol,
                        "coin": result.coin_id,
                        "price": price,
                        "quantity": qty_sell,
                        "value": qty_sell * price,
                        "order_id": order.order_id,
                        "status": order.status,
                        "dry_run": dry_run,
                    })
                else:
                    log.info(f"Pas de position {symbol} à vendre, ignoré")

        except Exception as e:
            log.error(f"Échec exécution {result.signal} {symbol}: {e}")
            executed.append({
                "action": result.signal,
                "symbol": symbol,
                "coin": result.coin_id,
                "error": str(e),
                "dry_run": dry_run,
            })

    # Résumé
    if executed:
        buys = sum(1 for e in executed if e.get("action") == "BUY" and "error" not in e)
        sells = sum(1 for e in executed if e.get("action") == "SELL" and "error" not in e)
        total_value = sum(e.get("value", 0) for e in executed if "value" in e)
        log.info(f"Exécution terminée: {buys} achat(s), {sells} vente(s), "
                 f"volume total ${total_value:.2f}")

    return executed


# ─── Utilitaires de conversion ────────────────────────────────────

def coin_id_to_symbol(coin_id: str) -> str:
    """Convertit un CoinGecko ID en symbol Binance (ex: 'bitcoin' -> 'BTCUSDT')."""
    reverse_mapping = {
        "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT",
        "solana": "SOLUSDT", "cardano": "ADAUSDT",
        "ripple": "XRPUSDT", "polkadot": "DOTUSDT",
        "dogecoin": "DOGEUSDT", "avalanche-2": "AVAXUSDT",
        "chainlink": "LINKUSDT", "matic-network": "MATICUSDT",
        "uniswap": "UNIUSDT", "cosmos": "ATOMUSDT",
        "litecoin": "LTCUSDT", "bitcoin-cash": "BCHUSDT",
        "stellar": "XLMUSDT", "filecoin": "FILUSDT",
        "near": "NEARUSDT", "aptos": "APTUSDT",
        "arbitrum": "ARBUSDT", "optimism": "OPUSDT",
        "sui": "SUIUSDT", "pepe": "PEPEUSDT",
        "injective": "INJUSDT", "fetch-ai": "FETUSDT",
        "render-token": "RNDRUSDT", "immutable-x": "IMXUSDT",
        "sei-network": "SEIUSDT", "celestia": "TIAUSDT",
        "kaspa": "KASUSDT",
        "litecoin": "LTCUSDT", "monero": "XMRUSDT",
        "eos": "EOSUSDT", "aave": "AAVEUSDT",
        "algorand": "ALGOUSDT", "tezos": "XTZUSDT",
        "hedera": "HBARUSDT", "internet-computer": "ICPUSDT",
    }
    coin_id = coin_id.strip()
    if coin_id in reverse_mapping:
        return reverse_mapping[coin_id]

    # Fallback: uppercase
    base = coin_id.replace("-", "").upper()[:8]
    # Mappings spéciaux pour les noms courts
    special = {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
        "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT",
        "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT",
    }
    if base in special:
        return special[base]

    return f"{base}USDT"


def symbol_to_coin_id(symbol: str) -> str:
    """Convertit un symbol Binance en CoinGecko ID (approximatif)."""
    mapping = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
        "SOLUSDT": "solana", "ADAUSDT": "cardano",
        "XRPUSDT": "ripple", "DOTUSDT": "polkadot",
    }
    s = symbol.upper()
    if s in mapping:
        return mapping[s]
    for suffix in ["USDT", "BUSD", "USDC"]:
        if s.endswith(suffix):
            return s[: -len(suffix)].lower()
    return s.lower()


# ─── CLI autonome (test) ─────────────────────────────────────────

def main():
    """Point d'entrée pour tester le connecteur indépendamment."""
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Exchange Connector")
    parser.add_argument("--mode", default="paper", choices=["paper", "live", "binance"],
                        help="Mode de trading")
    parser.add_argument("--capital", type=float, default=10000,
                        help="Capital initial (paper)")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbole à tester")
    parser.add_argument("--balance", action="store_true",
                        help="Afficher le solde")
    parser.add_argument("--price", action="store_true",
                        help="Afficher le prix")
    parser.add_argument("--buy", type=float, default=0,
                        help="Quantité à acheter (market)")
    parser.add_argument("--sell", type=float, default=0,
                        help="Quantité à vendre (market)")
    parser.add_argument("--create-keys", action="store_true",
                        help="Créer le fichier api_keys.json template")
    args = parser.parse_args()

    if args.create_keys:
        save_api_keys_template()
        return

    try:
        connector = get_connector(mode=args.mode, initial_capital=args.capital)
    except ValueError as e:
        print(f"  ❌ {e}")
        return

    print(f"  Connecteur: {connector.get_info()['name']} ({connector.get_info()['type']})")

    if args.balance:
        balances = connector.get_balance()
        print(f"\n  Solde:")
        for asset, bal in sorted(balances.items(), key=lambda x: x[0]):
            if bal.total > 0:
                print(f"    {asset:10s} {bal.free:>12.6f} (locked: {bal.locked:.6f})")

    if args.price:
        price = connector.get_price(args.symbol)
        print(f"\n  Prix {args.symbol}: ${price:.2f}")

    if args.buy > 0:
        try:
            order = connector.market_buy(args.symbol, args.buy)
            print(f"  ACHAT: {order}")
        except Exception as e:
            print(f"  ❌ ACHAT échoué: {e}")

    if args.sell > 0:
        try:
            order = connector.market_sell(args.symbol, args.sell)
            print(f"  VENTE: {order}")
        except Exception as e:
            print(f"  ❌ VENTE échoué: {e}")

    # Afficher le résumé si paper
    if isinstance(connector, PaperTradingConnector):
        summary = connector.get_portfolio_summary()
        print(f"\n  Résumé portefeuille:")
        print(f"    Cash: ${summary['cash']:,.2f}")
        print(f"    Holdings: ${summary['holdings_value']:,.2f}")
        print(f"    Total: ${summary['total_value']:,.2f}")
        print(f"    P&L: ${summary['pnl']:+,.2f} ({summary['pnl_pct']:+.2f}%)")
        print(f"    Trades: {summary['total_trades']}")


if __name__ == "__main__":
    main()
