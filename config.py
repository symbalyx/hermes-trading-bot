#!/usr/bin/env python3
"""
config.py — Centralised configuration loader for Hermes Trading Bot.
Reads from .env file first, then config.json, then defaults.
Inspired by Freqtrade's configuration system.
"""
import json
import os
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("hermes.config")

# Default paths
BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
DOTENV_PATH = BASE_DIR / ".env"


class Config:
    """Centralised configuration with env var override support."""

    _instance = None
    _data: dict = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None, dotenv_path: Optional[Path] = None):
        if self._data:
            return
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.dotenv_path = dotenv_path or DOTENV_PATH
        self._load_all()

    def _load_dotenv(self) -> dict:
        """Load .env file manually (no python-dotenv dependency)."""
        env = {}
        path = self.dotenv_path
        if not path.exists():
            return env
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("\"'")
            env[key] = val
            os.environ.setdefault(key, val)
        return env

    def _load_config_json(self) -> dict:
        """Load config.json if it exists."""
        path = self.config_path
        if not path.exists():
            log.info("No config.json found at %s, using defaults", path)
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to load config.json: %s", e)
            return {}

    DEFAULTS = {
        # Exchange
        "exchange": "binance",
        "exchange_mode": "paper",
        "api_key": "",
        "api_secret": "",
        # Trading
        "stake_currency": "USDT",
        "stake_amount": 100.0,
        "max_open_trades": 3,
        "taker_fee": 0.001,
        "maker_fee": 0.001,
        "dry_run": True,
        "dry_run_wallet": 10000.0,
        # Strategy
        "strategy": "HermesV4",
        "timeframe": "1d",
        "min_score": 0.25,
        "max_position_adjustment": 2,
        # Risk management
        "max_drawdown": 0.15,
        "stoploss": -0.05,
        "trailing_stop": True,
        "trailing_stop_positive": 0.01,
        "trailing_stop_positive_offset": 0.02,
        "risk_per_trade": 0.02,
        "kelly_fraction": 0.5,
        "cooldown_period": 24,
        "consecutive_losses_limit": 3,
        # Data
        "coin_list": [
            "bitcoin", "ethereum", "cardano", "solana", "ripple",
            "polkadot", "dogecoin", "avalanche", "chainlink", "polygon",
        ],
        "data_dir": "data",
        "cache_ttl": 600,
        "request_delay": 3.0,
        # Notifications
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        # LLM
        "llm_enabled": False,
        "llm_model": "qwen2.5:3b",
        "ollama_base_url": "http://localhost:11434",
        # API server
        "api_server_enabled": False,
        "api_server_port": 8080,
        "api_server_host": "127.0.0.1",
    }

    def _load_all(self):
        env = self._load_dotenv()
        cfg = self._load_config_json()
        # Merge: defaults -> config.json -> .env -> actual env vars
        self._data = dict(self.DEFAULTS)
        self._data.update(cfg)
        # .env values
        for key, val in env.items():
            self._data[key.lower()] = self._coerce(val)
        # Real env vars override
        for key in self._data:
            env_key = f"HERMES_{key.upper()}"
            if env_key in os.environ:
                self._data[key] = self._coerce(os.environ[env_key])

    @staticmethod
    def _coerce(val: str):
        """Coerce string env var to proper type."""
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        # Try JSON for lists/dicts
        if val.startswith(("[", "{")):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
        return val

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict:
        return dict(self._data)

    @property
    def exchange_config(self) -> dict:
        return {
            "name": self.get("exchange", "binance"),
            "api_key": self.get("api_key", ""),
            "api_secret": self.get("api_secret", ""),
            "testnet": self.get("exchange_mode") == "testnet",
            "paper": self.get("exchange_mode") == "paper" or self.get("dry_run", True),
        }

    @property
    def is_dry_run(self) -> bool:
        return self.get("dry_run", True)

    @property
    def stake_amount(self) -> float:
        return float(self.get("stake_amount", 100.0))


# Singleton accessor
config = Config()
