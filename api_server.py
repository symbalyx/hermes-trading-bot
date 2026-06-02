#!/usr/bin/env python3
"""
api_server.py — Minimal Flask REST API for Hermes Trading Bot.
Status, profit, trades, and signals endpoints.
Inspired by Freqtrade's REST API.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

log = logging.getLogger("hermes.api")

# Try to import bot modules
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from persistence import TradeDatabase
    HAS_PERSISTENCE = True
except ImportError:
    HAS_PERSISTENCE = False


def create_app(db_path: str = "data/trades.db",
               config_path: str = "config.json",
               bot_instance=None) -> Optional[object]:
    """
    Create and configure the Flask application.

    Args:
        db_path: Path to SQLite database
        config_path: Path to config.json
        bot_instance: Optional reference to running bot for live control

    Returns:
        Flask app, or None if Flask is not installed
    """
    if not HAS_FLASK:
        log.warning("Flask not installed. Install with: pip install flask")
        return None

    app = Flask(__name__)

    # CORS support for web UI
    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # Initialize database
    db = TradeDatabase(db_path) if HAS_PERSISTENCE else None
    bot = bot_instance

    # ─── Status endpoints ─────────────────────────────────────────

    @app.route("/api/v1/ping", methods=["GET"])
    def ping():
        """Health check."""
        return jsonify({"status": "pong", "timestamp": datetime.utcnow().isoformat()})

    @app.route("/api/v1/status", methods=["GET"])
    def status():
        """Bot status overview."""
        result = {
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "4.0.0",
            "strategy": "HermesV4",
        }
        if db:
            stats = db.get_stats()
            result.update({
                "total_trades": stats.get("total_trades", 0),
                "open_trades": stats.get("open_trades", 0),
                "winning_trades": stats.get("winning_trades", 0),
                "losing_trades": stats.get("losing_trades", 0),
                "win_rate": round(stats.get("win_rate", 0) * 100, 1),
            })
        if bot and hasattr(bot, "config"):
            cfg = bot.config if isinstance(bot.config, dict) else {}
            result["dry_run"] = cfg.get("dry_run", True)
            result["stake_amount"] = cfg.get("stake_amount", 100)
            result["max_open_trades"] = cfg.get("max_open_trades", 3)
        return jsonify(result)

    # ─── Profit endpoints ─────────────────────────────────────────

    @app.route("/api/v1/profit", methods=["GET"])
    def profit():
        """Profit summary."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        try:
            summary = db.get_profit_summary()
            summary["timestamp"] = datetime.utcnow().isoformat()
            return jsonify(summary)
        except Exception as e:
            log.error("Profit endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ─── Trades endpoints ─────────────────────────────────────────

    @app.route("/api/v1/trades", methods=["GET"])
    def list_trades():
        """List all trades."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        try:
            limit = request.args.get("limit", 100, type=int)
            status_filter = request.args.get("status")
            trades = db.get_all_trades(limit=limit)
            if status_filter:
                trades = [t for t in trades if t.get("status") == status_filter]
            return jsonify({"trades": trades, "count": len(trades)})
        except Exception as e:
            log.error("Trades endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/trades/<int:trade_id>", methods=["GET"])
    def get_trade(trade_id: int):
        """Get a single trade."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        trade = db.get_trade(trade_id)
        if not trade:
            return jsonify({"error": "Trade not found"}), 404
        orders = db.get_orders_for_trade(trade_id)
        trade["orders"] = orders
        return jsonify(trade)

    @app.route("/api/v1/trades/open", methods=["GET"])
    def open_trades():
        """List open trades."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        try:
            coin_id = request.args.get("coin_id")
            trades = db.get_open_trades(coin_id=coin_id)
            return jsonify({"trades": trades, "count": len(trades)})
        except Exception as e:
            log.error("Open trades endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ─── Signals endpoints ────────────────────────────────────────

    @app.route("/api/v1/signals", methods=["GET"])
    def list_signals():
        """List recent signals."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        try:
            limit = request.args.get("limit", 50, type=int)
            signals = db.get_recent_signals(limit=limit)
            return jsonify({"signals": signals, "count": len(signals)})
        except Exception as e:
            log.error("Signals endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ─── Performance endpoints ────────────────────────────────────

    @app.route("/api/v1/performance", methods=["GET"])
    def performance():
        """Daily performance data."""
        if not db:
            return jsonify({"error": "Database not available"}), 503
        try:
            conn = db._get_conn()
            rows = conn.execute(
                "SELECT * FROM performance ORDER BY date DESC LIMIT 90"
            ).fetchall()
            data = [dict(r) for r in rows]
            return jsonify({"performance": data, "count": len(data)})
        except Exception as e:
            log.error("Performance endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ─── Config endpoint ──────────────────────────────────────────

    @app.route("/api/v1/config", methods=["GET"])
    def get_config():
        """Get sanitized config (no secrets)."""
        if not bot or not hasattr(bot, "config"):
            return jsonify({"error": "Bot config not available"}), 503
        cfg = dict(bot.config) if isinstance(bot.config, dict) else {}
        # Remove secrets
        for key in ["api_key", "api_secret", "telegram_bot_token", "telegram_chat_id"]:
            cfg.pop(key, None)
        cfg.pop("api_key", None)
        cfg.pop("api_secret", None)
        return jsonify(cfg)

    # ─── Analysis endpoints ───────────────────────────────────────

    @app.route("/api/v1/analyze", methods=["POST"])
    def analyze():
        """Trigger analysis on provided data."""
        data = request.get_json(silent=True) or {}
        coin_id = data.get("coin_id", "bitcoin")
        try:
            # Try to use bot's analyzer
            if bot and hasattr(bot, "analyze_coin"):
                result = bot.analyze_coin(coin_id)
                if result:
                    return jsonify({"coin_id": coin_id, "result": result.__dict__ if hasattr(result, "__dict__") else result})
            return jsonify({"coin_id": coin_id, "message": "Analysis not available"}), 501
        except Exception as e:
            log.error("Analyze endpoint error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ─── Static web UI ────────────────────────────────────────────

    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.exists():
        @app.route("/", methods=["GET"])
        def index():
            return app.send_static_file_or_404(str(webui_dir / "index.html"))

        # Serve static files from webui/
        @app.route("/<path:filename>", methods=["GET"])
        def serve_static(filename: str):
            path = webui_dir / filename
            if path.exists() and path.is_file():
                return app.send_static_file_or_404(str(path))
            return jsonify({"error": "Not found"}), 404

    return app


def run_server(host: str = "127.0.0.1", port: int = 8080,
               db_path: str = "data/trades.db",
               config_path: str = "config.json",
               debug: bool = False):
    """Run the API server."""
    app = create_app(db_path=db_path, config_path=config_path)
    if not app:
        log.error("Cannot start API server: Flask not installed")
        return False
    log.info("Starting API server on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Trading Bot API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--db", default="data/trades.db", help="Database path")
    parser.add_argument("--config", default="config.json", help="Config path")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, db_path=args.db, config_path=args.config, debug=args.debug)
