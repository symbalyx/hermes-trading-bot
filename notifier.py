#!/usr/bin/env python3
"""
notifier.py — Module de notifications pour Hermes Trading Bot v5.
Support Telegram (Bot API) et Email (SMTP Zoho).
Lit la configuration depuis data/notifier_config.json.
Robuste : ne crash jamais, gère les erreurs silencieusement.
"""

import json
import logging
import smtplib
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("hermes.notifier")

# ─── Configuration ───────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
    },
    "email": {
        "enabled": False,
        "smtp_host": "smtp.zoho.eu",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "from_addr": "",
        "to_addrs": [""],
    },
    "notify_on": ["SIGNAL_FORT", "DIVERGENCE", "REGIME_CHANGE"],
    "min_score_notify": 0.4,
    "cooldown_minutes": 30,
}

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "data" / "notifier_config.json"


# ─── Utilitaires ──────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> dict:
    """Charge la configuration depuis le fichier JSON."""
    p = path or CONFIG_PATH
    if not p.exists():
        log.warning("Fichier config introuvable: %s", p)
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(p.read_text())
        # Fusion avec les défauts pour les clés manquantes
        config = dict(DEFAULT_CONFIG)
        for section in ["telegram", "email"]:
            if section in data:
                config[section].update(data[section])
        for key in ["notify_on", "min_score_notify", "cooldown_minutes"]:
            if key in data:
                config[key] = data[key]
        return config
    except (json.JSONDecodeError, OSError) as e:
        log.error("Erreur lecture config: %s", e)
        return dict(DEFAULT_CONFIG)


def save_config_template(path: Optional[Path] = None) -> Path:
    """Crée un fichier de configuration template."""
    p = path or CONFIG_PATH
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))
    log.info("Template config créé: %s", p)
    return p


# ─── Telegram Notifier ────────────────────────────────────────────────

class TelegramNotifier:
    """Envoie des notifications via Telegram Bot API.

    Caractéristiques :
      - Formatage HTML basique
      - Rate limiting : max 20 messages par minute
      - File d'attente si dépassement
    """

    API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_PER_MINUTE = 20
    WINDOW_SECONDS = 60

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = self.API_BASE.format(token=bot_token)
        self._sent_timestamps: list[float] = []
        self._queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._queue_thread: Optional[threading.Thread] = None
        self._running = True
        self._start_queue_processor()

    def _start_queue_processor(self):
        """Démarre un thread qui vide la file d'attente."""
        def _worker():
            while self._running:
                self._flush_queue()
                time.sleep(1)
        self._queue_thread = threading.Thread(target=_worker, daemon=True)
        self._queue_thread.start()

    def _flush_queue(self):
        """Envoie les messages en file d'attente dans la limite du rate limit."""
        with self._queue_lock:
            if not self._queue:
                return
            msgs = list(self._queue)
            self._queue.clear()

        for msg in msgs:
            # Vérifier le rate limit avant chaque envoi
            self._prune_timestamps()
            if len(self._sent_timestamps) >= self.MAX_PER_MINUTE:
                log.warning("Rate limit Telegram atteint, %d msg en attente", len(msgs) - msgs.index(msg))
                # Remettre le reste dans la file
                with self._queue_lock:
                    remaining = msgs[msgs.index(msg):]
                    self._queue = remaining + self._queue
                return
            self._send_single(msg)

    def _prune_timestamps(self):
        """Retire les timestamps plus vieux que la fenêtre."""
        cutoff = time.time() - self.WINDOW_SECONDS
        self._sent_timestamps = [t for t in self._sent_timestamps if t > cutoff]

    def _send_single(self, text: str) -> bool:
        """Envoie un message texte unique à Telegram."""
        try:
            r = requests.post(
                self.api_url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            self._sent_timestamps.append(time.time())
            if r.status_code == 429:
                retry_after = int(r.json().get("parameters", {}).get("retry_after", 30))
                log.warning("Telegram rate limit (429), pause %ds", retry_after)
                time.sleep(retry_after)
                # Réessayer
                return self._send_single(text)
            if r.status_code != 200:
                log.warning("Telegram error %s: %s", r.status_code, r.text[:200])
                return False
            return True
        except requests.Timeout:
            log.warning("Telegram timeout")
            return False
        except requests.ConnectionError:
            log.warning("Telegram connection error")
            return False
        except Exception as e:
            log.error("Telegram send error: %s", e)
            return False

    def send(self, text: str) -> bool:
        """Envoie un message ou le met en file d'attente si rate limit atteint."""
        self._prune_timestamps()
        if len(self._sent_timestamps) < self.MAX_PER_MINUTE:
            return self._send_single(text)
        # File d'attente
        with self._queue_lock:
            self._queue.append(text)
        log.info("Message mis en file d'attente Telegram (taille: %d)", len(self._queue))
        return True

    def send_test(self) -> bool:
        """Envoie un message de test."""
        return self.send("<b>🔔 HERMES BOT</b>\nTest de notification Telegram ✅\nLe bot fonctionne correctement.")

    def stop(self):
        """Arrête le thread de file d'attente."""
        self._running = False


# ─── Email Notifier ───────────────────────────────────────────────────

class EmailNotifier:
    """Envoie des notifications par email via SMTP.

    Caractéristiques :
      - Support SMTP (Zoho Europe: smtp.zoho.eu:587)
      - Destinataires multiples
      - Template HTML simple
    """

    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0a0e17; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 600px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #00bcd4, #0097a7);
          padding: 15px; border-radius: 8px 8px 0 0; text-align: center; }}
.header h1 {{ color: white; margin: 0; font-size: 1.3rem; }}
.alert {{ background: #111827; padding: 15px; margin: 10px 0; border-radius: 6px;
         border-left: 4px solid {border_color}; }}
.alert h3 {{ margin: 0 0 5px 0; color: {title_color}; }}
.alert .meta {{ color: #888; font-size: 0.85rem; }}
.alert .msg {{ margin: 5px 0; }}
.score {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-weight: bold; font-size: 0.85rem; }}
.score-pos {{ background: #00c85333; color: #00c853; }}
.score-neg {{ background: #ff174433; color: #ff1744; }}
.footer {{ margin-top: 20px; color: #666; font-size: 0.75rem; text-align: center; }}
</style></head>
<body>
<div class="container">
<div class="header"><h1>🔔 Hermes Trading Bot — Alerte</h1></div>
{alerts_html}
<div class="footer">
<p>Généré par Hermes Trading Bot v5 — {timestamp}</p>
<p>Pour configurer vos notifications, modifiez data/notifier_config.json</p>
</div>
</div>
</body>
</html>"""

    ALERT_HTML = """<div class="alert">
<h3>{icon} {type}</h3>
<div class="meta">Coin : <strong>{coin}</strong> | {timestamp}</div>
<div class="msg">{message}</div>
<div><span class="score {score_class}">Score : {score}</span></div>
</div>"""

    TYPE_CONFIG = {
        "SIGNAL_FORT":     {"icon": "🔴🔵", "border_color": "#ff9800", "title_color": "#ff9800"},
        "DIVERGENCE":      {"icon": "⚡",   "border_color": "#e040fb", "title_color": "#e040fb"},
        "SQUEEZE":         {"icon": "💥",   "border_color": "#00bcd4", "title_color": "#00bcd4"},
        "SCORE_EXTREME":   {"icon": "🔥",   "border_color": "#ff5252", "title_color": "#ff5252"},
        "REGIME_CHANGE":   {"icon": "🔄",   "border_color": "#69f0ae", "title_color": "#69f0ae"},
    }

    def __init__(self, smtp_host: str, smtp_port: int, username: str,
                 password: str, from_addr: str, to_addrs: list[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = [a for a in to_addrs if a]  # Filtrer les vides

    def _get_type_config(self, alert_type: str) -> dict:
        """Retourne la config visuelle pour un type d'alerte."""
        return self.TYPE_CONFIG.get(alert_type, {
            "icon": "⚠",
            "border_color": "#ffc107",
            "title_color": "#ffc107",
        })

    def _build_html(self, alerts: list[dict]) -> str:
        """Construit le HTML complet pour les alertes."""
        if not alerts:
            return ""

        alerts_html_parts = []
        for a in alerts:
            tc = self._get_type_config(a.get("type", "UNKNOWN"))
            score_val = a.get("score", 0.0)
            score_class = "score-pos" if score_val >= 0 else "score-neg"
            score_str = f"{score_val:+.2f}" if score_val != 0 else "0.00"

            alerts_html_parts.append(self.ALERT_HTML.format(
                icon=tc["icon"],
                type=a.get("type", "ALERTE"),
                coin=a.get("coin", "?"),
                timestamp=a.get("timestamp", datetime.now().strftime("%H:%M:%S")),
                message=a.get("message", ""),
                score=score_str,
                score_class=score_class,
            ))

        return self.HTML_TEMPLATE.format(
            border_color="#00bcd4",
            title_color="#00bcd4",
            alerts_html="\n".join(alerts_html_parts),
            timestamp=datetime.now().strftime("%d/%m/%Y %H:%M UTC"),
        )

    def send(self, subject: str, text_body: str, html_body: Optional[str] = None) -> bool:
        """Envoie un email multi-parties (texte + HTML optionnel)."""
        if not self.to_addrs:
            log.warning("Aucun destinataire email configuré")
            return False

        try:
            msg = MIMEText(html_body or text_body, "html" if html_body else "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            log.info("Email envoyé à %s: %s", self.to_addrs, subject)
            return True

        except smtplib.SMTPAuthenticationError:
            log.warning("Email: échec d'authentification SMTP — vérifiez votre mot de passe")
            return False
        except smtplib.SMTPException as e:
            log.warning("Email SMTP error: %s", e)
            return False
        except OSError as e:
            log.warning("Email connection error: %s", e)
            return False
        except Exception as e:
            log.error("Email send error: %s", e)
            return False

    def send_alerts(self, alerts: list[dict]) -> bool:
        """Envoie un email structuré avec les alertes."""
        if not alerts:
            return False

        # Sujet
        types = set(a["type"] for a in alerts)
        subject = f"[Hermes Bot] {len(alerts)} alerte(s) — {', '.join(sorted(types))[:80]}"

        # Texte brut en fallback
        text_parts = [f"=== HERMES TRADING BOT - ALERTES ==="]
        for a in alerts:
            text_parts.append(f"[{a['type']}] {a['coin']}: {a['message']} (score {a.get('score', 0):+.2f})")
        text_body = "\n".join(text_parts)

        # HTML
        html_body = self._build_html(alerts)

        return self.send(subject, text_body, html_body)

    def send_test(self) -> bool:
        """Envoie un email de test."""
        test_alerts = [{
            "type": "TEST",
            "coin": "bitcoin",
            "message": "Ceci est un test de notification. Si vous lisez ce message, la configuration email fonctionne correctement.",
            "score": 0.42,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }]
        return self.send_alerts(test_alerts)


# ─── Notifier Principal ───────────────────────────────────────────────

class Notifier:
    """Gestionnaire de notifications unifié (Telegram + Email).

    Points clés :
      - Lit la config depuis data/notifier_config.json
      - Gère le cooldown par type+coin (pas de doublon en N minutes)
      - Filtre les alertes selon notify_on
      - Gère les erreurs silencieusement
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.config = load_config(self.config_path)
        self._cooldown: dict[tuple[str, str], datetime] = {}  # (type, coin) → datetime
        self._lock = threading.Lock()

        # Initialiser les sous-notificateurs
        self.telegram: Optional[TelegramNotifier] = None
        self.email: Optional[EmailNotifier] = None

        tg = self.config.get("telegram", {})
        if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
            try:
                self.telegram = TelegramNotifier(tg["bot_token"], tg["chat_id"])
                log.info("Telegram notifier activé")
            except Exception as e:
                log.warning("Échec init Telegram: %s", e)

        em = self.config.get("email", {})
        if em.get("enabled") and em.get("username") and em.get("password"):
            try:
                self.email = EmailNotifier(
                    smtp_host=em.get("smtp_host", "smtp.zoho.eu"),
                    smtp_port=em.get("smtp_port", 587),
                    username=em["username"],
                    password=em["password"],
                    from_addr=em.get("from_addr", em["username"]),
                    to_addrs=em.get("to_addrs", []),
                )
                log.info("Email notifier activé")
            except Exception as e:
                log.warning("Échec init Email: %s", e)

        if not self.telegram and not self.email:
            log.warning("Aucun notificateur activé — vérifiez %s", self.config_path)

    def is_enabled(self) -> bool:
        """Vérifie si au moins un canal de notification est actif."""
        return bool(self.telegram) or bool(self.email)

    def send_alert(self, alert_type: str, coin: str, message: str,
                   score: float = 0.0) -> bool:
        """Méthode principale pour envoyer une alerte.

        Vérifie :
          1. Si le type d'alerte est dans notify_on
          2. Si le score est >= min_score_notify (si score != 0)
          3. Le cooldown (pas 2 fois la même alerte type+coin en N minutes)
        Ensuite, formate et envoie via Telegram + Email.

        Args:
            alert_type: Type d'alerte (SIGNAL_FORT, DIVERGENCE, SQUEEZE, etc.)
            coin: Identifiant du coin
            message: Message descriptif
            score: Score normalisé associé

        Returns:
            True si au moins un canal a envoyé
        """
        # 1. Vérifier si ce type est dans notify_on
        notify_on = self.config.get("notify_on", DEFAULT_CONFIG["notify_on"])
        if alert_type not in notify_on:
            return False

        # 2. Vérifier le score minimum
        min_score = self.config.get("min_score_notify", DEFAULT_CONFIG["min_score_notify"])
        if score != 0 and abs(score) < min_score:
            return False

        # 3. Vérifier le cooldown
        cooldown_min = self.config.get("cooldown_minutes", DEFAULT_CONFIG["cooldown_minutes"])
        dedup_key = (alert_type, coin)
        with self._lock:
            last_sent = self._cooldown.get(dedup_key)
            if last_sent and datetime.now() - last_sent < timedelta(minutes=cooldown_min):
                remaining = cooldown_min - (datetime.now() - last_sent).total_seconds() / 60
                log.debug("Cooldown actif pour %s/%s: encore %.1f min",
                          alert_type, coin, remaining)
                return False
            self._cooldown[dedup_key] = datetime.now()

        # 4. Formater le message
        sent = False

        # Message Telegram
        tg_text = self._format_telegram(alert_type, coin, message, score)
        if self.telegram and tg_text:
            try:
                if self.telegram.send(tg_text):
                    sent = True
            except Exception as e:
                log.warning("Telegram send_alert error: %s", e)

        # Message Email
        email_alert = {
            "type": alert_type,
            "coin": coin,
            "message": message,
            "score": score,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        if self.email:
            try:
                if self.email.send_alerts([email_alert]):
                    sent = True
            except Exception as e:
                log.warning("Email send_alert error: %s", e)

        if sent:
            log.info("Notification envoyée: [%s] %s — %s", alert_type, coin, message[:80])

        return sent

    def send_alerts_batch(self, alerts: list[dict]) -> int:
        """Envoie plusieurs alertes en une fois.

        Args:
            alerts: Liste de dicts avec clés type, coin, message, score

        Returns:
            Nombre d'alertes envoyées
        """
        sent_count = 0
        for a in alerts:
            if self.send_alert(
                a.get("type", ""),
                a.get("coin", ""),
                a.get("message", ""),
                a.get("score", 0.0),
            ):
                sent_count += 1
        return sent_count

    def send_test(self) -> dict:
        """Envoie une notification de test sur tous les canaux actifs.

        Returns:
            Dict avec le statut de chaque canal
        """
        results = {}
        if self.telegram:
            results["telegram"] = self.telegram.send_test()
        if self.email:
            results["email"] = self.email.send_test()
        return results

    def _format_telegram(self, alert_type: str, coin: str,
                         message: str, score: float) -> str:
        """Formate une alerte pour Telegram (HTML)."""
        esc_coin = coin.replace("-", " ").title()

        type_icons = {
            "SIGNAL_FORT": "🔴🔵",
            "DIVERGENCE": "⚡",
            "SQUEEZE": "💥",
            "SCORE_EXTREME": "🔥",
            "REGIME_CHANGE": "🔄",
        }
        icon = type_icons.get(alert_type, "⚠")

        score_str = f"{score:+.2f}" if score != 0 else "N/A"

        return (
            f"<b>HERMES ALERT</b>\n"
            f"{icon} <b>{alert_type}</b>: <b>{esc_coin}</b>\n"
            f"{message}\n"
            f"Score: {score_str}"
        )

    def reload_config(self) -> bool:
        """Recharge la configuration depuis le fichier."""
        try:
            new_config = load_config(self.config_path)
            self.config = new_config

            # Réinitialiser les sous-notificateurs si la config a changé
            tg = new_config.get("telegram", {})
            if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
                if not self.telegram:
                    self.telegram = TelegramNotifier(tg["bot_token"], tg["chat_id"])
            else:
                if self.telegram:
                    self.telegram.stop()
                    self.telegram = None

            em = new_config.get("email", {})
            if em.get("enabled") and em.get("username") and em.get("password"):
                if not self.email:
                    self.email = EmailNotifier(
                        smtp_host=em.get("smtp_host", "smtp.zoho.eu"),
                        smtp_port=em.get("smtp_port", 587),
                        username=em["username"],
                        password=em["password"],
                        from_addr=em.get("from_addr", em["username"]),
                        to_addrs=em.get("to_addrs", []),
                    )
            else:
                self.email = None

            log.info("Configuration rechargée")
            return True
        except Exception as e:
            log.error("Erreur rechargement config: %s", e)
            return False

    def cleanup(self):
        """Nettoie les ressources (threads, connexions)."""
        if self.telegram:
            self.telegram.stop()


# ─── Fonctions utilitaires ────────────────────────────────────────────

def create_notifier(config_path: Optional[Path] = None) -> Notifier:
    """Factory : crée un Notifier à partir du chemin de config."""
    return Notifier(config_path)


def test_notifications(config_path: Optional[Path] = None) -> dict:
    """Teste tous les canaux configurés et retourne les résultats."""
    n = Notifier(config_path)
    results = n.send_test()
    n.cleanup()
    return results


# ─── CLI autonome ─────────────────────────────────────────────────────

def main():
    """Point d'entrée CLI pour tester/manipuler le notifier."""
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Notifier — module de notifications")
    parser.add_argument("--create-config", action="store_true", help="Créer le fichier de config template")
    parser.add_argument("--test", action="store_true", help="Envoyer une notification de test")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Chemin du fichier de config")
    args = parser.parse_args()

    config_path = Path(args.config)

    if args.create_config:
        path = save_config_template(config_path)
        print(f"  ✓ Template créé: {path}")
        print(f"  Éditez le fichier avec vos identifiants Telegram/Email.")
        return

    if args.test:
        print("  Envoi des notifications de test...")
        results = test_notifications(config_path)
        for channel, ok in results.items():
            status = "✓ OK" if ok else "✗ ÉCHEC"
            print(f"  {channel}: {status}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
