"""Ligne de commande : python -m pmu.cli <commande> ..."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta

from . import collect, db, discord, pipeline, scheduler


def _date(s: str) -> date:
    s = s.lower()
    today = datetime.now(pipeline.TZ).date()
    if s in ("today", "aujourdhui"):
        return today
    if s in ("tomorrow", "demain"):
        return today + timedelta(days=1)
    if s in ("yesterday", "hier"):
        return today - timedelta(days=1)
    return date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(prog="pmu", description="Prédictions PMU")
    ap.add_argument("--db", help="chemin de la base SQLite (défaut: data/pmu.db)")
    ap.add_argument("--webhook", help="URL du webhook Discord (sinon $DISCORD_WEBHOOK_URL)")
    ap.add_argument("--place-min", type=float, default=pipeline.PLACE_MIN,
                    help="proba minimale du favori pour la stratégie suivie en simple placé")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="récupère l'historique des courses")
    c.add_argument("--start", type=_date, help="AAAA-MM-JJ")
    c.add_argument("--end", type=_date, default=None, help="AAAA-MM-JJ (défaut: hier)")
    c.add_argument("--days", type=int, help="nb de jours en arrière (alternative à --start)")
    c.add_argument("--no-dividends", action="store_true")

    t = sub.add_parser("train", help="entraîne le modèle et affiche l'évaluation")
    t.add_argument("--test-frac", type=float, default=0.2)

    for name, hlp in (("predict", "probabilités pour les courses à venir"),
                      ("check", "compare les pronostics aux arrivées")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--date", type=_date, default=_date("today"))
        p.add_argument("--value", type=float, default=1.2, help="seuil proba*cote pour un pari 'value'")
        p.add_argument("--min-proba", type=float, default=0.05)
        p.add_argument("--no-refresh", action="store_true")
        if name == "predict":
            p.add_argument("--notify", action="store_true", help="envoie les pronostics sur Discord")
            p.add_argument("--all", action="store_true", help="inclut les courses déjà parties")
        else:
            p.add_argument("--no-notify", action="store_true")
            p.add_argument("--recheck", action="store_true")

    r = sub.add_parser("run", help="mode automatique continu (pronostics + vérifications)")
    r.add_argument("--predict-hour", type=int, default=10)
    r.add_argument("--every", type=int, default=15, help="minutes entre deux vérifications")
    r.add_argument("--value", type=float, default=1.2)
    r.add_argument("--min-proba", type=float, default=0.05)

    sub.add_parser("stats", help="bilan cumulé des pronostics vérifiés")
    sub.add_parser("test-webhook", help="envoie un message de test sur Discord")

    a = ap.parse_args()
    pipeline.PLACE_MIN = a.place_min
    con = db.connect(a.db)
    if a.cmd == "collect":
        end = a.end or _date("yesterday")
        start = a.start or end - timedelta(days=(a.days or 365) - 1)
        collect.collect_range(con, start, end, with_dividends=not a.no_dividends)
    elif a.cmd == "train":
        pipeline.train(con, test_frac=a.test_frac)
    elif a.cmd == "predict":
        pipeline.predict(con, a.date, refresh=not a.no_refresh, notify=a.notify,
                         value_threshold=a.value, min_proba=a.min_proba,
                         only_future=not a.all, webhook=a.webhook)
    elif a.cmd == "check":
        pipeline.check(con, a.date, refresh=not a.no_refresh, notify=not a.no_notify,
                       value_threshold=a.value, min_proba=a.min_proba,
                       webhook=a.webhook, recheck=a.recheck)
    elif a.cmd == "run":
        scheduler.run(con, predict_hour=a.predict_hour, check_every_min=a.every,
                      value_threshold=a.value, min_proba=a.min_proba, webhook=a.webhook)
    elif a.cmd == "stats":
        print(json.dumps(pipeline.stats(con), indent=2, ensure_ascii=False))
    elif a.cmd == "test-webhook":
        discord.send([{"title": "Test", "description": "Le webhook PMU Predictor fonctionne ✅",
                       "color": discord.COLOR_OK}], url=a.webhook)


if __name__ == "__main__":
    main()
