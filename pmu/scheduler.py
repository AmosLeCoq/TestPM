"""Boucle automatique : pronostics le matin, vérification au fil de la journée."""
from __future__ import annotations

import time
import traceback
from datetime import datetime, timedelta

from . import collect, discord, pipeline


def run(con, predict_hour: int = 10, check_every_min: int = 15, retrain_weekday: int = 0,
        value_threshold: float = 1.2, min_proba: float = 0.05, webhook: str | None = None) -> None:
    """
    - chaque jour à `predict_hour` (heure suisse) : pronostics du jour -> Discord
    - toutes les `check_every_min` minutes : vérifie les courses terminées -> Discord
    - après minuit : collecte complète de la veille (+ ré-entraînement le lundi)
    """
    last_predict = last_nightly = None
    print("Boucle démarrée (Ctrl+C pour arrêter)", flush=True)
    while True:
        try:
            now = datetime.now(pipeline.TZ)
            today = now.date()
            if last_nightly != today:
                yesterday = today - timedelta(days=1)
                pipeline.check(con, yesterday, notify=True, value_threshold=value_threshold,
                               min_proba=min_proba, webhook=webhook)
                collect.collect_day(con, yesterday)
                if today.weekday() == retrain_weekday or not pipeline.ODDS_MODEL.exists():
                    pipeline.train(con, verbose=False)
                    print(f"{now:%H:%M} modèle ré-entraîné", flush=True)
                last_nightly = today
            if last_predict != today and now.hour >= predict_hour:
                pipeline.predict(con, today, notify=True, value_threshold=value_threshold,
                                 min_proba=min_proba, webhook=webhook)
                last_predict = today
            if last_predict == today:
                pipeline.check(con, today, notify=True, value_threshold=value_threshold,
                               min_proba=min_proba, webhook=webhook)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            discord.send([{"title": "⚠️ Erreur PMU Predictor", "color": discord.COLOR_KO,
                           "description": f"```{traceback.format_exc()[-1500:]}```"}], url=webhook)
        time.sleep(check_every_min * 60)
