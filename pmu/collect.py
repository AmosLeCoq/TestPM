"""Collecte du programme, des partants, des cotes et des résultats."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from . import api


def race_id(d: date, r: int, c: int) -> str:
    return f"{d.isoformat()}_R{r}C{c}"


def _rapport(p: dict, key: str) -> float | None:
    v = p.get(key)
    return float(v["rapport"]) if isinstance(v, dict) and v.get("rapport") else None


def _store_race(con: sqlite3.Connection, d: date, reunion: dict, course: dict,
                parts: list[dict], divs: list[dict]) -> None:
    r, c = reunion["numOfficiel"], course["numOrdre"]
    rid = race_id(d, r, c)
    con.execute(
        """INSERT OR REPLACE INTO races (race_id, date, reunion, course, hippodrome, pays,
           libelle, discipline, specialite, distance, montant_prix, nb_declares, heure_depart,
           statut, categorie_statut, ordre_arrivee, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
        (rid, d.isoformat(), r, c,
         (course.get("hippodrome") or reunion.get("hippodrome") or {}).get("libelleCourt"),
         (reunion.get("pays") or {}).get("code"),
         course.get("libelle"), course.get("discipline"), course.get("specialite"),
         course.get("distance"), course.get("montantPrix"), course.get("nombreDeclaresPartants"),
         course.get("heureDepart"), course.get("statut"), course.get("categorieStatut"),
         json.dumps(course.get("ordreArrivee")) if course.get("ordreArrivee") else None))
    for p in parts:
        g = p.get("gainsParticipant") or {}
        con.execute(
            """INSERT OR REPLACE INTO runners VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, p.get("numPmu"), p.get("nom"), p.get("age"), p.get("sexe"), p.get("statut"),
             p.get("driver"), p.get("entraineur"), p.get("musique"),
             p.get("nombreCourses"), p.get("nombreVictoires"), p.get("nombrePlaces"),
             g.get("gainsCarriere"), g.get("gainsAnneeEnCours"),
             p.get("handicapPoids"), p.get("handicapDistance"), p.get("placeCorde"),
             p.get("oeilleres"), p.get("deferre"), int(bool(p.get("indicateurInedit"))),
             _rapport(p, "dernierRapportReference"), _rapport(p, "dernierRapportDirect"),
             p.get("ordreArrivee")))
    for bloc in divs:
        for rap in bloc.get("rapports", []):
            con.execute("INSERT OR REPLACE INTO dividends VALUES (?,?,?,?)",
                        (rid, bloc.get("typePari"), rap.get("combinaison"),
                         (rap.get("dividendePourUnEuro") or 0) / 100.0))


def collect_day(con: sqlite3.Connection, d: date, with_dividends: bool = True,
                workers: int = 6) -> int:
    """Collecte toutes les courses d'une journée. Retourne le nombre de courses."""
    prog = api.programme(d)
    if not prog:
        return 0
    jobs = [(reu, cou) for reu in prog.get("reunions", []) for cou in reu.get("courses", [])]

    def fetch(job):
        reu, cou = job
        r, c = reu["numOfficiel"], cou["numOrdre"]
        parts = api.participants(d, r, c)
        divs = api.rapports(d, r, c) if with_dividends and cou.get("categorieStatut") == "ARRIVEE" else []
        return reu, cou, parts, divs

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch, jobs))
    for reu, cou, parts, divs in results:
        _store_race(con, d, reu, cou, parts, divs)
    con.commit()
    return len(results)


def collect_range(con: sqlite3.Connection, start: date, end: date, skip_existing: bool = True,
                  with_dividends: bool = True, verbose: bool = True) -> None:
    d = start
    while d <= end:
        if skip_existing:
            row = con.execute(
                "SELECT COUNT(*) n, SUM(categorie_statut='ARRIVEE') a FROM races WHERE date=?",
                (d.isoformat(),)).fetchone()
            if row["n"] and row["n"] == row["a"]:
                d += timedelta(days=1)
                continue
        n = collect_day(con, d, with_dividends=with_dividends)
        if verbose:
            print(f"{d.isoformat()} : {n} courses", flush=True)
        d += timedelta(days=1)
