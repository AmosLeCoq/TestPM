"""Stockage SQLite."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "pmu.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id TEXT PRIMARY KEY,          -- AAAA-MM-JJ_R1C1
    date TEXT NOT NULL,
    reunion INTEGER, course INTEGER,
    hippodrome TEXT, pays TEXT,
    libelle TEXT, discipline TEXT, specialite TEXT,
    distance INTEGER, montant_prix INTEGER,
    nb_declares INTEGER, heure_depart INTEGER,   -- epoch ms
    statut TEXT, categorie_statut TEXT,
    ordre_arrivee TEXT,                -- JSON
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);

CREATE TABLE IF NOT EXISTS runners (
    race_id TEXT NOT NULL, num INTEGER NOT NULL,
    nom TEXT, age INTEGER, sexe TEXT, statut TEXT,
    driver TEXT, entraineur TEXT, musique TEXT,
    nb_courses INTEGER, nb_victoires INTEGER, nb_places INTEGER,
    gains_carriere INTEGER, gains_annee INTEGER,
    handicap_poids INTEGER, handicap_distance INTEGER, place_corde INTEGER,
    oeilleres TEXT, deferre TEXT, inedit INTEGER,
    cote_ref REAL, cote_direct REAL,
    ordre_arrivee INTEGER,
    PRIMARY KEY (race_id, num)
);

CREATE TABLE IF NOT EXISTS dividends (
    race_id TEXT NOT NULL, type_pari TEXT NOT NULL, combinaison TEXT NOT NULL,
    dividende REAL,                    -- en euros pour 1 EUR
    PRIMARY KEY (race_id, type_pari, combinaison)
);

CREATE TABLE IF NOT EXISTS predictions (
    race_id TEXT NOT NULL, num INTEGER NOT NULL,
    proba REAL, rang INTEGER, cote_ref REAL, value REAL,
    model_version TEXT, predicted_at TEXT,
    PRIMARY KEY (race_id, num)
);

CREATE TABLE IF NOT EXISTS checks (
    race_id TEXT PRIMARY KEY, checked_at TEXT,
    gagnant INTEGER, rang_gagnant INTEGER, proba_gagnant REAL,
    top1_ok INTEGER, top3_dans_arrivee INTEGER,
    mise REAL, gain REAL, notified INTEGER DEFAULT 0,
    mise_place REAL DEFAULT 0, gain_place REAL DEFAULT 0
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(checks)")}
    for c in ("mise_place", "gain_place"):
        if c not in cols:                      # bases créées avant cette version
            con.execute(f"ALTER TABLE checks ADD COLUMN {c} REAL DEFAULT 0")
    con.commit()
    return con
