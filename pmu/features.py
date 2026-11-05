"""Construction des variables explicatives (uniquement des infos connues AVANT la course)."""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

ODDS_FEATURE = "lp_odds"
BASE_FEATURES = [
    "mus_mean5", "mus_wins5", "mus_top3_5", "mus_n", "mus_last",
    "wr_carriere", "pr_carriere", "log_gain_par_course", "log_gains_annee",
    "age", "inedit",
    "driver_wr", "driver_log_n", "trainer_wr", "trainer_log_n", "cheval_driver_wr",
    "recul", "poids_rel", "corde_rel", "oeilleres", "deferre4",
]
FEATURES_WITH_ODDS = [ODDS_FEATURE] + BASE_FEATURES

_MUS_RE = re.compile(r"(\d+|[A-Za-z])([a-z])")


@lru_cache(maxsize=None)
def parse_musique(m: str | None) -> tuple[float, float, float, float, float]:
    """Musique PMU (ex: '1a3aDa(25)5a') -> (moy 5 dernières, victoires, top3, nb, dernière)."""
    if not m:
        return (np.nan, 0.0, 0.0, 0.0, np.nan)
    m = re.sub(r"\(\d+\)", "", m)
    pos = []
    for tok, _disc in _MUS_RE.findall(m):
        if tok.isdigit():
            v = int(tok)
            pos.append(10 if v == 0 else min(v, 10))
        else:  # D(isqualifié) A(rrêté) T(ombé) R(etiré)...
            pos.append(11)
    if not pos:
        return (np.nan, 0.0, 0.0, 0.0, np.nan)
    last5 = pos[:5]
    return (float(np.mean(last5)), float(sum(p == 1 for p in last5)),
            float(sum(p <= 3 for p in last5)), float(len(pos)), float(pos[0]))


def load_runners(con: sqlite3.Connection) -> pd.DataFrame:
    q = """SELECT r.*, ra.date, ra.discipline, ra.specialite, ra.distance, ra.pays,
                  ra.hippodrome, ra.heure_depart, ra.categorie_statut, ra.libelle
           FROM runners r JOIN races ra USING(race_id)
           WHERE r.statut = 'PARTANT'"""
    df = pd.read_sql(q, con)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _smoothed_history(df: pd.DataFrame, key: str, prior: float, k: float,
                      prefix: str) -> pd.DataFrame:
    """Taux de victoire lissé de `key`, calculé uniquement sur les jours STRICTEMENT antérieurs."""
    fin = df[df["finished"]]
    agg = (fin.assign(w=fin["win"].astype(float), n=1.0)
              .groupby([key, "date"])[["w", "n"]].sum().reset_index()
              .sort_values([key, "date"]))
    agg[["cw", "cn"]] = agg.groupby(key)[["w", "n"]].cumsum()   # cumul jour inclus
    pairs = df[[key, "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(pairs, agg[[key, "date", "cw", "cn"]].sort_values("date"),
                      on="date", by=key, direction="backward", allow_exact_matches=False)
    cw, cn = m["cw"].fillna(0.0), m["cn"].fillna(0.0)
    m[f"{prefix}_wr"] = (cw + prior * k) / (cn + k)
    m[f"{prefix}_log_n"] = np.log1p(cn)
    return m[[key, "date", f"{prefix}_wr", f"{prefix}_log_n"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["finished"] = df["categorie_statut"].eq("ARRIVEE")
    df["win"] = df["ordre_arrivee"].eq(1) & df["finished"]
    n_race = df.groupby("race_id")["num"].transform("count")
    df["nb_partants"] = n_race

    # Cotes du matin (référence) -> probabilité implicite normalisée
    # Si au moins 75 % des partants ont une cote, les cotes manquantes sont remplacées
    # par la moitié de la plus petite probabilité implicite de la course.
    inv = 1.0 / df["cote_ref"].where(df["cote_ref"] > 1.0)
    frac = inv.notna().groupby(df["race_id"]).transform("mean")
    has = frac >= 0.75
    inv = inv.fillna(inv.groupby(df["race_id"]).transform("min") * 0.5)
    s = inv.groupby(df["race_id"]).transform("sum")
    df["has_odds"] = has
    df[ODDS_FEATURE] = np.log((inv / s).where(has))

    mus = np.array([parse_musique(m) for m in df["musique"].fillna("")])
    df["mus_mean5"], df["mus_wins5"], df["mus_top3_5"], df["mus_n"], df["mus_last"] = mus.T
    for c in ("mus_mean5", "mus_last"):
        df[c] = df[c].fillna(df.groupby("race_id")[c].transform("mean")).fillna(8.0)

    nc = df["nb_courses"].fillna(0)
    df["wr_carriere"] = (df["nb_victoires"].fillna(0) + 1) / (nc + 10)
    df["pr_carriere"] = (df["nb_places"].fillna(0) + 3) / (nc + 10)
    df["log_gain_par_course"] = np.log1p(df["gains_carriere"].fillna(0) / 100 / np.maximum(nc, 1))
    df["log_gains_annee"] = np.log1p(df["gains_annee"].fillna(0) / 100)
    df["age"] = df["age"].fillna(df["age"].median())
    df["inedit"] = df["inedit"].fillna(0)

    for key, prefix in (("driver", "driver"), ("entraineur", "trainer")):
        df[key] = df[key].fillna("?")
        h = _smoothed_history(df, key, prior=0.1, k=20, prefix=prefix)
        df = df.merge(h, on=[key, "date"], how="left")
    df["cheval_driver"] = df["nom"].fillna("?") + "|" + df["driver"]
    h = _smoothed_history(df, "cheval_driver", prior=0.1, k=5, prefix="cheval_driver")
    df = df.merge(h[["cheval_driver", "date", "cheval_driver_wr"]], on=["cheval_driver", "date"], how="left")

    g = df.groupby("race_id")
    hd = df["handicap_distance"]
    df["recul"] = (hd - g["handicap_distance"].transform("min")).fillna(0) / 25.0
    df["poids_rel"] = (df["handicap_poids"] - g["handicap_poids"].transform("mean")).fillna(0) / 10.0
    df["corde_rel"] = ((df["place_corde"] - 1) / df["nb_partants"]).fillna(0.5)
    df["oeilleres"] = df["oeilleres"].fillna("SANS_OEILLERES").ne("SANS_OEILLERES").astype(float)
    df["deferre4"] = df["deferre"].fillna("").str.contains("ANTERIEURS_POSTERIEURS").astype(float)

    for c in FEATURES_WITH_ODDS:
        if c != ODDS_FEATURE:
            df[c] = df[c].astype(float).fillna(0.0)
    return df.sort_values(["date", "race_id", "num"]).reset_index(drop=True)
