"""Entraînement, prédiction et vérification des pronostics."""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import collect, discord, features
from .model import MODELS_DIR, ConditionalLogit

TZ = ZoneInfo("Europe/Zurich")
ODDS_MODEL = MODELS_DIR / "model_avec_cotes.json"
BASE_MODEL = MODELS_DIR / "model_sans_cotes.json"
REPORT = MODELS_DIR / "rapport_entrainement.json"
MODEL_VERSION = "condlogit-v1"
# Stratégie suivie en simple placé : 1 EUR placé sur notre favori quand sa proba >= PLACE_MIN
PLACE_MIN = 0.40


# --------------------------------------------------------------------- utilitaires
def _training_set(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["finished"] & (df["nb_partants"] >= 3)]
    winners = df.groupby("race_id")["win"].transform("sum")
    return df[winners >= 1].sort_values(["date", "race_id", "num"]).reset_index(drop=True)


def _sg_dividends(con: sqlite3.Connection, pari: str = "SIMPLE_GAGNANT") -> dict[tuple[str, int], float]:
    rows = con.execute("SELECT race_id, combinaison, dividende FROM dividends "
                       "WHERE type_pari IN (?, ?)", (pari, "E_" + pari)).fetchall()
    out = {}
    for r in rows:
        if r["combinaison"].isdigit():
            out.setdefault((r["race_id"], int(r["combinaison"])), r["dividende"])
    return out


def _payout(row, divs) -> float:
    """Gain pour 1 EUR en simple gagnant (dividende officiel, sinon dernière cote)."""
    if not row["win"]:
        return 0.0
    return divs.get((row["race_id"], int(row["num"])), row["cote_direct"] or 0.0)


def _metrics(p: np.ndarray, df: pd.DataFrame) -> dict:
    y = df["win"].to_numpy(bool)
    rid = df["race_id"].to_numpy()
    n = df["race_id"].nunique()
    ll = -np.log(np.clip(p[y], 1e-15, 1)).sum() / y.sum()
    tmp = pd.DataFrame({"rid": rid, "p": p, "y": y})
    top = tmp.loc[tmp.groupby("rid")["p"].idxmax()]
    return {"courses": int(n), "logloss": round(float(ll), 4),
            "top1_gagnant_%": round(100 * float(top["y"].mean()), 2)}


def backtest(df: pd.DataFrame, p: np.ndarray, divs, thresholds=(1.0, 1.1, 1.2, 1.3, 1.5, 2.0),
             min_proba: float = 0.03) -> list[dict]:
    d = df.assign(p=p)
    d = d[d["cote_ref"].notna() & (d["cote_ref"] > 1)]
    d["payout"] = d.apply(lambda r: _payout(r, divs), axis=1)
    out = []
    fav = d.loc[d.groupby("race_id")["p"].idxmax()]
    out.append({"strategie": "1 EUR sur le favori du modele", "paris": len(fav),
                "mise": float(len(fav)), "retour": round(float(fav["payout"].sum()), 2),
                "roi_%": round(100 * (fav["payout"].sum() / max(len(fav), 1) - 1), 2)})
    for t in thresholds:
        b = d[(d["p"] * d["cote_ref"] >= t) & (d["p"] >= min_proba)]
        if len(b) == 0:
            continue
        out.append({"strategie": f"value >= {t} (p*cote_matin)", "paris": len(b),
                    "mise": float(len(b)), "retour": round(float(b["payout"].sum()), 2),
                    "roi_%": round(100 * (b["payout"].sum() / len(b) - 1), 2)})
    return out


# --------------------------------------------------------------------- entraînement
def train(con: sqlite3.Connection, test_frac: float = 0.2, verbose: bool = True) -> dict:
    all_df = features.build_features(features.load_runners(con))
    data = _training_set(all_df)
    dates = np.sort(data["date"].unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    tr, te = data[data["date"] < cut], data[data["date"] >= cut]
    divs = _sg_dividends(con)

    tr_o, te_o = tr[tr["has_odds"]], te[te["has_odds"]]
    m_odds = ConditionalLogit(features.FEATURES_WITH_ODDS, name="avec_cotes").fit(tr_o)
    m_base = ConditionalLogit(features.BASE_FEATURES, name="sans_cotes").fit(tr)
    m_only = ConditionalLogit([features.ODDS_FEATURE], name="cotes_seules").fit(tr_o)
    implied = np.exp(te_o[features.ODDS_FEATURE].to_numpy())

    report = {
        "periode_entrainement": [str(pd.Timestamp(tr["date"].min()).date()), str(pd.Timestamp(tr["date"].max()).date())],
        "periode_test": [str(pd.Timestamp(te["date"].min()).date()), str(pd.Timestamp(te["date"].max()).date())],
        "test_courses_avec_cotes": {
            "cotes_brutes_du_matin": _metrics(implied, te_o),
            "cotes_recalibrees": _metrics(m_only.predict_proba(te_o), te_o),
            "modele_avec_cotes": _metrics(m_odds.predict_proba(te_o), te_o),
        },
        "test_toutes_courses": {
            "hasard_uniforme": _metrics(1.0 / te["nb_partants"].to_numpy(float), te),
            "modele_sans_cotes": _metrics(m_base.predict_proba(te), te),
        },
        "backtest_modele_avec_cotes": backtest(te_o, m_odds.predict_proba(te_o), divs),
        "backtest_cotes_seules": backtest(te_o, implied, divs)[:1],
    }
    # Ré-entraînement final sur toutes les données
    m_odds = ConditionalLogit(features.FEATURES_WITH_ODDS, name="avec_cotes").fit(data[data["has_odds"]])
    m_base = ConditionalLogit(features.BASE_FEATURES, name="sans_cotes").fit(data)
    m_odds.save(ODDS_MODEL)
    m_base.save(BASE_MODEL)
    report["coefficients_modele_avec_cotes"] = {k: round(v, 4) for k, v in m_odds.coefficients().items()}
    report["entraine_le"] = datetime.now(TZ).isoformat(timespec="seconds")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if verbose:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# --------------------------------------------------------------------- prédiction
def _fmt_heure(ms) -> str:
    if ms is None or pd.isna(ms):
        return "?"
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%H:%M")


def predict(con: sqlite3.Connection, d: date, refresh: bool = True, notify: bool = False,
            value_threshold: float = 1.2, min_proba: float = 0.05, only_future: bool = True,
            webhook: str | None = None) -> pd.DataFrame:
    if refresh:
        collect.collect_day(con, d, with_dividends=False)
    m_odds, m_base = ConditionalLogit.load(ODDS_MODEL), ConditionalLogit.load(BASE_MODEL)
    all_df = features.build_features(features.load_runners(con))
    day = all_df[all_df["date"] == pd.Timestamp(d)].sort_values(["race_id", "num"]).copy()
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    if only_future:
        day = day[day["heure_depart"].fillna(0) > now_ms]
    day = day[day["nb_partants"] >= 2]
    if day.empty:
        print("Aucune course à prédire.")
        return day
    day["proba"] = np.nan
    for has, model in ((True, m_odds), (False, m_base)):
        sub = day[day["has_odds"] == has]
        if len(sub):
            day.loc[sub.index, "proba"] = model.predict_proba(sub)
    day["rang"] = day.groupby("race_id")["proba"].rank(ascending=False, method="first").astype(int)
    # "value" seulement si le modèle a pu utiliser les cotes de la course
    day["value"] = (day["proba"] * day["cote_ref"]).where(day["has_odds"])
    day = day.sort_values(["heure_depart", "race_id", "num"])
    stamp = datetime.now(TZ).isoformat(timespec="seconds")
    con.executemany(
        "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?)",
        [(r.race_id, int(r.num), float(r.proba), int(r.rang),
          None if pd.isna(r.cote_ref) else float(r.cote_ref),
          None if pd.isna(r.value) else float(r.value), MODEL_VERSION, stamp)
         for r in day.itertuples()])
    con.commit()
    _print_predictions(day, value_threshold, min_proba)
    if notify:
        discord.send(_prediction_embeds(day, value_threshold, min_proba), url=webhook,
                     content=f"🏇 **Pronostics du {d.strftime('%d/%m/%Y')}** "
                             f"({day['race_id'].nunique()} courses)")
    return day


def _is_value(r, thr, min_p) -> bool:
    return pd.notna(r.value) and r.value >= thr and r.proba >= min_p


def _print_predictions(day: pd.DataFrame, thr: float, min_p: float) -> None:
    for rid, g in day.groupby("race_id", sort=False):
        f = g.iloc[0]
        print(f"\n{rid}  {_fmt_heure(f.heure_depart)}  {f.hippodrome}  {f.libelle} "
              f"({f.specialite}, {f.distance} m, {len(g)} partants)")
        for r in g.sort_values("rang").head(5).itertuples():
            cote = f"{r.cote_ref:5.1f}" if pd.notna(r.cote_ref) else "  -  "
            flag = "  <- VALUE" if _is_value(r, thr, min_p) else ""
            if r.rang == 1 and r.proba >= PLACE_MIN and r.has_odds:
                flag += "  <- PLACE suivi"
            print(f"  {r.rang}. n°{r.num:<2} {r.nom:<22} p={100*r.proba:5.1f}%  cote matin {cote}{flag}")


def _prediction_embeds(day: pd.DataFrame, thr: float, min_p: float) -> list[dict]:
    embeds = []
    day = day.assign(reu=day["race_id"].str.extract(r"_(R\d+)")[0])
    day = day.sort_values(["heure_depart", "race_id", "num"])
    for reu, gr in day.groupby("reu", sort=False):
        lines = []
        for rid, g in gr.groupby("race_id", sort=False):
            f = g.iloc[0]
            top = g.sort_values("rang").head(3)
            picks = " · ".join(f"**{r.num}** {r.nom.title()} ({100*r.proba:.0f}%)" for r in top.itertuples())
            vals = [f"{r.num}" for r in g.itertuples() if _is_value(r, thr, min_p)]
            val = f"  💎 value: {', '.join(vals)}" if vals else ""
            fav = top.iloc[0]
            if fav.proba >= PLACE_MIN and fav.has_odds:
                val += f"  🎯 placé: {fav.num}"
            lines.append(f"`{rid.split('_')[1]}` {_fmt_heure(f.heure_depart)} – {picks}{val}")
        f = gr.iloc[0]
        embeds.append({"title": f"{reu} – {f.hippodrome}", "color": discord.COLOR_INFO,
                       "description": "\n".join(lines)})
    return embeds


# --------------------------------------------------------------------- vérification
def check(con: sqlite3.Connection, d: date, refresh: bool = True, notify: bool = True,
          value_threshold: float = 1.2, min_proba: float = 0.05, webhook: str | None = None,
          recheck: bool = False) -> pd.DataFrame:
    if refresh:
        collect.collect_day(con, d, with_dividends=True)
    divs = _sg_dividends(con)
    divs_pl = _sg_dividends(con, "SIMPLE_PLACE")
    odds_races = {r[0] for r in con.execute(
        "SELECT race_id FROM runners WHERE statut='PARTANT' GROUP BY race_id "
        "HAVING AVG(cote_ref IS NOT NULL) >= 0.75")}
    q = """SELECT p.*, ru.nom, ru.ordre_arrivee, ra.heure_depart, ra.hippodrome, ra.libelle,
                  ra.ordre_arrivee AS arrivee
           FROM predictions p JOIN races ra USING(race_id)
           JOIN runners ru ON ru.race_id = p.race_id AND ru.num = p.num
           WHERE ra.date = ? AND ra.categorie_statut = 'ARRIVEE'"""
    pr = pd.read_sql(q, con, params=(d.isoformat(),))
    if not recheck:
        done = {r[0] for r in con.execute("SELECT race_id FROM checks").fetchall()}
        pr = pr[~pr["race_id"].isin(done)]
    if pr.empty:
        print("Aucun nouveau résultat à vérifier.")
        return pr
    rows, lines = [], []
    stamp = datetime.now(TZ).isoformat(timespec="seconds")
    with_divs = {r[0] for r in con.execute(
        "SELECT DISTINCT d.race_id FROM dividends d JOIN races ra USING(race_id) WHERE ra.date = ?",
        (d.isoformat(),))}
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    for rid, g in pr.groupby("race_id", sort=False):
        g = g.sort_values("rang")
        # rapports officiels pas encore publiés : on attend (au plus 6 h après le départ)
        if rid not in with_divs and (g.iloc[0]["heure_depart"] or 0) > now_ms - 6 * 3600 * 1000:
            continue
        arrivee = [x[0] if isinstance(x, list) else x for x in json.loads(g.iloc[0]["arrivee"] or "[]")]
        if not arrivee:
            continue
        winner = int(arrivee[0])
        wrow = g[g["num"] == winner]
        rang_w = int(wrow["rang"].iloc[0]) if len(wrow) else None
        p_w = float(wrow["proba"].iloc[0]) if len(wrow) else None
        top3_pred = list(g["num"].head(3).astype(int))
        top3_hit = len(set(top3_pred) & set(int(x) for x in arrivee[:3]))
        bets = g[(g["value"].fillna(0) >= value_threshold) & (g["proba"] >= min_proba)]
        mise = float(len(bets))
        gain = float(sum(divs.get((rid, int(n)), 0.0) for n in bets["num"] if int(n) == winner))
        fav = g.iloc[0]
        mise_pl = 1.0 if (fav["proba"] >= PLACE_MIN and rid in odds_races) else 0.0
        gain_pl = divs_pl.get((rid, int(fav["num"])), 0.0) if mise_pl else 0.0
        rows.append((rid, stamp, winner, rang_w, p_w, int(rang_w == 1), top3_hit, mise, gain,
                     int(notify), mise_pl, gain_pl))
        icon = "✅" if rang_w == 1 else ("🟡" if rang_w and rang_w <= 3 else "❌")
        nom_w = wrow["nom"].iloc[0].title() if len(wrow) else "?"
        bet_txt = ""
        if mise:
            bet_txt = f" · value {','.join(str(int(n)) for n in bets['num'])} → {gain - mise:+.2f}€"
        if mise_pl:
            bet_txt += (f" · 🎯 placé {int(fav['num'])} " + ("✅" if gain_pl else "❌")
                        + f" {gain_pl - mise_pl:+.2f}€")
        lines.append(f"{icon} `{rid.split('_')[1]}` {g.iloc[0]['hippodrome']} – gagnant **{winner}** "
                     f"{nom_w} (notre rang {rang_w}, p={100*(p_w or 0):.0f}%) · "
                     f"arrivée {'-'.join(str(x) for x in arrivee[:5])} · top3 {top3_hit}/3{bet_txt}")
    con.executemany("""INSERT OR REPLACE INTO checks (race_id, checked_at, gagnant, rang_gagnant,
                       proba_gagnant, top1_ok, top3_dans_arrivee, mise, gain, notified,
                       mise_place, gain_place) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit()
    print("\n".join(lines))
    if notify and lines:
        embeds = []
        for i in range(0, len(lines), 12):
            embeds.append({"title": f"Résultats du {d.strftime('%d/%m/%Y')}",
                           "color": discord.COLOR_OK if any(r[5] for r in rows) else discord.COLOR_KO,
                           "description": "\n".join(lines[i:i + 12])})
        embeds.append(_stats_embed(con, d))
        discord.send(embeds, url=webhook)
    return pr


def stats(con: sqlite3.Connection, day: str | None = None) -> dict:
    q = "SELECT c.*, ra.date FROM checks c JOIN races ra USING(race_id)"
    df = pd.read_sql(q, con)
    if day:
        df = df[df["date"] == day]
    if df.empty:
        return {}
    ll = -np.log(df["proba_gagnant"].fillna(1e-3).clip(1e-6, 1)).mean()
    return {"courses": len(df), "favori_gagnant_%": round(100 * df["top1_ok"].mean(), 1),
            "gagnant_dans_top3_%": round(100 * (df["rang_gagnant"] <= 3).mean(), 1),
            "logloss": round(float(ll), 3), "paris_value": int(df["mise"].sum()),
            "mise": round(float(df["mise"].sum()), 2), "retour": round(float(df["gain"].sum()), 2),
            "benefice": round(float(df["gain"].sum() - df["mise"].sum()), 2),
            "roi_%": round(100 * (df["gain"].sum() / df["mise"].sum() - 1), 1) if df["mise"].sum() else None,
            **_place_stats(df)}


def _place_stats(df: pd.DataFrame) -> dict:
    b = df[df["mise_place"].fillna(0) > 0]
    n = len(b)
    return {"place_paris": n,
            "place_touche_%": round(100 * (b["gain_place"] > 0).mean(), 1) if n else None,
            "place_benefice": round(float(b["gain_place"].sum() - n), 2),
            "place_roi_%": round(100 * (b["gain_place"].sum() / n - 1), 1) if n else None}


def _stats_embed(con: sqlite3.Connection, d: date) -> dict:
    day, tot = stats(con, day=d.isoformat()), stats(con)

    def fmt(s):
        if not s:
            return "—"
        roi = f"{s['roi_%']:+.1f}%" if s["roi_%"] is not None else "—"
        return (f"{s['courses']} courses · favori gagnant {s['favori_gagnant_%']}% · "
                f"gagnant dans top3 {s['gagnant_dans_top3_%']}%\n"
                f"💎 Paris value : {s['paris_value']} · bénéfice {s['benefice']:+.2f}€ (ROI {roi})\n"
                + (f"🎯 Placé favori ≥{int(PLACE_MIN*100)}% : {s['place_paris']} paris · touché "
                   f"{s['place_touche_%']}% · bénéfice {s['place_benefice']:+.2f}€ "
                   f"(ROI {s['place_roi_%']:+.1f}%)" if s["place_paris"] else "🎯 Placé : aucun pari"))
    return {"title": "📊 Bilan", "color": discord.COLOR_INFO,
            "description": f"**Journée du {d.strftime('%d/%m')}** : {fmt(day)}\n\n**Depuis le début** : {fmt(tot)}"}
