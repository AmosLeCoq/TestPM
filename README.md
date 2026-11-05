# PMU Predictor

Système qui :
1. **récupère l'historique** des courses PMU (programme, partants, cotes du matin, arrivées, rapports officiels) dans une base SQLite ;
2. **calcule une probabilité de victoire** pour chaque cheval des courses à venir ;
3. **vérifie les pronostics** une fois les arrivées connues et envoie le bilan sur **Discord** (webhook).

> ⚠️ C'est un outil d'analyse. Il ne place **aucun pari**. Les rapports affichés sont ceux du PMU France ;
> ceux de Loro peuvent différer légèrement. Ne misez jamais plus que ce que vous acceptez de perdre.

## Fonctionnement automatique sur GitHub (sans PC allumé)

Le workflow `.github/workflows/pmu.yml` lance le système sur les serveurs de GitHub :

| Heure (UTC) | Action |
|---|---|
| 09:00 chaque jour | pronostics du jour → Discord |
| toutes les 15 min de 10:00 à 23:45 | vérification des courses terminées → Discord |
| 01:30 chaque nuit | bilan de la veille, collecte des données, ré-entraînement le lundi |

GitHub démarre souvent les tâches programmées avec 5 à 20 min de retard.

**Mise en place (une seule fois)** :
1. *Settings → Secrets and variables → Actions → New repository secret*
   - Name : `DISCORD_WEBHOOK_URL`
   - Secret : l'URL de votre webhook Discord
2. *Actions* → activer les workflows si GitHub le demande.
3. *Actions → PMU Predictor → Run workflow* → `test-webhook` pour vérifier Discord.
4. *Run workflow* → `predict` : le premier lancement recollecte un an d'historique (~1 h), ensuite chaque
   exécution prend 1 à 2 min.

La base de données (`data/pmu.db`) n'est pas dans le dépôt : elle est conservée dans le cache de GitHub Actions.
Si le cache disparaît (inutilisé pendant 7 jours), elle est recollectée automatiquement.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+.

## Utilisation

```bash
# 1. Récupérer l'historique (ici 365 jours ; ~5 s par jour, la commande peut être relancée, elle reprend)
python -m pmu.cli collect --days 365

# 2. Entraîner le modèle (affiche l'évaluation sur la période de test et la sauvegarde dans models/)
python -m pmu.cli train

# 3. Probabilités pour les courses d'aujourd'hui (ou --date demain / 2026-11-06)
python -m pmu.cli predict --date today --notify

# 4. Vérifier les pronostics (courses terminées) et envoyer le bilan sur Discord
python -m pmu.cli check --date today

# Bilan cumulé
python -m pmu.cli stats
```

### Discord

Dans Discord : *Paramètres du salon → Intégrations → Webhooks → Nouveau webhook → Copier l'URL*.

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/...."   # Windows : set DISCORD_WEBHOOK_URL=...
python -m pmu.cli test-webhook
```

(ou passer `--webhook URL` à chaque commande). Sans webhook, les messages sont affichés dans le terminal.

### Mode automatique

```bash
python -m pmu.cli run --predict-hour 10 --every 15
```

- chaque jour à 10 h (heure suisse) : pronostics du jour envoyés sur Discord ;
- toutes les 15 min : vérification des courses terminées → message Discord (✅ notre favori a gagné, 🟡 gagnant dans notre top 3, ❌ raté) + bilan de la journée et cumulé ;
- chaque nuit : collecte complète de la veille ; le lundi : ré-entraînement du modèle.

Il faut le laisser tourner sur une machine allumée (PC, Raspberry Pi, petit serveur). Alternative avec `cron` :

```cron
0 10 * * *   cd /chemin/pmu-predictor && python -m pmu.cli predict --notify
*/15 11-23 * * * cd /chemin/pmu-predictor && python -m pmu.cli check
30 2 * * *   cd /chemin/pmu-predictor && python -m pmu.cli check --date hier && python -m pmu.cli collect --days 2
0 4 * * 1    cd /chemin/pmu-predictor && python -m pmu.cli train
```

## Le modèle

**Logit conditionnel** (l'approche de Bolton & Chapman, reprise par Bill Benter) : chaque cheval reçoit un score
`w · x`, et les probabilités sont le softmax des scores à l'intérieur de la course (elles somment à 100 %).

Variables utilisées (toutes connues **avant** la course, pas de fuite d'information) :

| Variable | Description |
|---|---|
| `lp_odds` | log de la probabilité implicite de la **cote de référence du matin** (normalisée) |
| `mus_*` | musique : moyenne des 5 dernières places, victoires, top 3, dernière course |
| `wr_carriere`, `pr_carriere` | taux de victoire / place sur la carrière (lissés) |
| `log_gain_par_course`, `log_gains_annee` | gains |
| `driver_wr`, `trainer_wr`, `cheval_driver_wr` | taux de victoire du driver/jockey, de l'entraîneur, du couple cheval-driver, calculés **uniquement sur les jours précédents** |
| `recul`, `poids_rel`, `corde_rel` | handicap distance (trot), poids relatif et corde (galop) |
| `age`, `inedit`, `oeilleres`, `deferre4` | divers |

Deux modèles sont entraînés : **avec cotes** (utilisé quand les cotes du matin existent) et **sans cotes** (sinon).

`train` affiche une évaluation sur les 20 % de jours les plus récents (jamais vus à l'entraînement) :
- *logloss* (plus bas = meilleur) et % de courses où notre favori gagne, comparés aux **cotes brutes** ;
- un **backtest** : ROI de « 1 € sur le favori du modèle » et des paris « value » (`proba × cote ≥ seuil`),
  payés au **rapport officiel** du simple gagnant.

Le rapport complet est enregistré dans `models/rapport_entrainement.json`.

### Paris « value »

Un cheval est marqué 💎 *value* si `proba_modèle × cote_du_matin ≥ 1.2` (réglable avec `--value`) et
`proba ≥ 5 %` (`--min-proba`). Attention : la cote finale (pari mutuel) peut être très différente de la cote du matin.

### Suivi du simple placé 🎯

Le système suit aussi (sans parier) la stratégie **« 1 € en simple placé sur notre favori quand sa
probabilité ≥ 40 % »** (réglable : `--place-min 0.45`). Dans le backtest elle touchait ~75 % du temps
pour un ROI d'environ −2 % (marge d'erreur ±5 points). Les pronostics Discord indiquent `🎯 placé: n°X`,
les résultats affichent ✅/❌ et le gain, et le bilan donne le cumul. Une course n'est vérifiée qu'une fois
les rapports officiels publiés.

## Structure

```
pmu/api.py        appels HTTP à l'API turfinfo du PMU
pmu/collect.py    collecte -> SQLite (data/pmu.db)
pmu/features.py   variables explicatives
pmu/model.py      logit conditionnel (numpy, méthode de Newton)
pmu/pipeline.py   train / predict / check / stats
pmu/discord.py    webhook
pmu/scheduler.py  mode automatique
pmu/cli.py        ligne de commande
```
