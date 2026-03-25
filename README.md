# bball2026 — NCAA Tournament Bayesian Score Predictions

Hierarchical Bayesian model (Baio & Blangiardo, NegBin2 + NUTS via numpyro) trained on
2025-26 regular season inter-conference games. Predicts neutral-court scores, spreads,
and totals for tournament matchups.

---

## Running predictions

```bash
cd /Users/marclight/projects/bball2026/bball1
poetry run python3 predict_matchups.py figs/<round>.csv
```

**Input CSV format** — no header, four columns:

```
favorite team name, underdog team name, market spread, market over/under
```

Example (`figs/sweet16.csv`):
```
Purdue Boilermakers, Texas Longhorns, -7.5, 148.5
Duke Blue Devils, St. John's Red Storm, -6.5, 142.5
```

- Market spread convention: negative = favorite wins by that many (standard sportsbook format)
- Team names must match the full "Team Mascot" format used in `data/halftime_odds.tsv`
  (e.g. "UConn Huskies", not "Connecticut" or "UConn")
- Teams not in the model (mid-majors outside the 5 major conferences) are skipped with a warning

**Output:** `figs/<round>_analysis.md` — summary table + notable disagreements, printed to shell and written to disk.

---

## Virtual environment

```bash
poetry install        # first time only
poetry run python3 predict_matchups.py figs/<round>.csv
```

To recreate with pip instead:

```bash
pip install -r requirements.txt
```

---

## Posterior cache

The first run trains the model (4-chain NUTS, ~25s on M-series Mac) and saves the
posterior to:

```
notebooks/2026/posterior_2026.npz
```

Subsequent runs load the cache and complete in ~2s. Delete the `.npz` to retrain.

---

## Project structure

```
bball1/                          # git root
├── predict_matchups.py          # main script — takes any round CSV, outputs analysis MD
├── requirements.txt             # core dependencies
├── pyproject.toml               # poetry env (python 3.13, jax 0.4.38, numpyro 0.17)
├── figs/
│   ├── sweet16.csv              # input: favorite, underdog, mkt_spread, mkt_ou
│   ├── sweet16_analysis.md      # output: summary table + notable disagreements
│   ├── firstRound/              # spread/total PNGs from round 1
│   └── secondRound/             # spread/total PNGs from round 2
├── data/
│   ├── halftime_odds.tsv        # 2025-26 season game data (source of truth)
│   └── 2023Data.csv
├── docs/
│   ├── roundOne2026.md          # archived round 1 predictions
│   ├── roundTwo2026.md          # archived round 2 predictions
│   └── ...
├── notebooks/2026/
│   ├── ncaaBball2026.ipynb
│   └── posterior_2026.npz       # cached posterior (generated on first run, gitignored)
└── src/2026/
    ├── model.py                 # numpyro model definition
    ├── inference.py             # NUTS runner + save/load
    ├── predict.py               # predict_game(), evaluate_holdout()
    └── data_prep.py             # load/filter/encode pipeline
```

---

## Conferences in model (2025-26 rosters)

ACC, Big Ten (+ UCLA/USC/Oregon/Washington), Big 12 (+ AZ/ASU/BYU/Colo/Utah/Cinci/Houston/UCF),
SEC (+ Oklahoma/Texas), Big East — ~79 teams total.
