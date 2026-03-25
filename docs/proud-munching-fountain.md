# NCAA Basketball Score Prediction — 2026 Tournament

## Context
Implement the Baio & Blangiardo (2010) Bayesian hierarchical model adapted for NCAA basketball, scoped to the **2026 NCAA Tournament**. Train on 2022-23 regular season data from 5 major conferences (ACC, Big Ten, Big 12, SEC, Big East — ~78 teams). All tournament predictions use neutral-court mode (`home_adv` zeroed out). Goal: produce posterior distributions over scores, spreads, and totals for any tournament matchup.

Prior 2022 implementation exists in `bball1/notebooks/2022/ncaaBball2022v3.ipynb` (Pyro SVI, 50 teams). This new implementation migrates to numpyro NUTS and scopes to tournament-relevant teams.

**Inference: numpyro NUTS** — ~78 teams → ~160 parameters. PyVBMC's D≤20 limit makes it unsuitable even for this reduced scope. numpyro 0.17 + JAX 0.5.2 are already installed. Estimated runtime: ~5–10 min on M4 Pro for 4 chains × 1000 warmup + 1000 samples.

---

## Model Specification

**Likelihood:** `NegativeBinomial2` (not Poisson) — basketball scores have `var/mean ≈ 2.0`, making Poisson too narrow for spread/total predictions.

```
log θ_g1 = home_adv * (1 - is_neutral) + att[home_id[g]] + def[away_id[g]]
log θ_g2 =                                att[away_id[g]] + def[home_id[g]]
score_g1 ~ NegBin2(mean=exp(θ_g1), concentration=phi)
score_g2 ~ NegBin2(mean=exp(θ_g2), concentration=phi)
```

**For tournament predictions:** `is_neutral=1` for all games → `home_adv` drops out.

**Parameters:**

| Parameter | Prior | Meaning |
|---|---|---|
| `sigma_att` | `HalfNormal(0.5)` | SD of attack effects across teams |
| `sigma_def` | `HalfNormal(0.5)` | SD of defense effects across teams |
| `home_adv` | `Normal(0, 0.3)` | Global log-scale home advantage (training only) |
| `phi` | `Gamma(2, 0.1)` | NegBin2 concentration (overdispersion) |
| `att[T]` | `ZeroSumNormal(sigma_att)` | Per-team attack strength (sum-to-zero) |
| `def[T]` | `ZeroSumNormal(sigma_def)` | Per-team defense strength (sum-to-zero) |

`ZeroSumNormal` (numpyro 0.17) enforces identifiability natively — NUTS samples in unconstrained (T-1)-dimensional space automatically.

---

## Data Scope

**Source:** `data/2023Data.csv` (11,339 rows, 2022-23 season)

**Filter to 5 major conferences:**
- ACC, Big Ten, Big 12, SEC, Big East
- Keep only games where **both** teams are in these conferences
- ~78 teams, roughly 1,000–1,500 inter-conference games after deduplication

**Parse + deduplicate:**
1. Strip outer double-quotes (R-export style CSV)
2. Keep D1 games only (`D1==2`), exclude canceled/postponed
3. Deduplicate: keep `location=="H"` rows as canonical home-game view; keep one row per sorted pair for `location=="N"` games
4. Hold out 30 random games (ensuring all holdout teams appear in training)

---

## File Structure

All new files live inside the existing `bball1/` Poetry project:

```
bball1/
├── src/2023/
│   ├── data_prep.py     # CSV parsing, conference filter, dedup, train/holdout split
│   ├── model.py         # numpyro model definition
│   ├── inference.py     # NUTS runner, save/load posterior samples
│   ├── predict.py       # posterior predictive, spread/ou/win-prob, neutral-court API
│   └── utils.py         # attack/defense scatter, spread/ou ECDF plots
└── notebooks/2023/
    └── ncaaBball2023.ipynb  # end-to-end: EDA → inference → diagnostics → predictions
```

---

## Critical Files

| File | Role |
|---|---|
| `data/2023Data.csv` | Source — needs double-quote parser + conference filter + dedup |
| `bball1/pyproject.toml` | Poetry config — numpyro 0.17 + JAX 0.5.2 already present |
| `bball1/notebooks/2022/ncaaBball2022v3.ipynb` | Prior implementation to adapt (Pyro SVI → numpyro NUTS) |
| `bball1/src/2022/utils.py` | `plot_quality2()` pattern to reuse for attack/defense scatter |

---

## Key Functions

### `data_prep.py`
- `load_raw(path)` — parse double-quoted CSV
- `filter_to_conferences(df, conferences)` — keep games where both teams are in target conferences
- `build_game_table(df)` — deduplicate to one row per game
- `encode_teams(games)` → `(games_with_ids, team_index)` — string names to integers
- `train_holdout_split(games, holdout_n=30, seed=42)`

### `model.py`
- `basketball_model(home_id, away_id, is_neutral, n_teams, home_score=None, away_score=None)` — full numpyro model

### `inference.py`
- `run_nuts(model, model_args, num_warmup=1000, num_samples=1000, num_chains=4)` — `chain_method="vectorized"` for CPU JAX
- `save_samples(samples, path)` / `load_samples(path)` — `.npz` for cached reuse

### `predict.py`
- `predict_game(model, posterior_samples, team_index, team_a, team_b, neutral_court=True)` → `{score_a, score_b, spread, total, p_a_wins, percentiles}`
- `evaluate_holdout(model, posterior_samples, holdout_games, team_index)` → metrics DataFrame

---

## Implementation Sequence

1. `data_prep.py` — parse, filter to 5 conferences, deduplicate, encode teams, split
2. `model.py` — define model; verify with `numpyro.render_model()`
3. `inference.py` — NUTS runner; smoke-test on ACC-only subset first (~17 teams, ~38 params, ~1 min)
4. `predict.py` — `predict_game()` for neutral-court predictions, `evaluate_holdout()` for calibration
5. `utils.py` — attack/defense scatter, spread/ou ECDF plots
6. `ncaaBball2023.ipynb` — EDA, inference, diagnostics, example tournament matchup predictions

---

## Validation

For 30 held-out regular season games:
- **Win accuracy** vs. 50% baseline
- **Log-score:** mean log p(actual scores | posterior predictive)
- **Spread calibration:** actual spreads vs. predicted spread CDF
- **Total coverage:** % of actuals inside 90% credible interval (target ~90%)

## Tournament Prediction Output

For any two teams (e.g., UConn vs. Kansas):
```python
result = predict_game(model, samples, team_index,
                      "Connecticut", "Kansas", neutral_court=True)
# → spread distribution, total distribution, P(UConn wins), score percentiles
```
