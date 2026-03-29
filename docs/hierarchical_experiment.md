# Hierarchical Conference Model — Experiment Notes

**Date:** 2026-03-28

---

## Motivation

The flat model (model.py) estimates team-level attack and defense effects with
ZeroSumNormal priors, treating all teams as exchangeable draws from a single
global distribution. This ignores a structural feature of college basketball:
teams play the majority of their games within their own conference, and
conferences vary in overall strength. A Big Ten team playing against strong Big
Ten defenses all season will have its attack rating suppressed relative to a
comparable team in a weaker conference — the model has no mechanism to
"credit" the schedule difficulty at the conference level.

The Gelman multilevel modeling framework (partial pooling, analogous to the
radon-in-houses example) suggested a natural fix: add a conference level to
the hierarchy, so that team effects are drawn from conference-level means,
which are themselves drawn from a global prior. Cross-conference games provide
the data to identify those conference means.

---

## Model Specification

### Flat model (baseline, `src/2026/model.py`)

```
mu         ~ Normal(log(70), 0.3)
home_adv   ~ Normal(0, 0.3)
phi        ~ Gamma(2, 0.1)
sigma_att  ~ HalfNormal(0.5)
sigma_def  ~ HalfNormal(0.5)

att  ~ ZeroSumNormal(sigma_att, n_teams)
def  ~ ZeroSumNormal(sigma_def, n_teams)

log θ_home = mu + home_adv*(1-is_neutral) + att[home] + def[away]
log θ_away = mu +                           att[away] + def[home]
score      ~ NegBin2(exp(θ), phi)
```

### Hierarchical model (`src/2026/model_hierarchical.py`)

Adds a conference level between the global prior and the team effects.
Conference means are ZeroSumNormal (sum to zero globally) for identifiability.
Team effects are Normal centered on their conference mean rather than on zero.

```
sigma_conf_att ~ HalfNormal(0.2)
sigma_conf_def ~ HalfNormal(0.2)

mu_att_conf ~ ZeroSumNormal(sigma_conf_att, n_confs)   # shape (5,)
mu_def_conf ~ ZeroSumNormal(sigma_conf_def, n_confs)

sigma_att  ~ HalfNormal(0.5)
sigma_def  ~ HalfNormal(0.5)

att[t]  ~ Normal(mu_att_conf[conf_id[t]], sigma_att)   # shape (n_teams,)
def[t]  ~ Normal(mu_def_conf[conf_id[t]], sigma_def)

log θ_home = mu + home_adv*(1-is_neutral) + att[home] + def[away]
log θ_away = mu +                           att[away] + def[home]
score      ~ NegBin2(exp(θ), phi)
```

---

## Data

- **Source:** `data/halftime_odds.tsv` — 2025-26 regular season
- **Conferences:** ACC, Big 12, Big East, Big Ten, SEC (79 teams)
- **Games after filtering:** 604 total (both teams in the 5 conferences)
- **Split:** 574 training, 30 holdout (same random seed=42 split for both models)
- **NUTS:** 4 chains × 1000 warmup + 1000 samples

---

## Results

### Holdout evaluation (N=30 games)

| Metric | Flat | Hierarchical | Δ (H−F) |
|---|---|---|---|
| Win accuracy | 0.800 | 0.767 | −0.033 |
| Mean log-score | −1.338 | −1.337 | +0.002 |
| Spread 90% CI coverage | 1.000 | 1.000 | 0.000 |
| Total 90% CI coverage | 0.867 | 0.867 | 0.000 |

*Higher log-score is better. Target for 90% CI coverage is ~0.90.*

### Conference-level posteriors (hierarchical model)

| Conference | Att mean | Att 90% CI | Def mean | Def 90% CI |
|---|---|---|---|---|
| ACC | −0.001 | [−0.025, +0.020] | −0.005 | [−0.026, +0.013] |
| Big 12 | +0.011 | [−0.011, +0.040] | −0.006 | [−0.031, +0.013] |
| Big East | −0.015 | [−0.047, +0.010] | +0.001 | [−0.021, +0.023] |
| Big Ten | −0.014 | [−0.043, +0.007] | −0.002 | [−0.024, +0.017] |
| SEC | +0.019 | [−0.004, +0.049] | +0.012 | [−0.006, +0.040] |

*Att positive = above-average offense. Def negative = above-average defense
(suppresses opponent scoring). All 90% CIs include zero.*

### NUTS diagnostics (hierarchical model)

- **Divergences:** 130 — indicates some geometry issues in the posterior
- **r_hat:** all ≈ 1.00–1.01, chains mixed well
- **sigma_conf_att:** mean 0.03, 90% CI [0.00, 0.06]
- **sigma_conf_def:** mean 0.02, 90% CI [0.00, 0.04]

---

## Interpretation

The two models produce essentially identical predictive performance on the
holdout. The conference-level effects are estimated to be very small
(sigma_conf ≈ 0.02–0.03 on the log scale, vs sigma_att/def ≈ 0.06 at the
team level), and all conference means have 90% CIs that include zero. The
hierarchy adds parameters but does not meaningfully improve out-of-sample
predictions.

The SEC shows the largest estimated offensive and defensive means (+0.019 att,
+0.012 def), and the Big East and Big Ten are below average offensively
(−0.015, −0.014). These directions are plausible but the uncertainty is too
large to be conclusive.

---

## Why the Hierarchical Model Doesn't Help Here

Two structural problems limit the conference-level model with a single season
of data:

**1. Sparse cross-conference data.** Conference means can only be identified
through cross-conference games (~604 games total across 5 conferences ×
10 conference pairs ≈ ~60 games per conference pair on average). This is not
enough to pin down conference-level effects with much precision, especially
relative to the team-level variation already captured by the flat model.

**2. Selection bias in cross-conference games.** Cross-conference games in
college basketball are played predominantly between the *top teams* of each
conference. A top Big Ten team vs a top SEC team tells us about those two teams
specifically, but not about the average Big Ten vs average SEC quality. Some
conferences have a few elite teams and many mediocre ones; others (like the
Big Ten in 2025-26) have many strong teams throughout the standings. The
cross-conference sample is therefore a biased representation of conference
strength, and the conference-level hyperprior inherits this bias.

---

## Recommendation

Use the **flat model** (`model.py`) for 2026 tournament predictions. The team-
level effects are already well-identified within each conference (the
within-conference schedule creates a fully connected graph), and cross-
conference games anchor the conferences to a common scale despite the selection
bias. The hierarchical extension would become worthwhile with 3–4 seasons of
data, where cross-conference matchup graphs become denser and more varied.

The comparison infrastructure (`compare_models.py`) is in place to re-run this
experiment if multi-season data becomes available.

---

## Further Discussion

### Selection bias in cross-conference games

A deeper issue beyond data sparsity: cross-conference games are played
predominantly between the *top teams* of each conference. Some conferences
have a few elite teams and many mediocre ones; others (like the Big Ten in
2025-26) have many strong teams throughout the standings. This means the
cross-conference sample is a biased representation of overall conference
strength, and the conference-level hyperprior inherits that bias. The
conference means we estimate are really "top-of-conference" comparisons,
not "average-team" comparisons.


You're identifying a classic selection bias problem in college sports analytics — it's actually the same fundamental issue that makes the College Football Playoff selection so contentious every year.

  The problem precisely stated: cross-conference games are played between the top teams of each conference, not random samples. So when a top Big Ten team beats a top SEC team, we learn something about those two teams, but we can't cleanly infer "Big Ten is stronger than SEC on average" because we never observe average Big Ten vs average SEC.

  This means the conference-level hyperprior doesn't actually solve what you want it to solve — the cross-conference games that identify mu_att_conf and mu_def_conf are a biased sample of each conference.

  What would actually help:

  1. Multiple seasons of data. With 3-4 seasons, the cross-conference matchup graph becomes much denser and more varied Teams that were great one year drop off, so the sample isn't always "only the elites play OOC." This is the most principled fix.
  2. Strength of schedule as a latent variable per team, not per conference. Rather than a conference-level mean, let each team's prior be informed by a weighted average of their opponents' quality. This is essentially what KenPom/BPI do iteratively — the model is already doing a version of this implicitly, it just doesn't formalize it.
  3. Accept the model's implicit handling. The flat team model is actually reasonably well-identified within each conference — within-conference games create a fully connected graph, so all Big Ten teams are calibrated against each other. Cross-conference games then anchor the conferences to a common scale, selection bias and all. The resulting team ratings are arguably better than conference-level pooling because they don't assume conference membership is informative beyond what the games already reveal.

  The honest conclusion: for a single season of data with this conference structure, the flat model is probably the right tool. The hierarchical extension would pay off with multi-season data. Does that match your intuition, or do you want to explore the multi-season data route?



### KenPom / BPI — external rating systems

Both KenPom and ESPN BPI are proprietary rating systems designed to solve
exactly this problem.

**KenPom (kenpom.com, Ken Pomeroy)**
- Adjusted offensive and defensive efficiency ratings (points per 100
  possessions, adjusted for opponent strength)
- Uses an iterative method: a team's rating depends on its opponents'
  ratings, which depend on *their* opponents' ratings, until convergence
- Widely used by coaches, analysts, and gamblers; considered the gold
  standard for college basketball analytics
- Paid subscription (~$20/year)

**BPI (Basketball Power Index, ESPN)**
- ESPN's version of the same idea
- Also adjusts for opponent strength, location, and game-by-game performance
- Freely available on ESPN

**What they have that our model doesn't:**
1. Possession-level efficiency (points per 100 possessions) rather than raw
   scores — removes pace as a confounder
2. Multiple seasons of data informing priors
3. Pre-season priors based on returning players and recruiting rankings
4. Their "opponent adjustment" is essentially what our model does, but with
   far more data and iteration

**How KenPom/BPI ratings could fit into our model:**
Use their team ratings as informative priors on `att` and `def` instead of
the flat `HalfNormal(0.5)`. The model would start with an informed estimate
and update it with the current season's game results. This would be
particularly powerful early in the season when few games have been played,
and would sidestep the cross-conference identification problem entirely by
leveraging their multi-season, possession-adjusted ratings as a prior.

---

## Files

| File | Description |
|---|---|
| `src/2026/model.py` | Flat model (baseline) |
| `src/2026/model_hierarchical.py` | Two-level hierarchical model |
| `src/2026/data_prep.py` | Added `encode_conferences()`, `prepare_data()` now returns `conf_index`, `team_conf_id`, `n_confs` |
| `src/2026/predict.py` | Added `extra_model_args` param to `predict_game()` and `evaluate_holdout()` |
| `compare_models.py` | Trains both models, evaluates on same holdout, prints comparison table and conference posteriors |
| `notebooks/2026/posterior_flat_2026.npz` | Cached flat posterior (gitignored) |
| `notebooks/2026/posterior_hier_2026.npz` | Cached hierarchical posterior (gitignored) |
