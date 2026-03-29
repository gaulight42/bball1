"""
compare_models.py — Train both the flat and hierarchical basketball models on
the same data split, evaluate each on the holdout set, and print a side-by-side
comparison table.

Usage:
    poetry run python3 compare_models.py

Posteriors are cached so re-runs are fast (~2s each):
    notebooks/2026/posterior_flat_2026.npz
    notebooks/2026/posterior_hier_2026.npz

Delete either .npz file to force a retrain.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / "src" / "2026"
DATA_TSV     = PROJECT_ROOT / "data" / "halftime_odds.tsv"
POSTERIOR_FLAT = PROJECT_ROOT / "notebooks" / "2026" / "posterior_flat_2026.npz"
POSTERIOR_HIER = PROJECT_ROOT / "notebooks" / "2026" / "posterior_hier_2026.npz"

sys.path.insert(0, str(SRC_DIR))

from data_prep  import prepare_data
from model      import basketball_model
from model_hierarchical import basketball_model_hierarchical
from inference  import run_nuts, save_samples, load_samples
from predict    import evaluate_holdout

# ── 1.  Load data (same split for both models) ───────────────────────────────
print("Loading data …")
data           = prepare_data(DATA_TSV)
train          = data["train"]
holdout        = data["holdout"]
team_index     = data["team_index"]
team_conf_id   = data["team_conf_id"]   # int array (n_teams,)
n_confs        = data["n_confs"]
conf_index     = data["conf_index"]

print(f"  {len(train)} training games, {len(holdout)} holdout games")
print(f"  {data['n_teams']} teams, {n_confs} conferences")
print(f"  Conferences: {list(conf_index.index)}\n")

# ── 2.  Common model args ────────────────────────────────────────────────────
base_args = dict(
    home_id    = train["home_id"].values,
    away_id    = train["away_id"].values,
    is_neutral = train["is_neutral"].values,
    home_score = train["home_score"].values,
    away_score = train["away_score"].values,
)

flat_model_args = {**base_args, "n_teams": data["n_teams"]}
hier_model_args = {**base_args, "n_teams": data["n_teams"],
                   "conf_id": team_conf_id, "n_confs": n_confs}

# Extra args passed to predict_game / evaluate_holdout for hierarchical model
hier_extra = {"conf_id": team_conf_id, "n_confs": n_confs}

# ── 3.  Train or load flat model ─────────────────────────────────────────────
if POSTERIOR_FLAT.exists():
    print(f"Loading cached flat posterior …")
    samples_flat = load_samples(POSTERIOR_FLAT)
else:
    print("Training flat model …")
    samples_flat = run_nuts(basketball_model, flat_model_args)
    save_samples(samples_flat, POSTERIOR_FLAT)

# ── 4.  Train or load hierarchical model ─────────────────────────────────────
if POSTERIOR_HIER.exists():
    print(f"Loading cached hierarchical posterior …")
    samples_hier = load_samples(POSTERIOR_HIER)
else:
    print("Training hierarchical model …")
    samples_hier = run_nuts(basketball_model_hierarchical, hier_model_args)
    save_samples(samples_hier, POSTERIOR_HIER)

# ── 5.  Evaluate both on the same holdout ────────────────────────────────────
print("\n--- Flat model ---")
df_flat = evaluate_holdout(basketball_model, samples_flat, holdout, team_index)

print("\n--- Hierarchical model ---")
df_hier = evaluate_holdout(
    basketball_model_hierarchical, samples_hier, holdout, team_index,
    extra_model_args=hier_extra,
)

# ── 6.  Side-by-side comparison table ───────────────────────────────────────
metrics = {
    "Win accuracy":     ("correct",         "{:.3f}"),
    "Mean log-score":   ("log_score",       "{:.3f}"),
    "Spread 90% cov":   ("spread_in_90ci",  "{:.3f}"),
    "Total  90% cov":   ("total_in_90ci",   "{:.3f}"),
}

print("\n" + "="*52)
print(f"  Model Comparison  (N holdout = {len(holdout)})")
print("="*52)
print(f"  {'Metric':<20} {'Flat':>8}   {'Hierarchical':>12}   {'Δ (H−F)':>8}")
print(f"  {'-'*20}  {'-'*8}   {'-'*12}   {'-'*8}")

for label, (col, fmt) in metrics.items():
    v_flat = df_flat[col].mean()
    v_hier = df_hier[col].mean()
    delta  = v_hier - v_flat
    sign   = "+" if delta >= 0 else ""
    print(
        f"  {label:<20} {fmt.format(v_flat):>8}   {fmt.format(v_hier):>12}   "
        f"{sign}{delta:+.3f}"
    )

print("="*52)
print("  Target: spread/total 90% cov ≈ 0.90; higher log-score is better")
print()

# ── 7.  Conference strength posteriors (hierarchical only) ───────────────────
if "mu_att_conf" in samples_hier:
    conf_names = list(conf_index.index)
    mu_att = samples_hier["mu_att_conf"]   # shape (S, n_confs)
    mu_def = samples_hier["mu_def_conf"]

    print("Conference-level posteriors (hierarchical model):")
    print(f"  {'Conference':<12}  {'Att mean':>9}  {'Att 90% CI':>18}  "
          f"{'Def mean':>9}  {'Def 90% CI':>18}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*18}  {'-'*9}  {'-'*18}")
    for i, name in enumerate(conf_names):
        a_mean = float(np.mean(mu_att[:, i]))
        a_lo, a_hi = np.percentile(mu_att[:, i], [5, 95])
        d_mean = float(np.mean(mu_def[:, i]))
        d_lo, d_hi = np.percentile(mu_def[:, i], [5, 95])
        print(
            f"  {name:<12}  {a_mean:>+9.3f}  [{a_lo:+.3f}, {a_hi:+.3f}]  "
            f"{d_mean:>+9.3f}  [{d_lo:+.3f}, {d_hi:+.3f}]"
        )
    print()
    print("  Att: positive = above-average offense for conference")
    print("  Def: negative = above-average defense (suppresses opponent scoring)")
