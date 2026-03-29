"""
predict_matchups.py  —  Generate model predictions for any CSV of matchups
in the format: favorite, underdog, market_spread, market_over_under

Usage:
    poetry run python3 predict_matchups.py figs/sweet16.csv

Output: <input_stem>_analysis.md written alongside the input CSV.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 predict_matchups.py <matchups.csv>")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / "src" / "2026"
DATA_TSV     = PROJECT_ROOT / "data" / "halftime_odds.tsv"
POSTERIOR    = PROJECT_ROOT / "notebooks" / "2026" / "posterior_2026.npz"
CSV_IN       = (PROJECT_ROOT / sys.argv[1]).resolve()
MD_OUT       = CSV_IN.with_name(CSV_IN.stem + "_analysis.md")

sys.path.insert(0, str(SRC_DIR))

from data_prep  import prepare_data
from model      import basketball_model
from inference  import run_nuts, save_samples, load_samples
from predict    import predict_game

# ── 1.  Data ─────────────────────────────────────────────────────────────────
print("Loading data …")
data       = prepare_data(DATA_TSV)
team_index = data["team_index"]
train      = data["train"]

# ── 2.  Posterior (run if not cached) ────────────────────────────────────────
if POSTERIOR.exists():
    print(f"Loading cached posterior from {POSTERIOR}")
    samples = load_samples(POSTERIOR)
else:
    print("No cache found — running NUTS (4 chains × 1000 samples) …")
    model_args = dict(
        home_id    = train["home_id"].values,
        away_id    = train["away_id"].values,
        is_neutral = train["is_neutral"].values,
        n_teams    = data["n_teams"],
        home_score = train["home_score"].values,
        away_score = train["away_score"].values,
    )
    samples = run_nuts(basketball_model, model_args)
    save_samples(samples, POSTERIOR)

# ── 3.  Load matchups ────────────────────────────────────────────────────────
matchups = pd.read_csv(
    CSV_IN,
    header=None,
    names=["favorite", "underdog", "mkt_spread", "mkt_ou"],
)
matchups["mkt_spread"] = matchups["mkt_spread"].astype(float)
matchups["mkt_ou"]     = matchups["mkt_ou"].astype(float)

# ── 4.  Predict each game ────────────────────────────────────────────────────
results = []

for _, row in matchups.iterrows():
    fav = row["favorite"].strip()
    dog = row["underdog"].strip()
    mkt_spread = row["mkt_spread"]   # negative: fav wins by |mkt_spread|
    mkt_ou     = row["mkt_ou"]

    # Skip teams not in model
    if fav not in team_index.index or dog not in team_index.index:
        print(f"SKIP  {fav} vs {dog}  (team not in model)")
        results.append(None)
        continue

    pred = predict_game(
        basketball_model, samples, team_index,
        fav, dog, neutral_court=True
    )

    spread_samples = -pred["spread"]       # dog_score - fav_score (matches market convention)
    total_samples  = pred["total"]

    model_spread = float(np.median(spread_samples))   # negative = fav wins, positive = upset
    model_ou     = float(np.median(total_samples))
    p_fav        = pred["p_a_wins"]
    p_cover      = float(np.mean(spread_samples < mkt_spread))  # fav covers (both negative)
    p_over       = float(np.mean(total_samples  > mkt_ou))

    results.append({
        "favorite":    fav,
        "underdog":    dog,
        "mkt_spread":  mkt_spread,
        "mkt_ou":      mkt_ou,
        "p_fav":       p_fav,
        "model_spread": model_spread,
        "p_cover":     p_cover,
        "model_ou":    model_ou,
        "p_over":      p_over,
    })

# ── 5.  Build markdown ───────────────────────────────────────────────────────
rows_ok = [r for r in results if r is not None]

round_name = CSV_IN.stem.replace("_", " ").title()
header = (
    f"# 2026 NCAA Tournament — {round_name} Model Predictions\n\n"
    "Model spread convention: same as market — negative = favorite wins by that many, "
    "positive = model thinks underdog wins outright.\n\n"
)

# Summary table
col_w_fav = max(len(r["favorite"]) for r in rows_ok) + 2
col_w_dog = max(len(r["underdog"]) for r in rows_ok) + 2

table_lines = [
    f"## Summary Table\n",
    f"| {'Market Favorite':{col_w_fav}} | {'Underdog':{col_w_dog}} | P(fav) | Model spd | Mkt spd | P(cover) | Model O/U | Mkt O/U | P(over) |",
    f"|{'-'*(col_w_fav+2)}|{'-'*(col_w_dog+2)}|--------|-----------|---------|----------|-----------|---------|---------|",
]

for r in rows_ok:
    # Display model_spread with sign convention matching roundTwo docs:
    # negative = fav wins, positive = underdog better
    ms = r["model_spread"]
    ms_str = f"{ms:.1f}"
    # market spread already negative
    mkt_str = f"{r['mkt_spread']:.1f}"

    table_lines.append(
        f"| {r['favorite']:{col_w_fav}} | {r['underdog']:{col_w_dog}} "
        f"| {r['p_fav']:>6.1%} "
        f"| {ms_str:>9} "
        f"| {mkt_str:>7} "
        f"| {r['p_cover']:>8.1%} "
        f"| {r['model_ou']:>9.0f} "
        f"| {r['mkt_ou']:>7.1f} "
        f"| {r['p_over']:>7.1%} |"
    )

# Notable disagreements
notable = []
for r in rows_ok:
    # Both spreads negative = fav wins; diff > 0 → model less bullish on fav; diff < 0 → more bullish
    r["_diff"] = r["model_spread"] - r["mkt_spread"]

# Sort by biggest disagreement magnitude
rows_sorted = sorted(rows_ok, key=lambda r: abs(r["_diff"]), reverse=True)

notable_lines = ["\n## Notable Disagreements\n"]
threshold = 2.0   # at least 2 pts apart

for r in rows_sorted:
    ms    = r["model_spread"]
    mkt   = abs(r["mkt_spread"])
    diff  = r["_diff"]

    if abs(diff) < threshold:
        continue

    fav, dog = r["favorite"], r["underdog"]
    mkt_line = r["mkt_spread"]   # e.g. -7.5

    if ms > 0:
        # Model picks upset
        notable_lines.append(
            f"- **{fav} {mkt_line:.1f} vs {dog}:** "
            f"Model actually likes {dog} by {ms:.1f}; "
            f"P({fav} covers) = {r['p_cover']:.1%} — model and market disagree on winner"
        )
    elif diff > threshold:
        # model less bullish on fav (ms closer to 0 than mkt)
        notable_lines.append(
            f"- **{fav} {mkt_line:.1f} vs {dog}:** "
            f"Model has {fav} by only {abs(ms):.1f}; "
            f"P(cover) = {r['p_cover']:.1%} — model thinks spread is too large"
        )
    elif diff < -threshold:
        # model more bullish on fav
        notable_lines.append(
            f"- **{fav} {mkt_line:.1f} vs {dog}:** "
            f"Model has {fav} by {abs(ms):.1f}; "
            f"P(cover) = {r['p_cover']:.1%} — model thinks spread is too small"
        )

# Total disagreements
for r in rows_ok:
    ou_diff = r["model_ou"] - r["mkt_ou"]
    if abs(ou_diff) >= 3:
        direction = "over" if ou_diff > 0 else "under"
        fav, dog = r["favorite"], r["underdog"]
        notable_lines.append(
            f"- **{fav} vs {dog} (O/U {r['mkt_ou']:.1f}):** "
            f"Model projects {r['model_ou']:.0f} total; "
            f"P(over) = {r['p_over']:.1%} — model leans {direction}"
        )

md = header + "\n".join(table_lines) + "\n" + "\n".join(notable_lines) + "\n"

# ── 6.  Write & print ────────────────────────────────────────────────────────
MD_OUT.parent.mkdir(parents=True, exist_ok=True)
MD_OUT.write_text(md)
print(f"\nWrote {MD_OUT}")
print("\n" + md)
