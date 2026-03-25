"""
predict.py — Posterior predictive inference for tournament matchups.

Key functions:
    predict_game()     — neutral-court score/spread/total/win-prob distributions
    evaluate_holdout() — calibration metrics on held-out regular-season games
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import Predictive


def _make_model_args(home_id, away_id, is_neutral, n_teams):
    return dict(
        home_id=np.array([home_id]),
        away_id=np.array([away_id]),
        is_neutral=np.array([is_neutral]),
        n_teams=n_teams,
    )


def predict_game(
    model,
    posterior_samples: dict,
    team_index: pd.Series,
    team_a: str,
    team_b: str,
    neutral_court: bool = True,
    rng_seed: int = 1,
) -> dict:
    """
    Draw posterior predictive scores for a single matchup.

    In neutral-court mode (default), team_a is treated as the "home" slot
    for bookkeeping only — home_adv is zeroed out via is_neutral=1.

    Parameters
    ----------
    model              : numpyro model function
    posterior_samples  : dict from run_nuts / load_samples
    team_index         : Series mapping team name → integer id
    team_a             : name of team A (e.g. "Connecticut")
    team_b             : name of team B (e.g. "Kansas")
    neutral_court      : if True, home_adv is suppressed
    rng_seed           : JAX PRNG seed for predictive sampling

    Returns
    -------
    dict with keys:
        score_a, score_b : arrays of shape (S,) — sampled scores
        spread           : score_a - score_b (positive → team_a favoured)
        total            : score_a + score_b
        p_a_wins         : float, posterior P(team_a wins)
        percentiles      : nested dict with 5/25/50/75/95th percentiles
    """
    if team_a not in team_index.index:
        raise ValueError(f"Unknown team: {team_a!r}")
    if team_b not in team_index.index:
        raise ValueError(f"Unknown team: {team_b!r}")

    id_a = int(team_index[team_a])
    id_b = int(team_index[team_b])
    n_teams = len(team_index)
    is_neutral = 1 if neutral_court else 0

    model_args = _make_model_args(id_a, id_b, is_neutral, n_teams)

    predictive = Predictive(model, posterior_samples=posterior_samples)
    rng_key = jax.random.PRNGKey(rng_seed)
    preds = predictive(rng_key, **model_args)

    score_a = np.array(preds["home_score"]).flatten()
    score_b = np.array(preds["away_score"]).flatten()
    spread = score_a - score_b
    total = score_a + score_b

    pcts = [5, 25, 50, 75, 95]

    def _pcts(arr):
        return {p: float(np.percentile(arr, p)) for p in pcts}

    return {
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "spread": spread,
        "total": total,
        "p_a_wins": float(np.mean(score_a > score_b)),
        "p_b_wins": float(np.mean(score_b > score_a)),
        "percentiles": {
            "score_a": _pcts(score_a),
            "score_b": _pcts(score_b),
            "spread": _pcts(spread),
            "total": _pcts(total),
        },
    }


def evaluate_holdout(
    model,
    posterior_samples: dict,
    holdout: pd.DataFrame,
    team_index: pd.Series,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate posterior predictive calibration on held-out games.

    For each holdout game, compute:
        - predicted win (team with higher median score)
        - log-score = log p(actual scores | posterior predictive)
        - whether actual spread is inside 90% CI
        - whether actual total is inside 90% CI

    Returns a DataFrame with one row per holdout game, plus summary stats.
    """
    n_teams = len(team_index)
    results = []

    for _, row in holdout.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        actual_a = int(row["home_score"])
        actual_b = int(row["away_score"])
        is_neutral = int(row["is_neutral"])

        pred = predict_game(
            model,
            posterior_samples,
            team_index,
            team_a,
            team_b,
            neutral_court=(is_neutral == 1),
            rng_seed=rng_seed,
        )

        score_a = pred["score_a"]
        score_b = pred["score_b"]

        # Predicted winner = team with higher median score
        pred_winner = team_a if np.median(score_a) >= np.median(score_b) else team_b
        actual_winner = team_a if actual_a > actual_b else team_b
        correct = pred_winner == actual_winner

        # Log-score: fraction of samples matching actual scores (rough proxy)
        # More precisely: mean log p(obs | posterior) ≈ log(fraction of
        # samples within ±3 of actual), but use normal kernel approximation
        log_score_a = np.log(np.mean(np.abs(score_a - actual_a) <= 5) + 1e-8)
        log_score_b = np.log(np.mean(np.abs(score_b - actual_b) <= 5) + 1e-8)
        log_score = (log_score_a + log_score_b) / 2

        # 90% CI coverage
        spread_lo, spread_hi = np.percentile(pred["spread"], [5, 95])
        total_lo, total_hi = np.percentile(pred["total"], [5, 95])
        actual_spread = actual_a - actual_b
        actual_total = actual_a + actual_b

        results.append(
            {
                "home_team": team_a,
                "away_team": team_b,
                "actual_home": actual_a,
                "actual_away": actual_b,
                "pred_home_median": float(np.median(score_a)),
                "pred_away_median": float(np.median(score_b)),
                "pred_winner": pred_winner,
                "actual_winner": actual_winner,
                "correct": correct,
                "p_home_wins": pred["p_a_wins"],
                "log_score": float(log_score),
                "spread_lo_90": float(spread_lo),
                "spread_hi_90": float(spread_hi),
                "actual_spread": actual_spread,
                "spread_in_90ci": bool(spread_lo <= actual_spread <= spread_hi),
                "total_lo_90": float(total_lo),
                "total_hi_90": float(total_hi),
                "actual_total": actual_total,
                "total_in_90ci": bool(total_lo <= actual_total <= total_hi),
            }
        )

    df = pd.DataFrame(results)

    print("\n=== Holdout Evaluation ===")
    print(f"  N games:          {len(df)}")
    print(f"  Win accuracy:     {df['correct'].mean():.3f}  (baseline 0.500)")
    print(f"  Mean log-score:   {df['log_score'].mean():.3f}")
    print(f"  Spread 90% cov:   {df['spread_in_90ci'].mean():.3f}  (target ~0.90)")
    print(f"  Total  90% cov:   {df['total_in_90ci'].mean():.3f}  (target ~0.90)")

    return df


def print_matchup(result: dict) -> None:
    """Pretty-print a predict_game() result."""
    a = result["team_a"]
    b = result["team_b"]
    pa = result["percentiles"]

    print(f"\n{'='*50}")
    print(f"  {a} vs {b}  (neutral court)")
    print(f"{'='*50}")
    print(f"  P({a} wins): {result['p_a_wins']:.1%}")
    print(f"  P({b} wins): {result['p_b_wins']:.1%}")
    print()
    print(f"  Scores (5th / 25th / median / 75th / 95th):")
    for team, key in [(a, "score_a"), (b, "score_b")]:
        p = pa[key]
        print(
            f"    {team:30s}: "
            f"{p[5]:.0f} / {p[25]:.0f} / {p[50]:.0f} / {p[75]:.0f} / {p[95]:.0f}"
        )
    print()
    p = pa["spread"]
    print(
        f"  Spread ({a} − {b}):  "
        f"median {p[50]:.1f}  90% CI [{p[5]:.1f}, {p[95]:.1f}]"
    )
    p = pa["total"]
    print(
        f"  Total points:  "
        f"median {p[50]:.1f}  90% CI [{p[5]:.1f}, {p[95]:.1f}]"
    )
    print(f"{'='*50}")
