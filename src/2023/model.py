"""
model.py — Bayesian hierarchical model for NCAA basketball scores.

Adapted from Baio & Blangiardo (2010) for basketball:
  - NegativeBinomial2 likelihood (overdispersion vs. Poisson)
  - ZeroSumNormal priors on attack/defense for identifiability
  - Neutral-court mode: set is_neutral=1 → home_adv drops out

Model:
    log θ_g1 = home_adv * (1 - is_neutral) + att[home] + def[away]
    log θ_g2 =                                att[away] + def[home]
    score_g1 ~ NegBin2(mean=exp(θ_g1), concentration=phi)
    score_g2 ~ NegBin2(mean=exp(θ_g2), concentration=phi)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints


def basketball_model(
    home_id,
    away_id,
    is_neutral,
    n_teams: int,
    home_score=None,
    away_score=None,
):
    """
    Numpyro model for basketball score prediction.

    Parameters
    ----------
    home_id     : int array (n_games,) — index of home team (or team A for neutral)
    away_id     : int array (n_games,) — index of away team (or team B for neutral)
    is_neutral  : int array (n_games,) — 1 if neutral court, 0 if home/away
    n_teams     : number of teams
    home_score  : int array (n_games,) or None — observed home scores
    away_score  : int array (n_games,) or None — observed away scores
    """
    # ── Hyperpriors ────────────────────────────────────────────
    # mu: baseline log scoring rate. log(70) ≈ 4.25 (typical team scores ~70 pts)
    # Without mu, ZeroSumNormal (att+def both sum to 0) centers at exp(0)=1 pt.
    mu         = numpyro.sample("mu",        dist.Normal(jnp.log(70.0), 0.3))
    sigma_att  = numpyro.sample("sigma_att", dist.HalfNormal(0.5))
    sigma_def  = numpyro.sample("sigma_def", dist.HalfNormal(0.5))
    home_adv   = numpyro.sample("home_adv",  dist.Normal(0.0, 0.3))

    # NegBin2 concentration: higher phi → less overdispersion (→ Poisson as phi→∞)
    # Basketball var/mean ≈ 2.0, so phi ≈ mean/(var/mean - 1) ≈ 70
    # Gamma(2, 0.1) has mean=20, allows wide range
    phi = numpyro.sample("phi", dist.Gamma(2.0, 0.1))

    # ── Team parameters (sum-to-zero for identifiability) ──────
    att = numpyro.sample(
        "att",
        dist.ZeroSumNormal(sigma_att, event_shape=(n_teams,)),
    )
    defend = numpyro.sample(
        "def",
        dist.ZeroSumNormal(sigma_def, event_shape=(n_teams,)),
    )

    # ── Expected log-scores ────────────────────────────────────
    home_effect = home_adv * (1.0 - is_neutral.astype(float))

    log_theta1 = mu + home_effect + att[home_id] + defend[away_id]
    log_theta2 = mu +               att[away_id] + defend[home_id]

    mu1 = jnp.exp(log_theta1)
    mu2 = jnp.exp(log_theta2)

    # ── Likelihood ─────────────────────────────────────────────
    # NegBin2: P(k|mu, phi) = NB(r=phi, p=phi/(phi+mu))
    # var = mu + mu^2/phi
    with numpyro.plate("games", len(home_id)):
        numpyro.sample(
            "home_score",
            dist.NegativeBinomial2(mean=mu1, concentration=phi),
            obs=home_score,
        )
        numpyro.sample(
            "away_score",
            dist.NegativeBinomial2(mean=mu2, concentration=phi),
            obs=away_score,
        )
