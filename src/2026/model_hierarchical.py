"""
model_hierarchical.py — Two-level Bayesian hierarchical model for NCAA basketball.

Extends model.py by adding conference-level attack/defense hyperpriors.
Teams are partially pooled toward their conference mean; conference means are
themselves drawn from a global ZeroSumNormal (sum-to-zero for identifiability).

Cross-conference games identify how conference means relate to each other,
addressing the problem that teams playing within a strong defensive conference
(e.g. Big Ten) would otherwise have their attack ratings suppressed.

Model:
    # Level 2 — conference means
    mu_att_conf[c] ~ ZeroSumNormal(sigma_conf_att, n_confs)
    mu_def_conf[c] ~ ZeroSumNormal(sigma_conf_def, n_confs)

    # Level 1 — team effects centered on conference mean
    att[t]  ~ Normal(mu_att_conf[conf_id[t]], sigma_att)
    def[t]  ~ Normal(mu_def_conf[conf_id[t]], sigma_def)

    # Likelihood (unchanged from flat model)
    log θ_g1 = mu + home_adv*(1-is_neutral) + att[home] + def[away]
    log θ_g2 = mu +                           att[away] + def[home]
    score    ~ NegBin2(exp(θ), phi)
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def basketball_model_hierarchical(
    home_id,
    away_id,
    is_neutral,
    n_teams: int,
    conf_id,        # int array shape (n_teams,) — conference index for each team
    n_confs: int,
    home_score=None,
    away_score=None,
):
    """
    Two-level hierarchical numpyro model for basketball score prediction.

    Parameters
    ----------
    home_id     : int array (n_games,)
    away_id     : int array (n_games,)
    is_neutral  : int array (n_games,) — 1 = neutral court
    n_teams     : number of teams
    conf_id     : int array (n_teams,) — conference index per team (aligned with team_index)
    n_confs     : number of conferences
    home_score  : int array (n_games,) or None
    away_score  : int array (n_games,) or None
    """
    # ── Global hyperpriors (same as flat model) ─────────────────
    mu       = numpyro.sample("mu",       dist.Normal(jnp.log(70.0), 0.3))
    home_adv = numpyro.sample("home_adv", dist.Normal(0.0, 0.3))
    phi      = numpyro.sample("phi",      dist.Gamma(2.0, 0.1))

    # ── Conference-level variances ───────────────────────────────
    sigma_conf_att = numpyro.sample("sigma_conf_att", dist.HalfNormal(0.2))
    sigma_conf_def = numpyro.sample("sigma_conf_def", dist.HalfNormal(0.2))

    # Conference means — ZeroSumNormal keeps sum=0 for identifiability
    mu_att_conf = numpyro.sample(
        "mu_att_conf",
        dist.ZeroSumNormal(sigma_conf_att, event_shape=(n_confs,)),
    )
    mu_def_conf = numpyro.sample(
        "mu_def_conf",
        dist.ZeroSumNormal(sigma_conf_def, event_shape=(n_confs,)),
    )

    # ── Team-level variances ─────────────────────────────────────
    sigma_att = numpyro.sample("sigma_att", dist.HalfNormal(0.5))
    sigma_def = numpyro.sample("sigma_def", dist.HalfNormal(0.5))

    # Team effects centered on their conference mean
    with numpyro.plate("teams", n_teams):
        att    = numpyro.sample("att", dist.Normal(mu_att_conf[conf_id], sigma_att))
        defend = numpyro.sample("def", dist.Normal(mu_def_conf[conf_id], sigma_def))

    # ── Expected log-scores ──────────────────────────────────────
    home_effect = home_adv * (1.0 - is_neutral.astype(float))

    log_theta1 = mu + home_effect + att[home_id] + defend[away_id]
    log_theta2 = mu +               att[away_id] + defend[home_id]

    mu1 = jnp.exp(log_theta1)
    mu2 = jnp.exp(log_theta2)

    # ── Likelihood ───────────────────────────────────────────────
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
