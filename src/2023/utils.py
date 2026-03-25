"""
utils.py — Visualizations for the 2023 NCAA Basketball model.

Functions:
    plot_team_quality()  — attack vs defense scatter with uncertainty
    plot_spread_ecdf()   — complementary ECDF for spread predictions
    plot_total_ecdf()    — ECDF for over/under predictions
    plot_rhat()          — R-hat convergence diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


def _build_team_quality_df(posterior_samples: dict, team_index: pd.Series) -> pd.DataFrame:
    """
    Compute per-team posterior mean ± std for attack and defense.

    posterior_samples["att"]  shape: (S, T)
    posterior_samples["def"]  shape: (S, T)
    """
    att = posterior_samples["att"]   # (S, T)
    defend = posterior_samples["def"]  # (S, T)

    teams = team_index.index.tolist()
    rows = []
    for i, team in enumerate(teams):
        rows.append(
            {
                "team": team,
                "attack": float(att[:, i].mean()),
                "attack_sd": float(att[:, i].std()),
                "defend": float(defend[:, i].mean()),
                "defend_sd": float(defend[:, i].std()),
                "attack_lo": float(np.percentile(att[:, i], 10)),
                "attack_hi": float(np.percentile(att[:, i], 90)),
                "defend_lo": float(np.percentile(defend[:, i], 10)),
                "defend_hi": float(np.percentile(defend[:, i], 90)),
            }
        )
    return pd.DataFrame(rows)


def plot_team_quality(
    posterior_samples: dict,
    team_index: pd.Series,
    conferences: dict | None = None,
    figsize: tuple = (12, 9),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Scatter plot: attack strength (x) vs defense weakness (y).

    Interpretation:
        High attack, low defense → strong team (upper-left quadrant)
        Low attack, high defense → weak team (lower-right quadrant)

    Parameters
    ----------
    posterior_samples : dict from run_nuts / load_samples
    team_index        : Series mapping team_name → int id
    conferences       : optional dict mapping team_name → conference
                        (used to color-code teams)
    """
    df = _build_team_quality_df(posterior_samples, team_index)

    conf_colors = {
        "ACC": "#003087",
        "Big Ten": "#CC0033",
        "Big 12": "#003366",
        "SEC": "#FF8200",
        "Big East": "#005288",
    }
    default_color = "#888888"

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        color = default_color
        if conferences:
            color = conf_colors.get(conferences.get(row["team"], ""), default_color)

        # 80% CI error bars
        ax.errorbar(
            row["attack"], row["defend"],
            xerr=[[row["attack"] - row["attack_lo"]], [row["attack_hi"] - row["attack"]]],
            yerr=[[row["defend"] - row["defend_lo"]], [row["defend_hi"] - row["defend"]]],
            fmt="o", color=color, alpha=0.6, markersize=4, linewidth=0.8,
        )
        ax.annotate(
            row["team"],
            (row["attack"], row["defend"]),
            fontsize=6.5, alpha=0.85,
            xytext=(3, 2), textcoords="offset points",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Attack strength (log-scale, higher → more points scored)", fontsize=11)
    ax.set_ylabel("Defense weakness (log-scale, higher → more points allowed)", fontsize=11)
    ax.set_title("Team Quality: Attack vs Defense Strength\n(2022-23 season, major conferences)", fontsize=13)

    if conferences:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=conf)
            for conf, c in conf_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_spread_ecdf(
    spread_samples: np.ndarray,
    team_a: str,
    team_b: str,
    actual_spread: float | None = None,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Complementary ECDF of the spread (team_a score − team_b score).

    P(spread > x) on y-axis; x-axis is point differential.
    """
    sorted_spread = np.sort(spread_samples)
    n = len(sorted_spread)
    p_greater = 1.0 - np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(sorted_spread, p_greater, where="post", color="#003087", linewidth=2,
            label=f"P({team_a} wins by > x)")

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.6, label="Pick'em")

    if actual_spread is not None:
        ax.axvline(actual_spread, color="green", linewidth=1.5,
                   linestyle="-", label=f"Actual spread ({actual_spread:+.0f})")

    # Median marker
    median_spread = np.median(spread_samples)
    ax.annotate(
        f"Median: {median_spread:+.1f}",
        xy=(median_spread, 0.5),
        xytext=(median_spread + 2, 0.6),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel(f"Point spread  ({team_a} − {team_b})", fontsize=11)
    ax.set_ylabel("P(spread > x)", fontsize=11)
    ax.set_title(f"Spread Distribution: {team_a} vs {team_b}", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_total_ecdf(
    total_samples: np.ndarray,
    team_a: str,
    team_b: str,
    actual_total: float | None = None,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Complementary ECDF of the total (over/under).

    P(total > x) on y-axis; x-axis is total combined points.
    """
    sorted_total = np.sort(total_samples)
    n = len(sorted_total)
    p_over = 1.0 - np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(sorted_total, p_over, where="post", color="#CC0033", linewidth=2,
            label="P(over x)")

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    if actual_total is not None:
        ax.axvline(actual_total, color="green", linewidth=1.5,
                   linestyle="-", label=f"Actual total ({actual_total:.0f})")

    median_total = np.median(total_samples)
    ax.annotate(
        f"Median: {median_total:.1f}",
        xy=(median_total, 0.5),
        xytext=(median_total + 2, 0.65),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.set_xlabel("Total combined points", fontsize=11)
    ax.set_ylabel("P(total > x)", fontsize=11)
    ax.set_title(f"Over/Under Distribution: {team_a} vs {team_b}", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_rhat(
    posterior_samples: dict,
    figsize: tuple = (8, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart of max R-hat per parameter group.

    Uses a simple split-chain R-hat approximation.
    """
    from numpyro.diagnostics import summary

    # summary() expects shape (num_chains, num_samples, ...)
    # Our samples are already concatenated → reshape assuming equal chains
    # We can compute R-hat manually using numpyro.diagnostics
    try:
        # Attempt proper split-chain R-hat via arviz if available
        import arviz as az
        idata = az.from_dict(posterior_samples)
        rhats = az.rhat(idata)

        param_names = []
        rhat_vals = []
        for var in rhats.data_vars:
            vals = rhats[var].values.flatten()
            param_names.append(var)
            rhat_vals.append(float(np.max(vals)))

    except ImportError:
        # Fallback: rough split-chain R-hat on concatenated samples
        param_names = []
        rhat_vals = []
        for name, arr in posterior_samples.items():
            flat = arr.reshape(arr.shape[0], -1)
            n = flat.shape[0]
            half = n // 2
            chain1 = flat[:half]
            chain2 = flat[half:]
            between = np.var([chain1.mean(0), chain2.mean(0)], axis=0, ddof=1)
            within = (np.var(chain1, axis=0, ddof=1) + np.var(chain2, axis=0, ddof=1)) / 2
            rhat = np.sqrt(1 + between / (within + 1e-10))
            param_names.append(name)
            rhat_vals.append(float(np.max(rhat)))

    colors = ["#CC0033" if r > 1.01 else "#003087" for r in rhat_vals]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(param_names, rhat_vals, color=colors)
    ax.axvline(1.01, color="red", linewidth=1.5, linestyle="--", label="R-hat = 1.01")
    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Max R-hat (lower is better)", fontsize=11)
    ax.set_title("NUTS Convergence Diagnostics", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig
