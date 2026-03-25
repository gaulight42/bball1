"""
inference.py — NUTS sampler runner for the basketball model.

Usage:
    samples = run_nuts(model, model_args)
    save_samples(samples, "posterior.npz")
    samples = load_samples("posterior.npz")
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS


def run_nuts(
    model,
    model_args: dict,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    chain_method: str = "vectorized",
    seed: int = 0,
    progress_bar: bool = True,
) -> dict:
    """
    Run NUTS on the given model.

    Parameters
    ----------
    model       : numpyro model function
    model_args  : dict of keyword arguments to pass to model
    num_warmup  : number of warmup (tuning) steps per chain
    num_samples : number of posterior samples per chain
    num_chains  : number of parallel chains
    chain_method: "vectorized" (recommended for CPU), "parallel", or "sequential"
    seed        : PRNG seed

    Returns
    -------
    samples : dict of {param_name: array of shape (num_chains * num_samples, ...)}
    """
    numpyro.set_host_device_count(num_chains)

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )

    rng_key = jax.random.PRNGKey(seed)

    print(
        f"Running NUTS: {num_chains} chains × "
        f"{num_warmup} warmup + {num_samples} samples …"
    )
    t0 = time.time()
    mcmc.run(rng_key, **model_args)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    mcmc.print_summary(exclude_deterministic=True)

    # Concatenated across chains: shape (num_chains * num_samples, ...)
    # Used for posterior predictive and summary stats.
    samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}
    return samples


def save_samples(samples: dict, path: str | Path) -> None:
    """Save posterior samples to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **samples)
    print(f"Saved samples to {path}")


def load_samples(path: str | Path) -> dict:
    """Load posterior samples from a .npz file."""
    path = Path(path)
    data = np.load(path)
    return {k: data[k] for k in data.files}
