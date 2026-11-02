"""Sample size and power for win-ratio designs."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def sample_size_win_ratio(
    wr: float,
    p_tie: float = 0.0,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> Dict[str, Any]:
    """Total sample size for a two-sided test of ``WR = 1`` (Yu and Ganju, 2022).

    ``N = 4 (1 + p_tie) (z_{1-alpha/2} + z_{power})^2 / (3 k (1 - k) (1 - p_tie) log(WR)^2)``

    where ``k`` is the proportion allocated to arm A. The formula assumes an
    unmatched (all-pairs) comparison and approximates the null variance of
    the win ratio; check it by simulation (:func:`simulate_power`) when the
    hierarchy has many ties or censoring.

    Parameters
    ----------
    wr : float
        Anticipated win ratio (> 0, != 1).
    p_tie : float, default 0
        Anticipated proportion of tied pairs, in [0, 1).
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.8
        Target power.
    allocation : float, default 0.5
        Proportion of patients in arm A.

    Returns
    -------
    dict
        ``n_total``, ``n_a``, ``n_b`` (rounded up) and the inputs.

    Examples
    --------
    >>> sample_size_win_ratio(1.5, p_tie=0.2)["n_total"]
    382
    """

    if not wr > 0 or wr == 1:
        raise ValueError("wr must be positive and different from 1")
    if not 0 <= p_tie < 1:
        raise ValueError("p_tie must be in [0, 1)")
    if not 0 < allocation < 1:
        raise ValueError("allocation must be in (0, 1)")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    sigma2 = 4.0 * (1.0 + p_tie) / (3.0 * (1.0 - p_tie))
    n_total = sigma2 * z**2 / (allocation * (1 - allocation) * math.log(wr) ** 2)
    n_total = int(math.ceil(n_total))
    n_a = int(math.ceil(allocation * n_total))
    n_b = int(math.ceil((1 - allocation) * n_total))
    return {
        "n_total": n_a + n_b,
        "n_a": n_a,
        "n_b": n_b,
        "wr": wr,
        "p_tie": p_tie,
        "alpha": alpha,
        "power": power,
        "allocation": allocation,
        "method": "Yu and Ganju (2022)",
    }


def simulate_power(
    generate: Callable[[np.random.Generator], pd.DataFrame],
    analysis: Any,
    *,
    n_sim: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Monte Carlo power of a win-ratio analysis.

    Parameters
    ----------
    generate : callable
        ``generate(rng) -> DataFrame`` simulating one trial.
    analysis : WinRatio
        The analysis to apply to each simulated trial.
    n_sim : int, default 500
        Number of simulated trials.
    alpha : float, default 0.05
        Two-sided significance level.
    seed : int, default 42
        Seed of the simulation.
    **fit_kwargs
        Passed to ``analysis.fit`` (default analytic inference).

    Returns
    -------
    dict
        ``power`` (share of trials with p < alpha), its Monte Carlo standard
        error, the mean and median win ratio, and ``n_sim``.

    Examples
    --------
    >>> from pywinratio import Outcome, WinRatio, simulate_power
    >>> from pywinratio.datasets import make_heart_failure_trial
    >>> analysis = WinRatio([Outcome.time_to_event("time_hosp", "hosp")],
    ...                     group="arm", treatment="treatment", control="placebo")
    >>> sim = simulate_power(lambda rng: make_heart_failure_trial(
    ...     n=200, seed=int(rng.integers(1 << 31))), analysis, n_sim=20)
    >>> 0 <= sim["power"] <= 1
    True
    """

    rng = np.random.default_rng(seed)
    p_values = np.empty(n_sim)
    wrs = np.empty(n_sim)
    for i in range(n_sim):
        result = analysis.fit(generate(rng), alpha=alpha, **fit_kwargs)
        p_values[i] = result.p_value
        wrs[i] = result.wr
    rejected = np.nan_to_num(p_values, nan=1.0) < alpha
    power = float(rejected.mean())
    finite = wrs[np.isfinite(wrs)]
    return {
        "power": power,
        "mc_se": float(np.sqrt(power * (1 - power) / n_sim)),
        "mean_wr": float(finite.mean()) if finite.size else float("nan"),
        "median_wr": float(np.median(finite)) if finite.size else float("nan"),
        "n_sim": int(n_sim),
        "alpha": alpha,
    }
