"""Shared bootstrap machinery.

Every nonparametric bootstrap in pywinratio reduces to resampling *units*
(patients, assigned pairs or clusters) with replacement inside *cells*
(e.g. arm x stratum) and recomputing weighted win and loss totals. Because
the pairwise comparison does not depend on how often a patient is drawn,
the win/loss indicator matrices are computed once; each replicate only
re-weights them by the resampling frequencies, which is exact and fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def extended_quantile(samples: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile that tolerates infinite values.

    Agrees with ``numpy.quantile(..., method="linear")`` on finite data;
    interpolating towards an infinite order statistic returns that infinity.
    NaN values are ignored.
    """

    x = np.sort(np.asarray(samples, dtype=float)[~np.isnan(samples)])
    n = x.size
    if n == 0:
        return float("nan")
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi or x[lo] == x[hi]:
        return float(x[lo])
    if np.isinf(x[lo]) or np.isinf(x[hi]):
        return float(x[hi]) if x[hi] == np.inf else float(x[lo])
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """Equal-tailed percentile interval with :func:`extended_quantile`."""

    return (extended_quantile(samples, alpha / 2), extended_quantile(samples, 1 - alpha / 2))


@dataclass
class ResamplingDesign:
    """Which units are resampled, and inside which cells.

    Parameters
    ----------
    unit_of : ndarray of int, shape (n_patients,)
        Resampling unit of each patient (0 .. n_units-1).
    cell_of_unit : ndarray of int, shape (n_units,)
        Cell of each unit; units are drawn with replacement within cells so
        cell sizes (in units) are preserved.
    """

    unit_of: np.ndarray
    cell_of_unit: np.ndarray

    def frequencies(self, n_boot: int, rng: np.random.Generator) -> np.ndarray:
        """Draw patient frequencies of shape ``(n_patients, n_boot)``."""

        n_units = self.cell_of_unit.size
        unit_freq = np.zeros((n_units, n_boot), dtype=float)
        for cell in np.unique(self.cell_of_unit):
            members = np.flatnonzero(self.cell_of_unit == cell)
            size = members.size
            draws = rng.multinomial(size, np.full(size, 1.0 / size), size=n_boot).T
            unit_freq[members] = draws
        return unit_freq[self.unit_of]


def design_from_labels(units: Sequence[Any], cells: Sequence[Any]) -> ResamplingDesign:
    """Build a :class:`ResamplingDesign` from per-patient unit and cell labels.

    A unit must belong to a single cell.
    """

    unit_codes, unit_of = np.unique(np.asarray([str(u) for u in units]), return_inverse=True)
    cell_labels = np.asarray([str(c) for c in cells])
    cell_of_unit = np.empty(unit_codes.size, dtype=object)
    for code in range(unit_codes.size):
        values = np.unique(cell_labels[unit_of == code])
        if values.size != 1:
            raise ValueError(
                f"resampling unit {unit_codes[code]!r} spans several cells {values.tolist()}; "
                "use a design where each unit belongs to one arm/stratum"
            )
        cell_of_unit[code] = values[0]
    _, cell_codes = np.unique(cell_of_unit.astype(str), return_inverse=True)
    return ResamplingDesign(unit_of=unit_of, cell_of_unit=cell_codes)


@dataclass
class Block:
    """Win/loss indicator matrices (as floats) of one stratum and patient positions."""

    wins: np.ndarray
    losses: np.ndarray
    a_pos: np.ndarray
    b_pos: np.ndarray
    a_weights: np.ndarray
    b_weights: np.ndarray


def replicate_totals(
    blocks: Sequence[Block],
    design: ResamplingDesign,
    n_boot: int,
    rng: np.random.Generator,
    *,
    batch_cells: int = 40_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted wins, losses and pair totals in each bootstrap replicate."""

    n_patients = design.unit_of.size
    wins = np.zeros(n_boot)
    losses = np.zeros(n_boot)
    totals = np.zeros(n_boot)
    batch = max(1, min(n_boot, batch_cells // max(n_patients, 1)))
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot - start)
        freq = design.frequencies(size, rng)
        for blk in blocks:
            ua = freq[blk.a_pos] * blk.a_weights[:, None]
            ub = freq[blk.b_pos] * blk.b_weights[:, None]
            wins[start : start + size] += np.einsum("ir,ir->r", ua, blk.wins @ ub)
            losses[start : start + size] += np.einsum("ir,ir->r", ua, blk.losses @ ub)
            totals[start : start + size] += ua.sum(axis=0) * ub.sum(axis=0)
    return wins, losses, totals


def summarize_replicates(
    wins: np.ndarray,
    losses: np.ndarray,
    totals: np.ndarray,
    *,
    alpha: float,
) -> Dict[str, Any]:
    """Percentile intervals for WR, NB and WO from replicate totals.

    Replicates with wins but no losses give WR = inf and are kept, because
    discarding them would bias the interval toward the null. Replicates with
    neither wins nor losses have an undefined WR and are counted separately.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(losses > 0, wins / np.where(losses > 0, losses, 1.0), np.where(wins > 0, np.inf, np.nan))
        nb = np.where(totals > 0, (wins - losses) / np.where(totals > 0, totals, 1.0), np.nan)
        wo = np.where(
            np.isnan(nb), np.nan, np.where(np.abs(nb) < 1, (1 + nb) / (1 - nb), np.where(nb >= 1, np.inf, 0.0))
        )
    defined = ~np.isnan(wr)
    finite_log = np.log(wr[np.isfinite(wr) & (wr > 0)])
    return {
        "ci": percentile_interval(wr, alpha),
        "net_benefit_ci": percentile_interval(nb, alpha),
        "win_odds_ci": percentile_interval(wo, alpha),
        "se_log_wr": float(np.std(finite_log, ddof=1)) if finite_log.size > 1 else float("nan"),
        "se_net_benefit": float(np.nanstd(nb, ddof=1)) if np.isfinite(nb).sum() > 1 else float("nan"),
        "wr_samples": wr[defined].tolist(),
        "net_benefit_samples": nb[~np.isnan(nb)].tolist(),
        "n_infinite": int(np.isinf(wr).sum()),
        "n_undefined": int((~defined).sum()),
        "alpha": float(alpha),
    }


def pair_replicate_totals(
    results: np.ndarray,
    cells: Optional[np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample assigned pairs (optionally whole clusters of pairs)."""

    m = results.size
    units = np.arange(m) if cells is None else cells
    design = design_from_labels(units, np.zeros(m, dtype=int))
    freq = design.frequencies(n_boot, rng)
    win = (results == 1).astype(float)
    loss = (results == -1).astype(float)
    return win @ freq, loss @ freq, freq.sum(axis=0)


def bootstrap_payload(
    point_wr: float,
    replicate: Dict[str, Any],
    *,
    n_boot: int,
    seed: int,
    bootstrap_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the dictionary returned by the public bootstrap functions."""

    payload: Dict[str, Any] = {
        "wr": float(point_wr) if np.isfinite(point_wr) else point_wr,
        **replicate,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "bootstrap_type": bootstrap_type,
    }
    if extra:
        payload.update(extra)
    return payload


def labels_to_list(values: Any) -> List[Any]:
    return list(values) if values is not None else []
