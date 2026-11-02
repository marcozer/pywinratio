"""Hierarchy sensitivity analyses, sample size and figures.

Run: python examples/03_sensitivity_and_figures.py  (writes PNG files to the current directory)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pywinratio import (  # noqa: E402
    Outcome,
    compare_hierarchies,
    plot_forest,
    plot_tiers,
    sample_size_win_ratio,
    win_ratio,
)
from pywinratio.datasets import make_surgical_cohort  # noqa: E402

cohort = make_surgical_cohort()
death = Outcome.binary("death90", name="Death", terminal=True)
clavien = Outcome.ordinal("clavien", name="Clavien-Dindo")
popf = Outcome.ordinal("popf", name="POPF")
readmission = Outcome.binary("readmission", name="Readmission")
primary = [death, clavien, popf, readmission, Outcome.continuous("los", name="LOS", margin=1)]

table = compare_hierarchies(
    cohort,
    {
        "Primary (LOS margin 1 day)": primary,
        "Exact LOS": primary[:-1] + [Outcome.continuous("los", name="LOS")],
        "LOS categories": primary[:-1] + [Outcome.continuous("los", cutpoints=[7, 13], name="LOS")],
        "Without LOS": primary[:-1],
    },
    group="volume",
    treatment="high",
    control="low",
    cluster_col="hospital",
)
print(table[["hierarchy", "wr", "ci_lower", "ci_upper", "net_benefit", "tie_share"]].round(3).to_string(index=False))

ax = plot_forest(table, label="hierarchy")
ax.figure.savefig("sensitivity_forest.png", dpi=150, bbox_inches="tight")

result = win_ratio(cohort, primary, group="volume", treatment="high", control="low", cluster="hospital")
ax = plot_tiers(result, arm_labels=("High volume", "Low volume"))
ax.figure.savefig("tier_decomposition.png", dpi=150, bbox_inches="tight")

print(sample_size_win_ratio(1.3, p_tie=0.25, power=0.9))
