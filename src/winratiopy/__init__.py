"""Deprecated alias of :mod:`pywinratio` (WinRatioPy was renamed in 0.3.0).

``import winratiopy`` and ``from winratiopy.wr import ...`` keep working but
emit a :class:`DeprecationWarning`. Use ``import pywinratio`` instead.
"""

import importlib
import sys
import warnings

import pywinratio as _pywinratio
from pywinratio import *  # noqa: F401,F403
from pywinratio import __all__, __version__  # noqa: F401

warnings.warn(
    "the 'winratiopy' package was renamed 'pywinratio'; import pywinratio instead",
    DeprecationWarning,
    stacklevel=2,
)

for _name in (
    "analysis",
    "config",
    "diagnostics",
    "gpc",
    "inference",
    "match",
    "permutation",
    "plotting",
    "power",
    "resampling",
    "sensitivity",
    "summary",
    "weights",
    "wr",
):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"pywinratio.{_name}")
del _name, _pywinratio
