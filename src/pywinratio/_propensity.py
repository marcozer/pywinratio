"""Shared propensity-score model used by matching and weighting."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def treatment_indicator(
    data: pd.DataFrame,
    treatment: str,
    treated: Any,
    control: Any = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Restrict to the two compared arms and return a 0/1 treatment indicator."""

    if treatment not in data.columns:
        raise ValueError(f"treatment column {treatment!r} not found")
    data = data[data[treatment].notna()]
    levels = list(pd.unique(data[treatment]))
    if treated not in levels:
        raise ValueError(f"treated value {treated!r} not found in {treatment!r}")
    if control is None:
        others = [level for level in levels if level != treated]
        if len(others) != 1:
            raise ValueError(
                f"{treatment!r} has {len(levels)} levels {levels}; pass control=... to choose the comparator"
            )
        control = others[0]
    elif control not in levels:
        raise ValueError(f"control value {control!r} not found in {treatment!r}")
    data = data[data[treatment].isin([treated, control])].copy()
    return data, (data[treatment] == treated).astype(int)


def propensity_pipeline(
    data: pd.DataFrame,
    covariates: Sequence[str],
    *,
    missing: str,
    penalty: Optional[float],
    seed: int,
):
    """Logistic propensity model with scaling, one-hot encoding and optional imputation.

    ``penalty=None`` fits the unpenalized maximum-likelihood model, which is
    required for the exact mean-balance property of overlap weights (Li,
    Morgan and Zaslavsky, 2018). A float is the inverse regularization
    strength ``C`` of scikit-learn.
    """

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in covariates if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in covariates if c not in numeric]
    numeric_steps: list = []
    categorical_steps: list = []
    if missing == "simple":
        numeric_steps.append(("impute", SimpleImputer(strategy="median")))
        categorical_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    numeric_steps.append(("scale", StandardScaler()))
    categorical_steps.append(("encode", OneHotEncoder(drop="first", sparse_output=False)))
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    model = LogisticRegression(
        C=np.inf if penalty is None else float(penalty),
        max_iter=10000,
        tol=1e-10,
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


def fit_propensity(
    data: pd.DataFrame,
    covariates: Sequence[str],
    indicator: pd.Series,
    *,
    missing: str,
    penalty: Optional[float],
    seed: int,
):
    """Fit the propensity model; return ``(pipeline, scores)``."""

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    model = propensity_pipeline(data, covariates, missing=missing, penalty=penalty, seed=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(data[list(covariates)], indicator)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        warnings.warn(
            "the propensity model did not converge (possible separation); "
            "consider fewer covariates or a penalty, e.g. penalty=1.0",
            UserWarning,
            stacklevel=3,
        )
    scores = pd.Series(model.predict_proba(data[list(covariates)])[:, 1], index=data.index, name="propensity_score")
    return model, scores


def check_covariates(data: pd.DataFrame, covariates: Sequence[str]) -> list[str]:
    covariates = list(covariates)
    if not covariates:
        raise ValueError("at least one covariate is required")
    missing = [c for c in covariates if c not in data.columns]
    if missing:
        raise ValueError(f"covariates not found: {missing}")
    return covariates


def common_support_bounds(scores: pd.Series, indicator: pd.Series) -> Tuple[float, float]:
    """Overlap region ``[max(min_t, min_c), min(max_t, max_c)]`` of the scores."""

    treated = indicator == 1
    low = max(scores[treated].min(), scores[~treated].min())
    high = min(scores[treated].max(), scores[~treated].max())
    return float(low), float(high)
