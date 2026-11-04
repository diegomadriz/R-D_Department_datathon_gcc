"""Nested group CV+ intervals, evaluated only on outer held-out training groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcc_strength.audit import TARGETS
from gcc_strength.cv import repeated_group_splits
from gcc_strength.eda import assert_train
from gcc_strength.features import INPUT_COLUMNS, validate_inputs
from gcc_strength.multioutput import fit_architecture
from gcc_strength.split import GROUP_COL


def cvplus_bounds(centers, residuals, *, alpha=0.1):
    """Finite-rank CV+ bounds from one candidate per calibration group.

    Nominal 1-alpha is empirical, not an exact finite-sample CV+ guarantee. The
    generic worst-case CV+ lower coverage bound is approximately 1-2*alpha.
    Group maxima target simultaneous coverage of observed rows in a new assay.
    """
    centers, residuals = np.asarray(centers), np.asarray(residuals)
    if (
        centers.ndim != 2
        or residuals.shape != (len(centers),)
        or not np.isfinite(centers).all()
        or not np.isfinite(residuals).all()
        or (residuals < 0).any()
        or not 0 < alpha < 0.5
        or not len(residuals)
    ):
        raise ValueError("Aligned finite group centers and nonnegative residuals are required")
    n = len(residuals)
    lower_rank = int(np.floor((n + 1) * alpha))  # one-based order statistic
    upper_rank = int(np.ceil((n + 1) * (1 - alpha)))
    low = (
        np.sort(centers - residuals[:, None], axis=0)[lower_rank - 1]
        if lower_rank >= 1
        else np.full(centers.shape[1], -np.inf)
    )
    high = (
        np.sort(centers + residuals[:, None], axis=0)[upper_rank - 1]
        if upper_rank <= n
        else np.full(centers.shape[1], np.inf)
    )
    return low, high


def nested_cvplus(train, X, specs, architecture, config):
    """Re-tune each calibration model without its calibration or outer validation labels."""
    assert_train(train)
    validate_inputs(X)
    scores = [[] for _ in TARGETS]
    centers = [[] for _ in TARGETS]
    for _, _, fit, calibration in repeated_group_splits(
        train, n_splits=config["stage3"]["conformal_splits"], n_repeats=1
    ):
        model, _ = fit_architecture(train.iloc[fit], specs, architecture, config)
        held = train.iloc[calibration]
        predicted = model.predict(held[list(INPUT_COLUMNS)])
        future = model.predict(X)
        for j, target in enumerate(TARGETS):
            residual = pd.DataFrame(
                {
                    "group": held[GROUP_COL].to_numpy(),
                    "score": abs(held[target].to_numpy() - predicted[:, j]),
                }
            ).dropna()
            group_scores = residual.groupby("group", sort=True).score.max().to_numpy()
            scores[j].extend(group_scores)
            centers[j].extend(np.tile(future[:, j], (len(group_scores), 1)))
    lower, upper = [], []
    for j in range(3):
        lo, hi = cvplus_bounds(centers[j], scores[j], alpha=config["stage3"]["alpha"])
        lower.append(lo)
        upper.append(hi)
    # Do not project endpoints using target-order constraints: coverage is marginal by target.
    return np.column_stack(lower), np.column_stack(upper)


def coverage_table(intervals):
    """Row-weighted OOF coverage, unique row/group counts, width and rare-type warnings."""
    rows = []
    for target, target_part in intervals.groupby("target", sort=True):
        for scope, part in [("Todos", target_part), *target_part.groupby("ProductType", sort=True)]:
            group_coverage = part.groupby(["repeat", GROUP_COL], sort=True).covered.all().mean()
            coverage = float(part.covered.mean())
            rows.append(
                dict(
                    target=target,
                    scope=scope,
                    coverage=coverage,
                    group_coverage=float(group_coverage),
                    mean_width=float((part.upper - part.lower).mean()),
                    n_rows=part._row_id.nunique(),
                    n_groups=part[GROUP_COL].nunique(),
                    n_predictions=len(part),
                    reliable=part[GROUP_COL].nunique() >= 20 and scope not in {"Type 4", "Type 5"},
                    coverage_alarm=not 0.85 <= coverage <= 0.95,
                )
            )
    return pd.DataFrame(rows)
