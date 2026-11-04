"""OOF regression scores and paired group bootstrap comparisons against L1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, product_types) -> dict:
    """Return R2, MAE, MAPE in percent, RMSE and within-Type-1 scores.

    MAPE excludes zero targets and reports its denominator. R2 is undefined for
    fewer than two observations or constant targets, rather than forced to zero.
    """
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    types = np.asarray(product_types)
    if y.ndim != 1 or y.shape != pred.shape or y.shape != types.shape or not len(y):
        raise ValueError("Aligned nonempty one-dimensional arrays are required")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite observed targets and predictions")

    def r2(mask):
        values = y[mask]
        return (
            float(r2_score(values, pred[mask])) if len(values) >= 2 and np.ptp(values) else np.nan
        )

    type1 = types == "Type 1"
    nonzero = y != 0
    return {
        "n": len(y),
        "r2": r2(np.ones(len(y), dtype=bool)),
        "mae": float(mean_absolute_error(y, pred)),
        "mape_pct": float(100 * np.mean(np.abs((pred[nonzero] - y[nonzero]) / y[nonzero])))
        if nonzero.any()
        else np.nan,
        "n_mape": int(nonzero.sum()),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "n_type1": int(type1.sum()),
        "r2_type1": r2(type1),
        "mae_type1": float(mean_absolute_error(y[type1], pred[type1])) if type1.any() else np.nan,
    }


def within_type_r2_bootstrap(oof: pd.DataFrame, *, n_bootstrap=10000) -> dict:
    """Pool OOF predictions and resample whole groups within each product type."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    result = {}
    for number in (1, 2, 3):
        part = oof.loc[oof.ProductType.eq(f"Type {number}")]
        if part.empty:
            result[number] = dict(r2=np.nan, ci_low=np.nan, ci_high=np.nan)
            continue
        y = part.observed.to_numpy(dtype=float)
        pred = part.predicted.to_numpy(dtype=float)
        group = part.strength_group_id
        if not np.isfinite(y).all() or not np.isfinite(pred).all() or group.isna().any():
            raise ValueError("OOF values and groups must be complete and finite")
        stats = pd.DataFrame({"group": group, "y": y, "y2": y * y, "sse": (y - pred) ** 2})
        clusters = (
            stats.groupby("group", sort=True)
            .agg(n=("y", "size"), y=("y", "sum"), y2=("y2", "sum"), sse=("sse", "sum"))
            .to_numpy()
        )

        def score(values):
            n, sy, sy2, sse = np.moveaxis(values, -1, 0)
            sst = sy2 - sy * sy / n
            return np.divide(sse, sst, out=np.full_like(sst, np.nan), where=sst > 1e-10)

        point = 1 - score(clusters.sum(axis=0))
        rng = np.random.default_rng(42)
        draws = []
        for start in range(0, n_bootstrap, 256):
            indices = rng.integers(
                0, len(clusters), size=(min(256, n_bootstrap - start), len(clusters))
            )
            draws.extend(1 - score(clusters[indices].sum(axis=1)))
        draws = np.asarray(draws)
        # Undefined resamples make the percentile interval undefined, not artificially favorable.
        low, high = (
            np.quantile(draws, [0.025, 0.975]) if np.isfinite(draws).all() else (np.nan, np.nan)
        )
        result[number] = dict(r2=float(point), ci_low=float(low), ci_high=float(high))
    return result


def paired_mae_bootstrap(y_true, y_pred, l1_pred, groups, *, n_bootstrap=10000) -> dict:
    """Cluster percentile CI for row-weighted MAE(model)-MAE(L1), with seed 42.

    A sampled group contributes all its rows (and repeated OOF losses) together.
    The statistic is the ratio of summed losses to sampled row counts, so unequal
    group sizes retain the row-weighted MAE estimand. Repeats are not independent units.
    """
    y, pred, baseline = (np.asarray(v, dtype=float) for v in (y_true, y_pred, l1_pred))
    group = np.asarray(groups)
    if y.ndim != 1 or not len(y) or any(v.shape != y.shape for v in (pred, baseline, group)):
        raise ValueError("Aligned nonempty one-dimensional arrays are required")
    if not all(np.isfinite(v).all() for v in (y, pred, baseline)) or pd.isna(group).any():
        raise ValueError("Paired comparisons require finite values and nonmissing groups")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    loss = np.abs(pred - y) - np.abs(baseline - y)
    clusters = pd.DataFrame({"group": group, "loss": loss}).groupby("group", sort=True).loss
    sums = clusters.sum().to_numpy()
    counts = clusters.size().to_numpy()
    rng = np.random.default_rng(42)
    draws = np.empty(n_bootstrap)
    # Keep memory bounded even when a later stage uses a much larger dataset.
    batch = max(1, min(256, 1_000_000 // len(sums)))
    for start in range(0, n_bootstrap, batch):
        indices = rng.integers(0, len(sums), size=(min(batch, n_bootstrap - start), len(sums)))
        draws[start : start + len(indices)] = sums[indices].sum(axis=1) / counts[indices].sum(
            axis=1
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_mae": float(loss.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "beats_l1": bool(high < 0),
        "n_groups": len(sums),
        "n_bootstrap": n_bootstrap,
    }


def compare_oof(model_oof: pd.DataFrame, l1_oof: pd.DataFrame, *, n_bootstrap=10000) -> dict:
    """Align target/repeat/row pairs and retain losses from every repeat."""
    keys = ["target", "repeat", "_row_id"]
    paired = model_oof.merge(
        l1_oof, on=keys, how="outer", suffixes=("", "_l1"), validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all() or paired.target.nunique() != 1:
        raise ValueError("Compare one target on exactly the same OOF rows and repeats")
    for col in ("observed", "strength_group_id", "ProductType", "fold"):
        if not paired[col].eq(paired[f"{col}_l1"]).all():
            raise ValueError("OOF metadata and observations must agree")
    return paired_mae_bootstrap(
        paired.observed,
        paired.predicted,
        paired.predicted_l1,
        paired.strength_group_id,
        n_bootstrap=n_bootstrap,
    )
