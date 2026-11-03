"""Descriptive hydration-curve fits; fitted parameters are never predictor inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from gcc_strength.audit import CATEGORICAL, TARGETS

AGES = np.array([1.0, 7.0, 28.0])
EXPONENT_BOUNDS = (0.05, 1.5)


def strength_curve(t, f28, s, exponent=0.5) -> np.ndarray:
    """Evaluate a positive, monotone curve for positive ages and parameters."""
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0) or np.any(np.asarray(f28) <= 0):
        raise ValueError("Ages and f28 must be positive")
    if np.any(np.asarray(s) < 0) or np.any(np.asarray(exponent) <= 0):
        raise ValueError("s must be nonnegative and exponent positive")
    return np.asarray(f28) * np.exp(np.asarray(s) * (1 - (28 / t) ** exponent))


def fit_curve(values, exponent=0.5) -> dict:
    """Fit (f28, s) by raw-strength least squares, using all three observed ages."""
    y = np.asarray(values, dtype=float)
    if y.shape != (3,) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Exactly three finite positive strengths are required")
    # Scale all residuals by the same constant to improve conditioning; the optimum is SSE.
    scale = y[-1]
    result = least_squares(
        lambda p: (strength_curve(AGES, p[0] * scale, p[1], exponent) - y) / scale,
        x0=[1.0, 0.25],
        bounds=([1e-8, 0.0], [np.inf, 5.0]),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    if not result.success:
        raise RuntimeError(f"Curve fit did not converge: {result.message}")
    return {"f28": result.x[0] * scale, "s": result.x[1], "exponent": exponent}


def fit_type_exponent(values, fixed_parameters) -> tuple[np.ndarray, float]:
    """Jointly fit one exponent per type and two parameters per sample (raw SSE).

    The sparse Jacobian connects each sample only to its own pair and the shared exponent.
    Multiple deterministic starting exponents reduce sensitivity to a local optimum.
    """
    y = np.asarray(values, dtype=float)
    initial = np.asarray(fixed_parameters, dtype=float)
    if y.ndim != 2 or y.shape[1] != 3 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("A positive finite n x 3 array is required")
    if initial.shape != (len(y), 2) or len(y) == 0:
        raise ValueError("One initial (f28, s) pair per sample is required")
    scale = np.median(y[:, -1])
    n = len(y)
    pattern = lil_matrix((3 * n, 2 * n + 1), dtype=int)
    for i in range(n):
        pattern[3 * i : 3 * i + 3, [2 * i, 2 * i + 1, 2 * n]] = 1
    low = np.tile([1e-8, 0], n).tolist() + [EXPONENT_BOUNDS[0]]
    high = np.tile([np.inf, 5], n).tolist() + [EXPONENT_BOUNDS[1]]

    def residual(p):
        pairs = p[:-1].reshape(n, 2)
        predicted = strength_curve(AGES[None, :], pairs[:, :1] * scale, pairs[:, 1:], p[-1])
        return ((predicted - y) / scale).ravel()

    fits = []
    for start in (0.25, 0.5, 0.9):
        p = initial.copy()
        p[:, 0] /= scale
        p[:, 1] = np.clip(p[:, 1] * 0.5 / start, 1e-8, 4.99)
        fit = least_squares(
            residual,
            np.r_[p.ravel(), start],
            bounds=(low, high),
            jac_sparsity=pattern.tocsr(),
            max_nfev=1000,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if fit.success:
            fits.append(fit)
    if not fits:
        raise RuntimeError("No type-exponent fit converged")
    best = min(fits, key=lambda fit: fit.cost)
    parameters = best.x[:-1].reshape(n, 2).copy()
    parameters[:, 0] *= scale
    return parameters, float(best.x[-1])


def fit_samples(train: pd.DataFrame) -> pd.DataFrame:
    """Return fixed and type-exponent in-sample fits on complete training targets only."""
    from gcc_strength.eda import assert_train

    assert_train(train)
    rows = []
    for ptype, subset in train.groupby(CATEGORICAL, observed=True):
        complete = subset.dropna(subset=TARGETS)
        if complete.empty:
            continue
        values = complete[TARGETS].to_numpy(dtype=float)
        fixed = [fit_curve(y) for y in values]
        initial = np.array([[p["f28"], p["s"]] for p in fixed])
        parameters, exponent = fit_type_exponent(values, initial)
        for model, pairs, a in (("sqrt", initial, 0.5), ("free", parameters, exponent)):
            for (index, sample), y, pair in zip(complete.iterrows(), values, pairs, strict=True):
                pred = strength_curve(AGES, *pair, a)
                row = {
                    "_row_id": sample.get("_row_id", index),
                    CATEGORICAL: ptype,
                    "model": model,
                    "f28": pair[0],
                    "s": pair[1],
                    "exponent": a,
                }
                for target, observed, fitted in zip(TARGETS, y, pred, strict=True):
                    row[f"ape_{target}"] = 100 * abs(fitted / observed - 1)
                    row[f"bias_{target}"] = 100 * (fitted / observed - 1)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_fits(fits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parameter distributions and in-sample relative errors by type and form."""
    rows = []
    if fits.empty:
        return pd.DataFrame()
    for (ptype, model), part in fits.groupby([CATEGORICAL, "model"], observed=True):
        row = {
            CATEGORICAL: ptype,
            "model": model,
            "n_complete": len(part),
            "exponent": part.exponent.iloc[0],
            "exponent_at_bound": bool(np.isclose(part.exponent.iloc[0], EXPONENT_BOUNDS).any()),
            "s_mean": part.s.mean(),
            "s_std": part.s.std(),
            "s_min": part.s.min(),
            "s_q25": part.s.quantile(0.25),
            "s_median": part.s.median(),
            "s_q75": part.s.quantile(0.75),
            "s_max": part.s.max(),
        }
        for target in TARGETS:
            row[f"MAPE_{target}"] = part[f"ape_{target}"].mean()
            row[f"bias_{target}"] = part[f"bias_{target}"].mean()
        row["R1_overestimated_pct"] = 100 * (part.bias_Strength1 > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def anchored_diagnostic(train: pd.DataFrame) -> pd.DataFrame:
    """Compare the chemical R7-anchored sqrt diagnostic with the full 3-age fit.

    Observed R28 fixes f28, and R7 fixes s; the R1 reconstruction is a retrospective
    curve-shape diagnostic only. No observed target is offered as a model feature.
    """
    from gcc_strength.eda import assert_train

    assert_train(train)
    rows = []
    for ptype, subset in train.groupby(CATEGORICAL, observed=True):
        complete = subset.dropna(subset=TARGETS)
        if complete.empty:
            continue
        s7 = -np.log(complete.Strength7 / complete.Strength28)
        predicted = strength_curve(1, complete.Strength28, s7)
        relative = 100 * (predicted / complete.Strength1 - 1)
        rows.append(
            {
                CATEGORICAL: ptype,
                "n_complete": len(complete),
                "s7_median": s7.median(),
                "R1_bias_mean_pct": relative.mean(),
                "R1_bias_median_pct": relative.median(),
                "R1_MAPE_pct": relative.abs().mean(),
            }
        )
    return pd.DataFrame(rows)
