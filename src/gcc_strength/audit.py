"""Reproducible data-quality audit of the raw GCC dataset (Phase 0).

Scope: data quality only. Feature -> target relationships are out of scope here; the only
target analyses are their own consistency, their per-``ProductType`` profile, the target
differences between near-duplicate feature vectors (empirical noise floor) and row-order drift.

Usage::

    uv run python -m gcc_strength.audit            # writes docs/fase0/data_audit_results.md
    uv run python -m gcc_strength.audit --stdout   # prints the report instead
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "datathon_data_gcc.csv"
DEFAULT_OUT = ROOT / "docs" / "fase0" / "data_audit_results.md"

TARGETS = ["Strength1", "Strength7", "Strength28"]
CATEGORICAL = "ProductType"
PHYSICAL = ["Physical_Metric_A", "Physical_Metric_B"]
PHASES = [f"Mineral_Phase_{x}" for x in "ABCDEFGH"]
CHEMICAL = ["Chemical_Composition_A", "Chemical_Composition_B"]
FEATURES = PHYSICAL + PHASES + CHEMICAL
EXPECTED_COLUMNS = [CATEGORICAL, *FEATURES, *TARGETS]
EXPECTED_TYPES = [f"Type {i}" for i in range(1, 6)]

N_BLOCKS = 8
# If two independent measurements have normal error sd s, |e1 - e2| has median
# sqrt(2) * 0.67449 * s and a single error has mean absolute value sqrt(2/pi) * s, so
# MAE = median|diff| * sqrt(2/pi) / (sqrt(2) * 0.67449) = 0.8365 * median|diff|.
NOISE_MEDIAN_TO_MAE = float(np.sqrt(2 / np.pi) / (np.sqrt(2) * stats.norm.ppf(0.75)))


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_raw(path: Path = RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw strings, parsed numeric) views of the CSV."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    parsed = pd.read_csv(path)
    return raw, parsed


def sha256(path: Path = RAW_CSV) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
def schema_report(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        non_numeric = 0
        if col != CATEGORICAL:
            s = raw[col].str.strip()
            s = s[s != ""]
            non_numeric = int(pd.to_numeric(s, errors="coerce").isna().sum())
        rows.append(
            {
                "column": col,
                "expected": "category" if col == CATEGORICAL else "float",
                "parsed_dtype": str(df[col].dtype),
                "non_numeric_tokens": non_numeric,
                "n_unique": int(df[col].nunique()),
            }
        )
    return pd.DataFrame(rows)


def product_type_raw_values(raw: pd.DataFrame) -> pd.DataFrame:
    vc = raw[CATEGORICAL].map(repr).value_counts()
    out = vc.rename_axis("raw_value").reset_index(name="n")
    out["canonical"] = out["raw_value"].str.strip("'\"").isin(EXPECTED_TYPES)
    return out


# --------------------------------------------------------------------------------------
# Missingness
# --------------------------------------------------------------------------------------
def nulls_by_column(df: pd.DataFrame) -> pd.DataFrame:
    n = df.isna().sum()
    return pd.DataFrame({"n_null": n, "pct_null": (100 * n / len(df)).round(2)})


def nulls_per_row(df: pd.DataFrame) -> pd.DataFrame:
    vc = df.isna().sum(axis=1).value_counts().sort_index()
    return vc.rename_axis("n_null_in_row").reset_index(name="n_rows")


def empty_rows(df: pd.DataFrame) -> pd.Index:
    """Rows where every feature and every target is missing (only ProductType present)."""
    return df.index[df[FEATURES + TARGETS].isna().all(axis=1)]


def null_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Co-missingness: how often columns are missing together (Jaccard of null masks)."""
    m = df[FEATURES + TARGETS].isna()
    cols = [c for c in m.columns if m[c].any()]
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            inter = (m[a] & m[b]).sum()
            union = (m[a] | m[b]).sum()
            out.loc[a, b] = inter / union if union else np.nan
    return out.round(2)


def target_null_pattern(df: pd.DataFrame) -> pd.DataFrame:
    m = df[TARGETS].isna()
    pat = m.apply(lambda r: "".join("N" if v else "o" for v in r), axis=1)
    out = pat.value_counts().rename_axis("pattern R1 R7 R28 (N=null)").reset_index(name="n_rows")
    return out


def null_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES + TARGETS if df[c].isna().any()]
    rates = df[cols].isna().groupby(df[CATEGORICAL]).mean().mul(100).round(1)
    rates.insert(0, "n_rows", df.groupby(CATEGORICAL).size())
    return rates


def missingness_vs_type_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Asymptotic chi-square test of independence between each null indicator and ProductType.

    A significant association rules out MCAR (missingness depends at least on the type).
    Types 4 and 5 have 6 and 1 rows, so expected counts are often < 5: ``min_expected`` is
    reported and those p-values are only indicative.
    """
    rows = []
    for c in FEATURES + TARGETS:
        m = df[c].isna()
        if m.sum() == 0:
            continue
        tab = pd.crosstab(df[CATEGORICAL], m)
        if tab.shape[1] < 2:
            continue
        chi2, p, dof, expected = stats.chi2_contingency(tab)
        rows.append(
            {
                "column": c,
                "chi2": round(chi2, 2),
                "dof": dof,
                "p_value": f"{p:.2e}",
                "min_expected": round(float(expected.min()), 2),
                "pct_cells_expected_lt5": round(100 * float((expected < 5).mean()), 1),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column", "chi2", "dof", "p_value", "min_expected", "pct_cells_expected_lt5"],
    )


def missingness_vs_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compare feature distributions when ``col`` is null vs not null (Mann-Whitney U).

    Only features (never targets) are compared, so this stays a data-quality check.
    """
    m = df[col].isna()
    rows = []
    for f in FEATURES:
        if f == col:
            continue
        a, b = df.loc[m, f].dropna(), df.loc[~m, f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u = stats.mannwhitneyu(a, b)
        rows.append(
            {
                "feature": f,
                "median_when_null": round(a.median(), 3),
                "median_when_present": round(b.median(), 3),
                "p_value": f"{u.pvalue:.2e}",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Duplicates and noise floor
# --------------------------------------------------------------------------------------
def duplicate_summary(df: pd.DataFrame) -> dict:
    empty = empty_rows(df)
    non_empty = df.drop(index=empty)
    feat_cols = [CATEGORICAL, *FEATURES]
    feat_dup = non_empty[non_empty.duplicated(subset=feat_cols, keep=False)]
    informative = feat_dup[feat_dup[FEATURES].notna().any(axis=1)]
    no_features = df[FEATURES].isna().all(axis=1)
    no_targets = df[TARGETS].isna().all(axis=1)
    return {
        "rows_without_any_feature_but_some_target": list(df.index[no_features & ~no_targets]),
        "rows_with_features_but_no_target": list(df.index[~no_features & no_targets]),
        "exact_duplicate_rows (keep=first)": int(df.duplicated().sum()),
        "exact_duplicates_among_non_empty_rows": int(non_empty.duplicated().sum()),
        "empty_rows (all features and targets null)": len(empty),
        "feature_duplicates_non_empty (keep=False)": len(feat_dup),
        "feature_duplicates_with_any_non_null_feature": len(informative),
    }


def strength_groups(df: pd.DataFrame, min_shared: int = 2) -> pd.Series:
    """Group id for runs of consecutive rows that look like the same strength test.

    Row i joins the group of row i-1 when both share at least ``min_shared`` non-null targets
    and every jointly observed target is equal. Rows that do not match get their own group.
    The id is partition metadata only: it must never be used as a model feature.
    """
    t = df[TARGETS]
    prev = t.shift()
    both = t.notna() & prev.notna()
    equal = ((t == prev) | ~both).all(axis=1)
    same_as_prev = equal & (both.sum(axis=1) >= min_shared)
    same_as_prev.iloc[0] = False
    return (~same_as_prev).cumsum().rename("strength_group_id") - 1


def target_triple_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-row strength groups (see :func:`strength_groups`) and how their features differ.

    These look like one strength test associated with several quality records, so they must be
    kept together in splits and CV folds (group leakage).
    """
    cols = ["n_rows", "product_types", "row_ids", "max_row_gap", "n_features_identical_in_group"]
    gid = strength_groups(df)
    sizes = gid.map(gid.value_counts())
    dup = df[sizes > 1].copy()
    if dup.empty:
        return pd.DataFrame(columns=cols)
    dup["group"] = gid[sizes > 1]
    agg = dup.groupby("group").agg(
        n_rows=(CATEGORICAL, "size"),
        product_types=(CATEGORICAL, lambda s: ",".join(sorted(set(s)))),
        row_ids=(CATEGORICAL, lambda s: ",".join(map(str, s.index))),
        max_row_gap=(CATEGORICAL, lambda s: int(np.max(np.diff(s.index))) if len(s) > 1 else 0),
    )
    feats_equal = []
    for _, g in dup.groupby("group"):
        same = [f for f in FEATURES if g[f].nunique(dropna=False) == 1]
        feats_equal.append(len(same))
    agg["n_features_identical_in_group"] = feats_equal
    return agg.reset_index(drop=True)[cols]


def consecutive_equal_targets(df: pd.DataFrame) -> int:
    t = df[TARGETS]
    return int((t.eq(t.shift()) & t.notna()).all(axis=1).sum())


def near_duplicate_pairs(df: pd.DataFrame, tols=(0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    """Same-type pairs whose complete feature vectors differ by < tol (Chebyshev distance).

    ``Physical_Metric_A`` is excluded (76% null). ``Physical_Metric_B`` (z-scored) is divided by
    10 so that its spread is comparable with the [0, 1] min-max columns.
    For each tolerance it reports how far apart the targets of those pairs are: this is the
    empirical noise floor (repeatability) if the data contained replicates.
    """
    feats = [f for f in FEATURES if f != "Physical_Metric_A"]
    c = df.dropna(subset=feats)
    X = c[feats].to_numpy(dtype=float).copy()
    X[:, feats.index("Physical_Metric_B")] /= 10
    dist = squareform(pdist(X, "chebyshev"))
    same = c[CATEGORICAL].to_numpy()[:, None] == c[CATEGORICAL].to_numpy()[None, :]
    rows = []
    for tol in tols:
        i, j = np.where(np.triu((dist < tol) & same, 1))
        row = {"tol": tol, "n_pairs": len(i)}
        for t in TARGETS:
            a, b = c[t].to_numpy()[i], c[t].to_numpy()[j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum():
                rel = np.abs(a - b)[ok] / ((a + b)[ok] / 2) * 100
                row[f"{t}_n"] = int(ok.sum())
                row[f"{t}_median_abs_diff"] = float(np.median(np.abs(a - b)[ok]))
                row[f"{t}_median_rel_diff_pct"] = round(float(np.median(rel)), 2)
                row[f"{t}_implied_mape_pct"] = round(NOISE_MEDIAN_TO_MAE * float(np.median(rel)), 2)
            else:
                row[f"{t}_n"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Ranges, scaling and impossible values
# --------------------------------------------------------------------------------------
def range_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES + TARGETS:
        s = df[c].dropna()
        rows.append(
            {
                "column": c,
                "n": len(s),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "median": s.median(),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "mean": s.mean(),
                "std_ddof0": s.std(ddof=0),
                "n_eq_0": int((s == 0).sum()),
                "n_eq_1": int((s == 1).sum()),
                "rows_at_min": ",".join(map(str, s.index[s == s.min()][:6])),
                "rows_at_max": ",".join(map(str, s.index[s == s.max()][:6])),
                "all_multiple_of_10": bool(((s % 10) == 0).all()) if c in TARGETS else None,
            }
        )
    return pd.DataFrame(rows)


def scaling_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Infer how each feature was pre-scaled by the data provider."""
    rows = []
    for c in FEATURES:
        s = df[c].dropna()
        minmax = np.isclose(s.min(), 0) and np.isclose(s.max(), 1)
        zscore = abs(s.mean()) < 1e-9 and np.isclose(s.std(ddof=0), 1)
        kind = "min-max [0,1]" if minmax else ("z-score (ddof=0)" if zscore else "unknown")
        rows.append({"column": c, "inferred_scaling": kind, "n_non_null": len(s)})
    return pd.DataFrame(rows)


def robust_outliers(df: pd.DataFrame, z: float = 3.5) -> pd.DataFrame:
    """IQR (1.5x) and robust z (MAD) outlier counts per column, on the full data (audit only)."""
    rows = []
    for c in FEATURES + TARGETS:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        mad = stats.median_abs_deviation(s, scale="normal")
        dev = np.abs(s - s.median())
        # With MAD == 0 any value different from the median is infinitely far in robust z.
        rz = dev / mad if mad > 0 else dev.where(dev == 0, np.inf)
        rows.append(
            {
                "column": c,
                "iqr_outliers": int(iqr_out),
                f"robust_z>{z}": int((rz > z).sum()),
                "skew": round(stats.skew(s), 2),
                "kurtosis_excess": round(stats.kurtosis(s), 2),
            }
        )
    return pd.DataFrame(rows)


def suspicious_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Anomalous rows: phases at the scaled minimum, many extremes, order violations, |z|>4.

    A scaled 0 is the observed column minimum, not necessarily a physical zero.
    """
    flags = pd.DataFrame(index=df.index)
    ph = df[PHASES]
    flags["phases_at_scaled_min_ge7"] = (ph.fillna(-1) == 0).sum(axis=1) >= len(PHASES) - 1
    flags["n_phases_at_0_or_1"] = ((ph == 0) | (ph == 1)).sum(axis=1)
    flags["r1_gt_r7"] = df["Strength1"] > df["Strength7"]
    flags["r7_gt_r28"] = df["Strength7"] > df["Strength28"]
    flags["pmb_abs_z_gt_4"] = df["Physical_Metric_B"].abs() > 4
    keep = (
        flags["phases_at_scaled_min_ge7"]
        | (flags["n_phases_at_0_or_1"] >= 3)
        | flags["r1_gt_r7"]
        | flags["r7_gt_r28"]
        | flags["pmb_abs_z_gt_4"]
    )
    out = flags[keep].copy()
    out.insert(0, CATEGORICAL, df.loc[keep, CATEGORICAL])
    return out


# --------------------------------------------------------------------------------------
# Compositional structure of the mineral phases
# --------------------------------------------------------------------------------------
@dataclass
class LinearRelation:
    target: str
    predictors: list[str]
    intercept: float
    coefs: list[float]
    r2: float
    max_abs_residual: float
    n: int


def phase_dependency_r2(df: pd.DataFrame) -> pd.DataFrame:
    """R^2 of each phase regressed (OLS + intercept) on the other seven."""
    X = df[PHASES].dropna()
    rows = []
    for t in PHASES:
        others = [p for p in PHASES if p != t]
        A = np.c_[np.ones(len(X)), X[others].to_numpy()]
        y = X[t].to_numpy()
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"phase": t, "r2_from_other_phases": round(r2, 6)})
    return pd.DataFrame(rows)


def phase_singular_values(df: pd.DataFrame) -> np.ndarray:
    X = df[PHASES].dropna()
    return np.linalg.svd((X - X.mean()).to_numpy(), compute_uv=False)


def fit_linear_relation(df: pd.DataFrame, target: str, predictors: list[str]) -> LinearRelation:
    X = df[[target, *predictors]].dropna()
    A = np.c_[np.ones(len(X)), X[predictors].to_numpy()]
    y = X[target].to_numpy()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return LinearRelation(
        target=target,
        predictors=predictors,
        intercept=float(coef[0]),
        coefs=[float(c) for c in coef[1:]],
        r2=float(r2),
        max_abs_residual=float(np.abs(resid).max()),
        n=len(X),
    )


def phase_sum_report(df: pd.DataFrame) -> pd.DataFrame:
    """Row sums of the (scaled) phases. In scaled units a sum near a constant would not be
    expected even if the raw phases closed to 100%, so the closure test is the SVD/R^2 one."""
    s = df[PHASES].dropna().sum(axis=1)
    return s.describe().to_frame("sum_scaled_phases").T


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
def target_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in [("all", df), *df.groupby(CATEGORICAL)]:
        r17 = sub.dropna(subset=["Strength1", "Strength7"])
        r728 = sub.dropna(subset=["Strength7", "Strength28"])
        r128 = sub.dropna(subset=["Strength1", "Strength28"])
        rows.append(
            {
                "group": name,
                "n_R1_R7": len(r17),
                "pct_R1>R7": round(100 * (r17.Strength1 > r17.Strength7).mean(), 2)
                if len(r17)
                else np.nan,
                "n_R7_R28": len(r728),
                "pct_R7>R28": round(100 * (r728.Strength7 > r728.Strength28).mean(), 2)
                if len(r728)
                else np.nan,
                "pct_R1>R28": round(100 * (r128.Strength1 > r128.Strength28).mean(), 2)
                if len(r128)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def target_profile_by_type(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(CATEGORICAL)[TARGETS]
    prof = g.agg(["count", "mean", "std", "median"]).round(0)
    prof.columns = [f"{t}_{s}" for t, s in prof.columns]
    ratio = (g.median()["Strength1"] / g.median()["Strength28"]).round(3).rename("R1/R28_median")
    ratio7 = (g.median()["Strength7"] / g.median()["Strength28"]).round(3).rename("R7/R28_median")
    return prof.join(ratio).join(ratio7)


def baseline_by_type_in_sample(df: pd.DataFrame) -> pd.DataFrame:
    """L0 (global mean) and L1 (mean per ProductType) fitted and scored on the FULL data.

    Only for sizing: it is the per-type target profile expressed as errors. It uses future test
    rows, so it must never drive a modelling decision; the real L1 is computed by CV on train.
    """
    rows = []
    for t in TARGETS:
        s = df.dropna(subset=[t])
        y = s[t]
        preds = {
            "L0_global_mean": pd.Series(y.mean(), index=y.index),
            "L1_mean_by_type": s.groupby(CATEGORICAL)[t].transform("mean"),
        }
        for name, pred in preds.items():
            err = y - pred
            rows.append(
                {
                    "target": t,
                    "baseline": name,
                    "n": len(y),
                    "r2": round(1 - (err**2).sum() / ((y - y.mean()) ** 2).sum(), 3),
                    "mae": round(err.abs().mean(), 0),
                    "rmse": round(float(np.sqrt((err**2).mean())), 0),
                    "mape_pct": round(100 * (err.abs() / y).mean(), 2),
                }
            )
    return pd.DataFrame(rows)


def feature_profile_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Median of each feature per ProductType (evidence for the variable fingerprint)."""
    return df.groupby(CATEGORICAL)[FEATURES].median().round(3).T


def target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation among targets (structure of the outputs, not feature -> target)."""
    return df[TARGETS].corr(method="pearson").round(3)


# --------------------------------------------------------------------------------------
# Row-order drift
# --------------------------------------------------------------------------------------
def block_drift(df: pd.DataFrame, n_blocks: int = N_BLOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis across contiguous row blocks, overall and within Type 1 only.

    If the file order were temporal, drift would show as block-to-block shifts.
    Within-Type-1 tests remove the confounding of the type mix changing along the file.
    """
    blocks = pd.Series(np.arange(len(df)) * n_blocks // len(df), index=df.index, name="block")
    rows = []
    for scope, sub in [("all", df), ("Type 1", df[df[CATEGORICAL] == "Type 1"])]:
        b = blocks.loc[sub.index]
        for c in FEATURES + TARGETS:
            groups = [sub.loc[b == k, c].dropna() for k in range(n_blocks)]
            groups = [g for g in groups if len(g) >= 5]
            if len(groups) < 3:
                continue
            h, p = stats.kruskal(*groups)
            rho, prho = stats.spearmanr(sub.index[sub[c].notna()], sub[c].dropna())
            rows.append(
                {
                    "scope": scope,
                    "column": c,
                    "kruskal_H": round(h, 1),
                    "p_value": f"{p:.1e}",
                    "spearman_rho_vs_row": round(rho, 3),
                    "p_rho": f"{prho:.1e}",
                }
            )
    tests = pd.DataFrame(rows)
    t1 = df[CATEGORICAL] == "Type 1"
    means = df.loc[t1, FEATURES + TARGETS].groupby(blocks[t1]).median().round(3)
    means.insert(0, "n_type1", t1.groupby(blocks).sum())
    means.insert(0, "rows", [f"{s.index.min()}-{s.index.max()}" for _, s in df.groupby(blocks)])
    return tests, means


def type_positions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(CATEGORICAL)
        .apply(
            lambda s: pd.Series(
                {
                    "n": len(s),
                    "first_row": s.index.min(),
                    "last_row": s.index.max(),
                    "median_row": int(np.median(s.index)),
                }
            ),
            include_groups=False,
        )
        .astype(int)
    )


def feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].corr(method="spearman").round(2)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _md(obj, floatfmt: str = ".4g") -> str:
    if isinstance(obj, pd.DataFrame):
        return obj.to_markdown(floatfmt=floatfmt)
    if isinstance(obj, dict):
        return "\n".join(f"- {k}: **{v}**" for k, v in obj.items())
    return str(obj)


def build_report(path: Path = RAW_CSV) -> str:
    raw, df = load_raw(path)
    out: list[str] = []
    add = out.append

    add("# Resultados del data audit (generado)")
    add("")
    add(
        "> Archivo generado por `uv run python -m gcc_strength.audit`. No se edita a mano. "
        "La interpretación está en `data_audit.md`."
    )
    add("")
    add(f"- Archivo: `data/{path.name}` · SHA-256 `{sha256(path)}`")
    add(f"- Filas: **{len(df)}** · Columnas: **{df.shape[1]}**")
    add(f"- Columnas esperadas presentes y en orden: **{list(df.columns) == EXPECTED_COLUMNS}**")
    add("")

    add("## 1. Esquema y tipos")
    add(_md(schema_report(raw, df).set_index("column")))
    add("")
    add("### Valores crudos de `ProductType`")
    add(_md(product_type_raw_values(raw).set_index("raw_value")))
    add("")

    add("## 2. Nulos")
    add("### Por columna")
    add(_md(nulls_by_column(df)))
    add("")
    add("### Por fila (número de nulos en la fila)")
    add(_md(nulls_per_row(df).set_index("n_null_in_row")))
    add("")
    er = empty_rows(df)
    add(f"Filas vacías (solo `ProductType`): **{len(er)}**, filas {list(er)}.")
    add("")
    add("### Patrón de nulos en targets")
    add(_md(target_null_pattern(df).set_index("pattern R1 R7 R28 (N=null)")))
    add("")
    add("### Co-ocurrencia de nulos (Jaccard de máscaras)")
    add(_md(null_blocks(df)))
    add("")
    add("### Tasa de nulos (%) por `ProductType`")
    add(_md(null_rate_by_type(df)))
    add("")
    add("### Independencia nulo vs `ProductType` (chi²)")
    add(_md(missingness_vs_type_tests(df).set_index("column")))
    add("")
    add("### `Physical_Metric_A` nulo vs presente: distribución de las demás features")
    add(_md(missingness_vs_features(df, "Physical_Metric_A").set_index("feature")))
    add("")

    add("## 3. Duplicados y piso de ruido")
    add(_md(duplicate_summary(df)))
    add("")
    add("### Grupos de filas con el mismo trío (R1, R7, R28) y features distintas")
    ttg = target_triple_groups(df)
    n_rows_grouped = int(ttg["n_rows"].sum()) if len(ttg) else 0
    add(
        "Regla: filas consecutivas que comparten ≥ 2 targets no nulos y todos los observados son "
        "iguales (`strength_groups`)."
    )
    add(f"Grupos: **{len(ttg)}**; filas involucradas: **{n_rows_grouped}**.")
    n_consec = consecutive_equal_targets(df)
    add(f"Filas cuyo trío de targets es igual al de la fila anterior: **{n_consec}**.")
    add("")
    add(_md(ttg))
    add("")
    add("### Casi-duplicados en features (mismo tipo, distancia de Chebyshev < tol)")
    add(
        "Sin `Physical_Metric_A` (76% nulos); `Physical_Metric_B` (z-score) dividida entre 10. "
        f"`implied_mape_pct` = {NOISE_MEDIAN_TO_MAE:.4f} × diferencia relativa mediana: es el "
        "MAPE de un predictor perfecto si toda la diferencia fuera ruido normal del ensayo "
        "(cota superior, no medición)."
    )
    add(_md(near_duplicate_pairs(df).set_index("tol")))
    add("")

    add("## 4. Rangos, escalado y valores sospechosos")
    add("### Escalado inferido")
    add(_md(scaling_diagnosis(df).set_index("column")))
    add("")
    add("### Rangos")
    add(_md(range_report(df).set_index("column")))
    add("")
    add("### Outliers (IQR 1.5× y z robusto > 3.5), asimetría y curtosis; dataset completo")
    add(_md(robust_outliers(df).set_index("column")))
    add("")
    add("### Filas sospechosas")
    add(_md(suspicious_rows(df)))
    add("")

    add("## 5. Estructura composicional de `Mineral_Phase_A..H`")
    add("### Suma por fila (unidades escaladas)")
    add(_md(phase_sum_report(df)))
    add("")
    add("### R² de cada fase explicada por las otras siete (OLS + intercepto)")
    add(_md(phase_dependency_r2(df).set_index("phase")))
    add("")
    sv = phase_singular_values(df)
    add("Valores singulares de la matriz de fases centrada: " + ", ".join(f"{v:.4g}" for v in sv))
    add("")
    rel = fit_linear_relation(df, "Mineral_Phase_A", ["Mineral_Phase_B", "Mineral_Phase_C"])
    add(
        f"Relación casi exacta: `Mineral_Phase_A = {rel.intercept:.6f} + {rel.coefs[0]:.6f}·B + "
        f"{rel.coefs[1]:.6f}·C` (n={rel.n}, R²={rel.r2:.8f}, residuo máximo "
        f"{rel.max_abs_residual:.2e}; VIF implícito ≈ {1 / (1 - rel.r2):.3g})."
    )
    add("")
    others = [p for p in PHASES if p not in ("Mineral_Phase_A",)]
    sv2 = np.linalg.svd(
        (df[others].dropna() - df[others].dropna().mean()).to_numpy(), compute_uv=False
    )
    add(
        "Sin `Mineral_Phase_A`, valor singular mínimo de B..H: "
        f"**{sv2.min():.4g}** (sin otra dependencia lineal exacta, no hay cierre a constante)."
    )
    add("")

    add("## 6. Consistencia de targets")
    add(_md(target_consistency(df).set_index("group")))
    add("")
    add("### Correlación entre targets (Pearson)")
    add(_md(target_correlations(df)))
    add("")
    add("### Baselines in-sample sobre el dataset completo (solo dimensionamiento)")
    add(
        "> Usa también las filas que serán test: no se usa para decidir nada. "
        "El baseline oficial es L1 por CV sobre train (Fase 3)."
    )
    add(_md(baseline_by_type_in_sample(df).set_index(["target", "baseline"])))
    add("")
    add("### Perfil de R1/R7/R28 por `ProductType`")
    add(_md(target_profile_by_type(df)))
    add("")

    add("## 7. ¿El orden de filas es temporal? Deriva por bloques")
    add("### Posición de cada tipo en el archivo")
    add(_md(type_positions(df)))
    add("")
    tests, means = block_drift(df)
    add(f"### Mediana por bloque, solo Type 1 ({N_BLOCKS} bloques contiguos del archivo)")
    add(_md(means))
    add("")
    add("### Pruebas de deriva (Kruskal-Wallis entre bloques; Spearman vs número de fila)")
    add(_md(tests.set_index(["scope", "column"])))
    add("")

    add("## 8. Perfil de features por `ProductType` (mediana)")
    add(_md(feature_profile_by_type(df)))
    add("")

    add("## 9. Correlación entre features (Spearman; sin targets)")
    add(_md(feature_correlations(df)))
    add("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.stdout:
        print(report)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
