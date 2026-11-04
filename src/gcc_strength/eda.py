"""Train-only descriptive EDA and physics diagnostics (no predictive selection).

Usage: ``uv run python -m gcc_strength.eda``. Only ``load_train`` reads local Gold.
All analysis entrypoints require the provenance contract created by ``train_only``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gcc_strength.audit import CATEGORICAL, FEATURES, TARGETS
from gcc_strength.config import ROOT
from gcc_strength.split import PARTITION_LABELS

DEFAULT_GOLD = ROOT / "outputs/medallion/datathon_data_gold.parquet"
DEFAULT_OUT = ROOT / "docs/fase2"
NUMERIC = FEATURES + TARGETS
RATIOS = [
    "log_C_over_B",
    "C_over_A",
    "log_D_over_A",
    "log_F_over_E",
    "log_A",
    "log_H_over_F",
    "F_over_E_plus_F",
    "log_H_over_A",
]
RAW_VIF_FEATURES = [c for c in FEATURES if c != "Physical_Metric_A"]
BASE_FEATURES = [c for c in RAW_VIF_FEATURES if c != "Mineral_Phase_A"]
RATIO_DESIGN_FEATURES = [
    "Physical_Metric_B",
    "Mineral_Phase_E",
    "Mineral_Phase_G",
    "Mineral_Phase_H",
    "Chemical_Composition_A",
    "Chemical_Composition_B",
    "log_C_over_B",
    "log_D_over_A",
    "log_F_over_E",
]
PROVENANCE = "eda_train_only"
N_BLOCKS = 8
BOOTSTRAPS = 2000
NEUTRAL_MARGIN = 0.1


def train_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an already train-filtered frame, drop split, and attach provenance.

    The notebook must filter on the server before collection. This function deliberately
    rejects mixed frames rather than silently allowing another analysis ingestion path.
    """
    if "split" not in frame or frame.empty or not frame["split"].eq("train").all():
        raise ValueError("EDA requires a nonempty frame containing only split == train")
    result = frame.drop(columns="split").copy()
    result.attrs[PROVENANCE] = True
    return result


def assert_train(frame: pd.DataFrame) -> None:
    """Reject unfiltered, unmarked or still partition-labelled analysis inputs."""
    if frame.attrs.get(PROVENANCE) is not True or "split" in frame.columns:
        raise ValueError("Use load_train or train_only before analysis; test rows are forbidden")


def load_train(path: Path | str = DEFAULT_GOLD) -> pd.DataFrame:
    """Read only persisted train rows, validate their labels, and discard split.

    A split-only scan validates the partition vocabulary without accessing test payloads.
    Parquet predicate pushdown excludes test features, targets and metadata from the frame.
    """
    labels = pd.read_parquet(path, columns=["split"])["split"]
    if not labels.isin(PARTITION_LABELS).all():
        raise ValueError("Unexpected or missing partition label")
    selected = pd.read_parquet(path, filters=[("split", "==", "train")])
    result = train_only(selected)
    result.attrs["lockbox_size"] = len(labels) - int(labels.eq("train").sum())
    return result


def descriptive_statistics(train: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric variables globally and by nominal product category."""
    assert_train(train)
    rows = []
    parts = [("Todos", train), *train.groupby(CATEGORICAL, observed=True)]
    for ptype, part in parts:
        for col in NUMERIC:
            s = part[col].dropna()
            rows.append(
                {
                    "scope": ptype,
                    "column": col,
                    "n_rows": len(part),
                    "n": s.count(),
                    "mean": s.mean(),
                    "std": s.std(),
                    "min": s.min(),
                    "q25": s.quantile(0.25),
                    "median": s.median() if len(s) else np.nan,
                    "q75": s.quantile(0.75),
                    "max": s.max(),
                    "skew": s.skew(),
                    "kurtosis_excess": s.kurt(),
                    "reliable": ptype not in {"Type 4", "Type 5"} and s.count() >= 20,
                }
            )
    return pd.DataFrame(rows)


def null_report(train: pd.DataFrame) -> pd.DataFrame:
    """Report missingness for every received train column, including metadata."""
    assert_train(train)
    return pd.DataFrame(
        {
            "column": train.columns,
            "n_null": train.isna().sum().to_numpy(),
            "pct_null": 100 * train.isna().mean().to_numpy(),
        }
    )


def outlier_report(train: pd.DataFrame) -> pd.DataFrame:
    """Count descriptive IQR and robust-z outliers, without modifying any observation.

    If MAD is zero, median values have z=0 and nonmedian observed values z=infinity.
    Null values are not counted; a completely missing column has undefined thresholds.
    """
    assert_train(train)
    rows = []
    for col in NUMERIC:
        s = train[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        median = s.median() if len(s) else np.nan
        mad = (s - median).abs().median() if len(s) else np.nan
        if mad == 0:
            robust = (s != median).sum()
        else:
            robust = (0.67448975 * (s - median).abs() / mad > 3.5).sum()
        rows.append(
            {
                "column": col,
                "n": len(s),
                "iqr_low": q1 - 1.5 * iqr,
                "iqr_high": q3 + 1.5 * iqr,
                "iqr_outliers": ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum(),
                "mad": mad,
                "mad_zero": bool(mad == 0),
                "robust_z_outliers": robust,
            }
        )
    return pd.DataFrame(rows)


def temporal_blocks(train: pd.DataFrame) -> pd.Series:
    """Eight equal-width row-id blocks derived solely from train, with intact groups.

    Assign a repeated assay to the block containing its first train row. Block numbers
    are nuisance controls for EDA only and are never offered as predictor features.
    """
    assert_train(train)
    ids = train["_row_id"]
    if "strength_group_id" in train:
        ids = train.groupby("strength_group_id", observed=True)["_row_id"].transform("min")
    lo, hi = train._row_id.min(), train._row_id.max()
    blocks = np.floor(N_BLOCKS * (ids - lo) / (hi - lo + 1)).astype(int).clip(0, 7)
    return blocks.rename("block")


def _rank_residuals(values: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, axis=0, method="average")
    residuals = ranked.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        residuals[mask] -= ranked[mask].mean(axis=0)
    return residuals


def _correlation(x, y) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def correlation_report(train: pd.DataFrame, columns=FEATURES) -> pd.DataFrame:
    """Pairwise-complete Pearson, Spearman and block-partial Spearman with sample sizes."""
    assert_train(train)
    rows = []
    for scope, part in (("Todos", train), ("Type 1", train[train.ProductType == "Type 1"])):
        blocks = temporal_blocks(train).reindex(part.index)
        for feature in columns:
            for target in TARGETS:
                pair = part[[feature, target]].dropna()
                values = pair.to_numpy(dtype=float)
                for method in ("pearson", "spearman", "partial_spearman"):
                    if method == "partial_spearman" and scope != "Type 1":
                        continue
                    transformed = values
                    if len(pair) and method == "spearman":
                        transformed = rankdata(values, axis=0)
                    elif len(pair) and method == "partial_spearman":
                        transformed = _rank_residuals(values, blocks.loc[pair.index].to_numpy())
                    rho = _correlation(transformed[:, 0], transformed[:, 1])
                    rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "feature": feature,
                            "target": target,
                            "n": len(pair),
                            "rho": rho,
                        }
                    )
    return pd.DataFrame(rows)


def target_correlations(train: pd.DataFrame) -> pd.DataFrame:
    """Describe target dependence globally and within Type 1, with pairwise n."""
    assert_train(train)
    rows = []
    for scope, part in (("Todos", train), ("Type 1", train[train.ProductType == "Type 1"])):
        for method in ("pearson", "spearman"):
            for i, left in enumerate(TARGETS):
                for right in TARGETS[i + 1 :]:
                    pair = part[[left, right]].dropna()
                    value = pair.corr(method=method).iloc[0, 1]
                    rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "n_blocks": 0,
                            "left": left,
                            "right": right,
                            "n": len(pair),
                            "rho": value,
                        }
                    )
        if scope == "Type 1":
            for n_blocks in (8, 24, 48):
                for i, left in enumerate(TARGETS):
                    for right in TARGETS[i + 1 :]:
                        n, rho = rank_block_partial(part, left, right, [], n_blocks)
                        rows.append(
                            {
                                "scope": scope,
                                "method": "partial_spearman_rank_blocks",
                                "n_blocks": n_blocks,
                                "left": left,
                                "right": right,
                                "n": n,
                                "rho": rho,
                            }
                        )
    return pd.DataFrame(rows)


def rank_block_partial(
    train: pd.DataFrame, left: str, right: str, controls: list[str], n_blocks: int
) -> tuple[int, float]:
    """Residualize ranks on controls and row-order quantile blocks on complete cases.

    This sensitivity follows the chemical review: blocks are rebuilt per complete-case
    set from row-id ranks. Unlike temporal_blocks, assay groups may cross boundaries.
    """
    assert_train(train)
    complete = train[[left, right, *controls, "_row_id"]].dropna()
    if len(complete) < n_blocks:
        return len(complete), np.nan
    blocks = pd.qcut(complete._row_id.rank(), n_blocks, labels=False)
    ranks = rankdata(complete[[left, right, *controls]].to_numpy(), axis=0)
    design = np.column_stack([pd.get_dummies(blocks, dtype=float), ranks[:, 2:]])
    residuals = ranks[:, :2] - design @ np.linalg.lstsq(design, ranks[:, :2], rcond=None)[0]
    return len(complete), _correlation(residuals[:, 0], residuals[:, 1])


def blaine_sensitivity(train: pd.DataFrame) -> pd.DataFrame:
    """Report Type-1 block/control sensitivity without causal or inferential claims."""
    assert_train(train)
    part = train[train.ProductType == "Type 1"]
    rows = []
    for n_blocks in (8, 24, 48):
        for label, controls in (
            ("ninguno", []),
            ("D", ["Mineral_Phase_D"]),
            (
                "D + Chem_A + C + G + H",
                [
                    "Mineral_Phase_D",
                    "Chemical_Composition_A",
                    "Mineral_Phase_C",
                    "Mineral_Phase_G",
                    "Mineral_Phase_H",
                ],
            ),
        ):
            for target in ("Strength1", "Strength28"):
                n, rho = rank_block_partial(part, "Physical_Metric_B", target, controls, n_blocks)
                rows.append(
                    {"n_blocks": n_blocks, "controls": label, "target": target, "n": n, "rho": rho}
                )
    return pd.DataFrame(rows)


def compositional_features(train: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic phase candidates with explicit validity flags.

    Log-ratios shift by a constant under multiplicative rescaling, not additive shifts.
    F/(E+F) allows individual zeros and is invariant only to a common scale of E and F.
    The unknown original minima prevent physical identification.
    """
    assert_train(train)
    result = train.copy()
    for name, numerator, denominator, log in (
        (RATIOS[0], "C", "B", True),
        (RATIOS[1], "C", "A", False),
        (RATIOS[2], "D", "A", True),
        (RATIOS[3], "F", "E", True),
        ("log_H_over_F", "H", "F", True),
        ("log_H_over_A", "H", "A", True),
    ):
        top = train[f"Mineral_Phase_{numerator}"]
        bottom = train[f"Mineral_Phase_{denominator}"]
        valid = top.gt(0) & bottom.gt(0) & top.notna() & bottom.notna()
        ratio = top.where(valid) / bottom.where(valid)
        result[name] = np.log(ratio) if log else ratio
        result[f"flag_invalid_{name}"] = ~valid
    a = train["Mineral_Phase_A"]
    valid = a.gt(0) & a.notna()
    result["log_A"] = np.log(a.where(valid))
    result["flag_invalid_log_A"] = ~valid
    e, f = train["Mineral_Phase_E"], train["Mineral_Phase_F"]
    valid = e.ge(0) & f.ge(0) & (e + f).gt(0)
    result["F_over_E_plus_F"] = f.where(valid) / (e + f).where(valid)
    result["flag_invalid_F_over_E_plus_F"] = ~valid
    return result


def vif_report(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Types 1-3 VIF with intercept, complete cases and nominal type dummies."""
    assert_train(train)
    allowed = set(FEATURES + RATIOS + [CATEGORICAL])
    if not set(columns) <= allowed:
        raise ValueError("VIF accepts quality features and nominal type only")
    complete = train.loc[train.ProductType.isin(["Type 1", "Type 2", "Type 3"]), columns].dropna()
    matrix = (
        pd.get_dummies(complete, columns=[CATEGORICAL], drop_first=True, dtype=float)
        if CATEGORICAL in columns
        else complete.astype(float)
    )
    names = list(matrix.columns)
    if len(complete) <= len(names) + 1:
        return pd.DataFrame({"feature": names, "n_complete": len(complete), "vif": np.nan})
    x = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vifs = [variance_inflation_factor(x, i + 1) for i in range(len(names))]
    return pd.DataFrame({"feature": names, "n_complete": len(complete), "vif": vifs})


def reduce_vif(
    train: pd.DataFrame, columns: list[str], threshold=10.0
) -> tuple[list, pd.DataFrame]:
    """Propose a target-blind linear design by removing the largest VIF iteratively.

    Ties use column order. This is an EDA proposal to recheck per training fold in Phase 3.
    """
    assert_train(train)
    remaining = list(columns)
    history = []
    while len(remaining) > 1:
        table = vif_report(train, remaining)
        if table.vif.isna().any():
            raise ValueError("Insufficient complete cases for VIF proposal")
        worst = table.loc[table.vif.idxmax()]
        if worst.vif < threshold:
            break
        remaining.remove(worst.feature)
        history.append({"removed": worst.feature, "vif": worst.vif, "n_complete": worst.n_complete})
    return remaining, pd.DataFrame(history, columns=["removed", "vif", "n_complete"])


def bootstrap_partial(
    train: pd.DataFrame, feature: str, targets: list[str], n_bootstrap=BOOTSTRAPS
) -> dict:
    """Block-adjusted rank effects with stratified assay-cluster percentile bootstrap.

    Ranks and block means are recomputed in every draw. For two targets, report the paired
    difference rho(feature, first) - rho(feature, second), on the SAME complete cases.
    """
    assert_train(train)
    if feature not in FEATURES or not set(targets) <= set(TARGETS) or len(targets) not in (1, 2):
        raise ValueError("Expected one quality feature and one or two targets")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    part = train.loc[
        train.ProductType == "Type 1", [feature, *targets, "strength_group_id"]
    ].dropna()
    blocks = temporal_blocks(train).loc[part.index].to_numpy()
    values = part[[feature, *targets]].to_numpy(dtype=float)
    groups = part.strength_group_id.to_numpy()
    clusters = []
    for block in np.unique(blocks):
        clusters.append(
            [np.flatnonzero(groups == group) for group in np.unique(groups[blocks == block])]
        )

    def effect(indices):
        r = _rank_residuals(values[indices], blocks[indices])
        correlations = [_correlation(r[:, 0], r[:, i]) for i in range(1, r.shape[1])]
        return correlations[0] if len(targets) == 1 else correlations[0] - correlations[1]

    observed = effect(np.arange(len(part))) if len(part) >= 3 else np.nan
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_bootstrap):
        if not clusters:
            break
        indices = np.concatenate(
            [
                group
                for pool in clusters
                for group in [pool[i] for i in rng.integers(len(pool), size=len(pool))]
            ]
        )
        estimates.append(effect(indices))
    valid = np.asarray(estimates)
    valid = valid[np.isfinite(valid)]
    low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
    return {
        "feature": feature,
        "contrast": " - ".join(targets),
        "n": len(part),
        "n_groups": len(np.unique(groups)),
        "effect": observed,
        "ci_low": low,
        "ci_high": high,
        "valid_bootstraps": len(valid),
    }


def direction_verdict(low: float, high: float, expected: str) -> str:
    """Classify directional or equivalence evidence; crossing zero is not neutrality."""
    if not np.isfinite([low, high]).all():
        return "no concluyente"
    if expected == "neutral":
        if low >= -NEUTRAL_MARGIN and high <= NEUTRAL_MARGIN:
            return "apoya"
        if low > NEUTRAL_MARGIN or high < -NEUTRAL_MARGIN:
            return "no apoya"
        return "no concluyente"
    if expected == "+":
        return "apoya" if low > 0 else ("no apoya" if high < 0 else "no concluyente")
    if expected == "-":
        return "apoya" if high < 0 else ("no apoya" if low > 0 else "no concluyente")
    raise ValueError("Unknown expected direction")


def physics_tests(train: pd.DataFrame, n_bootstrap=BOOTSTRAPS) -> pd.DataFrame:
    """Evaluate prespecified physics signs and a practical |rho| <= .10 neutrality region."""
    assert_train(train)
    rows = []
    specifications = [
        ("Blaine R1", "Physical_Metric_B", ["Strength1"], "+"),
        ("Blaine R28", "Physical_Metric_B", ["Strength28"], "+"),
        ("Blaine R1 > R28", "Physical_Metric_B", ["Strength1", "Strength28"], "+"),
        ("D caliza R28", "Mineral_Phase_D", ["Strength28"], "-"),
        ("F C3A R1", "Mineral_Phase_F", ["Strength1"], "+"),
        *[(f"E ferrita {t}", "Mineral_Phase_E", [t], "neutral") for t in TARGETS],
    ]
    for name, feature, targets, expected in specifications:
        row = bootstrap_partial(train, feature, targets, n_bootstrap)
        row.update(
            {
                "test": name,
                "expected": expected,
                "verdict": direction_verdict(row["ci_low"], row["ci_high"], expected),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def top_features(correlations: pd.DataFrame) -> pd.DataFrame:
    """Rank top three raw features per target by absolute Type-1 Spearman, with n."""
    selected = correlations[
        (correlations.scope == "Type 1") & (correlations.method == "spearman")
    ].copy()
    selected["abs_rho"] = selected.rho.abs()
    return selected.sort_values("abs_rho", ascending=False).groupby("target").head(3)


def drift_report(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Type-1 medians and observed counts in the eight training row-id blocks."""
    assert_train(train)
    part = train[train.ProductType == "Type 1"].copy()
    part["block"] = temporal_blocks(train).loc[part.index]
    rows = []
    for block, subset in part.groupby("block"):
        for col in columns:
            rows.append(
                {
                    "block": block + 1,
                    "row_id_min": subset._row_id.min(),
                    "row_id_max": subset._row_id.max(),
                    "column": col,
                    "n": subset[col].count(),
                    "median": subset[col].median() if subset[col].notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def analyze(train: pd.DataFrame, n_bootstrap=BOOTSTRAPS) -> dict[str, pd.DataFrame]:
    """Build every descriptive table from an explicitly train-only frame."""
    from gcc_strength.kinetics import anchored_diagnostic, fit_samples, summarize_fits

    assert_train(train)
    ratios = compositional_features(train)
    correlations = correlation_report(train)
    top = top_features(correlations)
    final, removed = reduce_vif(train, BASE_FEATURES)
    combined, removed_combined = reduce_vif(ratios, BASE_FEATURES + RATIOS)
    final_design = [CATEGORICAL, *combined]
    # Check the actual nominal design too; the numeric-only proposal can mask type effects.
    design = vif_report(ratios, final_design)
    return {
        "01_descriptivos": descriptive_statistics(train),
        "02_nulos": null_report(train),
        "03_outliers": outlier_report(train),
        "04_correlaciones": correlations,
        "05_correlacion_targets": target_correlations(train),
        "06_vif_con_A": vif_report(train, RAW_VIF_FEATURES),
        "07_vif_sin_A": vif_report(train, BASE_FEATURES),
        "08_vif_crudo_propuesto": vif_report(train, final),
        "09_vif_descartes_crudos": removed,
        "10_top_features": top,
        "11_ratios_nulos": pd.DataFrame(
            [
                {
                    "feature": c,
                    "n_invalid": ratios[f"flag_invalid_{c}"].sum(),
                    "pct_invalid": 100 * ratios[f"flag_invalid_{c}"].mean(),
                }
                for c in RATIOS
            ]
        ),
        "12_correlaciones_ratios": correlation_report(ratios, RATIOS),
        "13_vif_ratios": vif_report(ratios, RATIOS),
        "14_vif_crudas_ratios": vif_report(ratios, BASE_FEATURES + RATIOS),
        "15_vif_descartes_combinados": removed_combined,
        "16_vif_combinado_numerico": vif_report(ratios, combined),
        "17_vif_combinado_con_tipo": design,
        "17b_vif_crudo_con_tipo": vif_report(train, [CATEGORICAL, *BASE_FEATURES]),
        "17c_vif_ratios_sustitucion": vif_report(ratios, [CATEGORICAL, *RATIO_DESIGN_FEATURES]),
        "17d_vif_nuevas_candidatas": pd.concat(
            [
                vif_report(ratios, [CATEGORICAL, *RATIO_DESIGN_FEATURES, candidate]).assign(
                    candidate=candidate
                )
                for candidate in RATIOS[4:]
            ],
            ignore_index=True,
        ),
        "18_cinetica": summarize_fits(fit_samples(train)),
        "18b_cinetica_anclada_R7": anchored_diagnostic(train),
        "18c_cobertura_cinetica": train.groupby(CATEGORICAL, observed=True)[TARGETS]
        .apply(
            lambda part: pd.Series(
                {"n_rows": len(part), "n_complete": part.notna().all(axis=1).sum()}
            )
        )
        .reset_index(),
        "19_fisica": physics_tests(train, n_bootstrap),
        "19b_sensibilidad_blaine": blaine_sensitivity(train),
        "20_deriva": drift_report(train, list(dict.fromkeys(TARGETS + top.feature.tolist()))),
        "21_tipos": train.groupby(CATEGORICAL, observed=True)
        .size()
        .rename("n")
        .reset_index()
        .assign(reliable=lambda d: ~d.ProductType.isin(["Type 4", "Type 5"])),
    }


def make_figures(train: pd.DataFrame, tables: dict, out_dir: Path) -> list[Path]:
    """Render compact train-only figures, capped at 24 PNGs at 90 dpi."""
    assert_train(train)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
        paths.append(path)

    # One moderate image per variable, with both required distribution views.
    for col in NUMERIC:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
        s = train[col].dropna()
        axes[0].hist(s, bins=25, color="#216e8a", edgecolor="white")
        axes[0].set_title(f"{col} · train (n={len(s)})", fontsize=9)
        axes[0].set_ylabel("Frecuencia")
        types = sorted(train.ProductType.unique())
        axes[1].boxplot(
            [train.loc[train.ProductType == t, col].dropna() for t in types],
            tick_labels=[t.replace("Type ", "T") for t in types],
            orientation="horizontal",
        )
        axes[1].set_title("Por tipo · T4/T5 no confiables", fontsize=9)
        save(fig, f"distribution_{col}")
    for target in TARGETS:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        selected = tables["10_top_features"].query("target == @target")
        part = train[train.ProductType == "Type 1"]
        for ax, row in zip(axes, selected.itertuples(), strict=True):
            ax.scatter(part[row.feature], part[target], s=8, alpha=0.45)
            ax.set_xlabel(row.feature, fontsize=8)
            ax.set_ylabel(target)
            ax.set_title(f"Type 1 · ρ={row.rho:.3f} · n={row.n}", fontsize=9)
        save(fig, f"scatter_{target}")
    for method in ("spearman", "partial_spearman"):
        part = tables["04_correlaciones"]
        matrix = (
            part[(part.scope == "Type 1") & (part.method == method)]
            .pivot(index="feature", columns="target", values="rho")
            .reindex(index=FEATURES, columns=TARGETS)
        )
        fig, ax = plt.subplots(figsize=(6.5, 5))
        heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        ax.set_xticks(range(3), matrix.columns, fontsize=8)
        ax.set_title(f"Type 1 · {method} · train", fontsize=10)
        for i in range(len(matrix)):
            for j in range(3):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(heat, ax=ax)
        save(fig, f"correlations_{method}")
    drift = tables["20_deriva"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for col in drift.column.unique():
        part = drift[drift.column == col]
        if col in TARGETS:
            axes[0].plot(part.block, part["median"], marker="o", label=col)
        else:
            values = part["median"]
            span = values.max() - values.min()
            axes[1].plot(
                part.block,
                (values - values.min()) / span if span else values * 0,
                marker=".",
                label=col,
            )
    for ax in axes:
        ax.set_xlabel("Bloque de orden (Type 1, train)")
        ax.legend(fontsize=6)
    axes[0].set_title("Mediana de targets")
    axes[1].set_title("Medianas de top features (rango visual 0–1)")
    save(fig, "drift")
    fits = tables["18_cinetica"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, target in zip(axes, TARGETS, strict=True):
        for model, offset in (("sqrt", -0.15), ("free", 0.15)):
            part = fits[fits.model == model]
            ax.bar(np.arange(len(part)) + offset, part[f"MAPE_{target}"], width=0.3, label=model)
            ax.set_xticks(np.arange(len(part)), part.ProductType, rotation=30, fontsize=7)
        ax.set_title(target)
        ax.set_ylabel("MAPE de ajuste in-sample (%)")
        ax.legend(fontsize=7)
    save(fig, "kinetics_fit_error")
    assert len(paths) <= 25
    return paths


def write_report(
    train: pd.DataFrame, out_dir: Path = DEFAULT_OUT, n_bootstrap=BOOTSTRAPS
) -> dict[str, pd.DataFrame]:
    """Regenerate tables and images; no Gold access occurs in the reporting path."""
    assert_train(train)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = analyze(train, n_bootstrap)
    paths = make_figures(train, tables, out_dir / "figures")
    text = [
        "# Fase 2: resultados descriptivos de train\n",
        "Generado con `uv run python -m gcc_strength.eda`. "
        f"Población: **{len(train)} filas de train**.",
        "No se usan valores del lockbox. Ninguna tabla es una métrica predictiva de CV.",
        "Método: casos observados por variable/pareja; asimetría muestral y curtosis de exceso.",
        "Spearman parcial: rangos de cada pareja residualizados contra "
        "intercepto + dummies de 8 bloques.",
        "Tabla 05 (parciales) y 19b: bloques por cuantiles del rango de `_row_id` "
        "dentro de Type 1 y de los casos completos de cada prueba (8/24/48); "
        "controles adicionales también en rangos. No conservan grupos íntegros. "
        "La tabla 19b es sensibilidad puntual, sin IC ni ajuste por multiplicidad.",
        "Bloques: anchura igual sobre min/max de `_row_id` en train; "
        "grupos íntegros asignados por su primera fila.",
        f"Física: IC percentil 95%, {n_bootstrap} remuestreos de grupos "
        "dentro de bloque; semilla 42.",
        "Se recalculan rangos en cada remuestreo; el contraste de edades "
        "usa casos completos comunes.",
        "La tabla 19 ajusta solo por bloque; 19b añade los controles indicados. "
        "Ninguna asociación demuestra causalidad.",
        "IC marginales exploratorios, sin corrección por multiplicidad; "
        "no resuelven dependencia serial residual.",
        "Neutralidad E: IC completo dentro de [−0.10, +0.10]; "
        "cruzar cero no demuestra equivalencia.",
        "VIF: solo Types 1–3, statsmodels, intercepto, casos completos por diseño; no imputación. "
        "`inf` indica singularidad.",
        "Cinética: mínimos cuadrados en unidades del target, 3 edades; f28>0, 0≤s≤5; 0.05≤a≤1.5.",
        "El exponente libre se comparte por tipo; tres inicios deterministas. Error **in-sample**.",
        "Dos parámetros por muestra para tres puntos, más un exponente compartido: "
        "los MAPE libres bajo el piso de ruido de conf/thresholds.yaml son "
        "sobreajuste por construcción, no evidencia predictiva. Curva: interpolación.",
        "MAD=0: valores distintos de la mediana reciben z robusto infinito. "
        "No se elimina ningún outlier.",
        "Logs/cocientes positivos: operandos no positivos o faltantes → NaN + bandera. "
        "F/(E+F) admite E/F iguales a cero si E+F>0 y ambos son no negativos. "
        "Sin epsilon ni imputación. Tabla 17d: una candidata añadida a la vez.",
        "Los ratios presuponen mínimos físicos nulos, todavía no confirmados; "
        "no hay cierre composicional.",
        "Types 4 y 5: descriptivos no confiables. "
        "Un target sin observaciones no admite ajuste cinético.\n",
    ]
    for name, table in tables.items():
        text.extend([f"\n## {name}\n", table.to_markdown(index=False, floatfmt=".4f"), ""])
    text.append("\n## 22_figuras\n")
    text.extend(f"- [{path.stem}](figures/{path.name})" for path in paths)
    (out_dir / "eda_results.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    train = load_train(args.gold)
    write_report(train, args.out)
    print(f"EDA complete: {len(train)} train rows -> {args.out}")


if __name__ == "__main__":
    main()
