"""Stage 3 architecture comparison, nested CV+ and train-only sanity diagnostics.

Run ``uv run python -m gcc_strength.stage3``. No ingestion path other than load_train.
Cached tasks require identical source, configuration, finalists, versions and train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT, load_yaml
from gcc_strength.cv import load_model_config, repeated_group_splits, temporal_group_splits
from gcc_strength.eda import assert_train, load_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.metrics import compare_oof, paired_mae_bootstrap, regression_metrics
from gcc_strength.multioutput import fit_architecture
from gcc_strength.split import GROUP_COL
from gcc_strength.uncertainty import coverage_table, nested_cvplus

DEFAULT_OUT = ROOT / "outputs/fase3/stage3"
DEFAULT_FINALISTS = ROOT / "outputs/fase3/stage2/finalists.csv"
SCORES = ("mae", "r2", "mape_pct", "rmse", "mae_type1", "r2_type1", "train_mae", "overfit_gap")


def read_finalists(path):
    """Read the frozen algorithm/representation/A choices, with exactly one per output."""
    finalists = pd.read_csv(path)
    expected = {
        (family, target)
        for family in ("linear", "lightgbm", "gaussian_process")
        for target in TARGETS
    }
    if set(zip(finalists.family, finalists.target, strict=True)) != expected or len(finalists) != 9:
        raise ValueError("Exactly three frozen families x three targets are required")
    return {
        family: [part.set_index("target").loc[target].to_dict() for target in TARGETS]
        for family, part in finalists.groupby("family", sort=True)
    }


def _score_frame(frame, predicted, **identifiers):
    rows, folds = [], []
    for j, target in enumerate(TARGETS):
        observed = frame[target].notna().to_numpy()
        part = frame.loc[observed, ["_row_id", GROUP_COL, "ProductType"]].copy()
        part["observed"] = frame.loc[observed, target].to_numpy()
        part["predicted"] = predicted[observed, j]
        part["target"] = target
        for key, value in identifiers.items():
            part[key] = value
        rows.append(part)
        folds.append(
            dict(
                target=target,
                **identifiers,
                **regression_metrics(part.observed, part.predicted, part.ProductType),
            )
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(folds)


def evaluate_fold(train, specs, architecture, config, fit, valid, identifiers):
    """Evaluate all three targets on common, intact outer validation groups."""
    assert_train(train)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        model, selection = fit_architecture(train.iloc[fit], specs, architecture, config)
        prediction = model.predict(train.iloc[valid][list(INPUT_COLUMNS)])
        oof, scores = _score_frame(train.iloc[valid], prediction, **identifiers)
        _, training = _score_frame(
            train.iloc[fit], model.predict(train.iloc[fit][list(INPUT_COLUMNS)])
        )
        scores["train_mae"] = training.mae.to_numpy()
        scores["train_r2"] = training.r2.to_numpy()
        scores["overfit_gap"] = scores.train_r2 - scores.r2
        scores["train_n"] = training.n.to_numpy()
        scores["monotonicity"] = float(np.all(np.diff(prediction, axis=1) >= 0, axis=1).mean())
    selection["warnings"] = dict(
        pd.Series([type(w.message).__name__ for w in caught]).value_counts().items()
    )
    return oof, scores, selection


def _cached_task(directory, kind, train, specs, architecture, config, fit, valid, identifiers):
    directory = Path(directory)
    if (directory / "complete").is_file():
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "comparison":
        oof, scores, selection = evaluate_fold(
            train, specs, architecture, config, fit, valid, identifiers
        )
        oof.to_csv(directory / "oof.csv", index=False)
        scores.to_csv(directory / "folds.csv", index=False)
        (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    else:
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            lower, upper = nested_cvplus(
                train.iloc[fit], train.iloc[valid][list(INPUT_COLUMNS)], specs, architecture, config
            )
        observed = train.iloc[valid]
        parts = []
        for j, target in enumerate(TARGETS):
            mask = observed[target].notna().to_numpy()
            part = observed.loc[mask, ["_row_id", GROUP_COL, "ProductType"]].copy()
            part["target"], part["observed"] = target, observed.loc[mask, target].to_numpy()
            part["lower"], part["upper"] = lower[mask, j], upper[mask, j]
            part["covered"] = part.observed.between(part.lower, part.upper)
            for key, value in identifiers.items():
                part[key] = value
            parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(directory / "intervals.csv", index=False)
        (directory / "warnings.json").write_text(
            json.dumps(
                dict(pd.Series([type(w.message).__name__ for w in caught]).value_counts().items())
            )
            + "\n"
        )
    (directory / "complete").write_text("ok\n")
    return str(directory)


def _execute(tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(_cached_task, *args) for args in tasks]
        for count, future in enumerate(as_completed(pending), 1):
            print(f"Completed {count}/{len(tasks)}: {future.result()}", flush=True)


def _summarize(folds):
    result = (
        folds.groupby(["family", "architecture", "scheme", "target"], sort=True)[list(SCORES)]
        .agg(["mean", "std"])
        .reset_index()
    )
    result.columns = ["_".join(c).rstrip("_").replace("_std", "_sd") for c in result.columns]
    return result


def select_architecture(oof, config):
    """Require paired superiority over reference (a); (c) must also pass the R1 gate."""
    repeated = oof.loc[oof.scheme.eq("repeated_group")]
    rows = []
    for family, part in repeated.groupby("family", sort=True):
        reference = part.loc[part.architecture.eq("a")]
        for arch in ("b", "c"):
            alternative = part.loc[part.architecture.eq(arch)]
            comparisons = {}
            for target in TARGETS:
                comparisons[target] = compare_oof(
                    alternative.loc[alternative.target.eq(target)],
                    reference.loc[reference.target.eq(target)],
                    n_bootstrap=config["bootstrap"]["n_resamples"],
                )
            keys = ["target", "repeat", "_row_id", GROUP_COL]
            merged = alternative.merge(
                reference, on=keys, validate="one_to_one", suffixes=("", "_a")
            )
            total = paired_mae_bootstrap(
                merged.observed,
                merged.predicted,
                merged.predicted_a,
                merged[GROUP_COL],
                n_bootstrap=config["bootstrap"]["n_resamples"],
            )
            eligible = total["ci_high"] < 0 and (
                arch != "c" or comparisons[TARGETS[0]]["ci_high"] <= 0
            )
            for target, comparison in comparisons.items():
                rows.append(
                    dict(
                        family=family,
                        architecture=arch,
                        target=target,
                        eligible=eligible,
                        **comparison,
                    )
                )
            rows.append(
                dict(
                    family=family,
                    architecture=arch,
                    target="mean_targets",
                    eligible=eligible,
                    **total,
                )
            )
    gates = pd.DataFrame(rows)
    means = (
        repeated.assign(error=lambda p: abs(p.predicted - p.observed))
        .groupby(["family", "architecture"])
        .error.mean()
    )
    choices = []
    for family in repeated.family.unique():
        eligible = [
            "a",
            *gates.loc[
                gates.family.eq(family) & gates.target.eq("mean_targets") & gates.eligible,
                "architecture",
            ].tolist(),
        ]
        winner = min(eligible, key=lambda a: (means[family, a], a))
        choices.append(dict(family=family, architecture=winner, mae=float(means[family, winner])))
    selected = min(choices, key=lambda r: (r["mae"], r["family"]))
    return gates, {"selected": selected, "by_family": choices, "status": "provisional_cv_selection"}


def _group_subset(frame, fraction):
    groups = np.sort(frame[GROUP_COL].unique())
    order = np.random.default_rng(42).permutation(groups)
    selected = order[: max(10, int(np.ceil(len(groups) * fraction)))]
    return frame.loc[frame[GROUP_COL].isin(selected)].copy()


def permute_group_labels(train, rng):
    """Shuffle joint target/missingness blocks between equally sized assay groups."""
    assert_train(train)
    result = train.copy()
    groups = [part.index.to_numpy() for _, part in train.groupby(GROUP_COL, sort=True)]
    for size in sorted({len(g) for g in groups}):
        equal = [g for g in groups if len(g) == size]
        for destination, source_index in zip(equal, rng.permutation(len(equal)), strict=True):
            result.loc[destination, TARGETS] = train.loc[equal[source_index], TARGETS].to_numpy()
    return result


def _sanity_task(
    directory,
    kind,
    train,
    specs,
    architecture,
    config,
    fit,
    valid,
    identifiers,
    fraction=1.0,
    permutation=0,
):
    directory = Path(directory)
    if (directory / "complete").exists():
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "learning":
        subset = _group_subset(train.iloc[fit], fraction)
        # Preserve the same validation population at every fraction.
        combined = pd.concat([subset, train.iloc[valid]])
        combined.attrs = train.attrs.copy()
        oof, scores, selection = evaluate_fold(
            combined,
            specs,
            architecture,
            config,
            np.arange(len(subset)),
            np.arange(len(subset), len(combined)),
            identifiers,
        )
    else:
        shuffled = permute_group_labels(
            train, np.random.default_rng(np.random.SeedSequence([42, permutation]))
        )
        oof, scores, selection = evaluate_fold(
            shuffled, specs, architecture, config, fit, valid, identifiers
        )
    oof.to_csv(directory / "oof.csv", index=False)
    scores.to_csv(directory / "folds.csv", index=False)
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (directory / "complete").write_text("ok\n")
    return str(directory)


def run_stage3(
    train, *, output=DEFAULT_OUT, finalists_path=DEFAULT_FINALISTS, config=None, workers=2
):
    """Run the declared CV comparison and diagnostics, with resumable per-fold artifacts."""
    assert_train(train)
    config = load_model_config() if config is None else config
    train = train.sort_values("_row_id", kind="stable").reset_index(drop=True)
    families = read_finalists(finalists_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = (
        "stage3.py",
        "multioutput.py",
        "uncertainty.py",
        "models.py",
        "features.py",
        "cv.py",
        "metrics.py",
        "split.py",
        "eda.py",
    )
    manifest = dict(
        config=config,
        finalists_sha256=hashlib.sha256(Path(finalists_path).read_bytes()).hexdigest(),
        train_sha256=hashlib.sha256(
            pd.util.hash_pandas_object(train, index=True).values.tobytes()
        ).hexdigest(),
        sources={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in sources
        },
        versions={
            name: version(name) for name in ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm")
        },
        lockbox_accessed=False,
    )
    previous = output / "manifest.json"
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError("Output manifest differs; use a fresh output directory")
    previous.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    folds = list(
        repeated_group_splits(
            train, n_splits=config["cv"]["n_splits"], n_repeats=config["cv"]["n_repeats"]
        )
    )
    temporal = list(temporal_group_splits(train, n_splits=config["cv"]["temporal_splits"]))
    tasks, comparison_paths = [], []
    for family, specs in families.items():
        for architecture in ("a", "b", "c"):
            for scheme, partitions in (("repeated_group", folds), ("temporal", temporal)):
                for repeat, fold, fit, valid in partitions:
                    ids = dict(
                        family=family,
                        architecture=architecture,
                        scheme=scheme,
                        repeat=repeat,
                        fold=fold,
                    )
                    directory = (
                        output / "comparison" / family / architecture / scheme / f"{repeat}_{fold}"
                    )
                    comparison_paths.append(directory)
                    tasks.append(
                        (
                            directory,
                            "comparison",
                            train,
                            specs,
                            architecture,
                            config,
                            fit,
                            valid,
                            ids,
                        )
                    )
    _execute(tasks, workers)
    oof = pd.concat([pd.read_csv(p / "oof.csv") for p in comparison_paths], ignore_index=True)
    scores = pd.concat([pd.read_csv(p / "folds.csv") for p in comparison_paths], ignore_index=True)
    oof.to_csv(output / "oof.csv", index=False)
    scores.to_csv(output / "folds.csv", index=False)
    summary = _summarize(scores)
    floors = load_yaml(ROOT / "conf/thresholds.yaml")["noise_floor_mape_pct_upper_bound"]
    summary["noise_floor_alarm"] = summary.mape_pct_mean < summary.target.map(floors)
    summary.to_csv(output / "comparison.csv", index=False)
    gates, decision = select_architecture(oof, config)
    gates.to_csv(output / "architecture_gate.csv", index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    selected = decision["selected"]
    family, architecture = selected["family"], selected["architecture"]
    specs = families[family]
    tasks, interval_paths = [], []
    for repeat, fold, fit, valid in folds:
        directory = output / "conformal" / f"{repeat}_{fold}"
        interval_paths.append(directory)
        ids = dict(
            family=family,
            architecture=architecture,
            scheme="repeated_group",
            repeat=repeat,
            fold=fold,
        )
        tasks.append((directory, "conformal", train, specs, architecture, config, fit, valid, ids))
    _execute(tasks, workers)
    intervals = pd.concat(
        [pd.read_csv(p / "intervals.csv") for p in interval_paths], ignore_index=True
    )
    intervals.to_csv(output / "intervals.csv", index=False)
    coverage_table(intervals).to_csv(output / "coverage.csv", index=False)
    # Sanity diagnostics use the first outer repeat: five intact group folds.
    diagnostic_folds = [f for f in folds if f[0] == 0]
    futures, learning_paths, permutation_paths = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for repeat, fold, fit, valid in diagnostic_folds:
            for fraction in config["stage3"]["learning_fractions"]:
                directory = output / "learning" / f"{fraction}_{fold}"
                learning_paths.append(directory)
                ids = dict(
                    family=family,
                    architecture=architecture,
                    repeat=repeat,
                    fold=fold,
                    fraction=fraction,
                )
                futures.append(
                    executor.submit(
                        _sanity_task,
                        directory,
                        "learning",
                        train,
                        specs,
                        architecture,
                        config,
                        fit,
                        valid,
                        ids,
                        fraction,
                    )
                )
            for permutation in range(config["stage3"]["permutations"]):
                directory = output / "permutation" / f"{permutation}_{fold}"
                permutation_paths.append(directory)
                ids = dict(
                    family=family,
                    architecture=architecture,
                    repeat=repeat,
                    fold=fold,
                    permutation=permutation,
                )
                futures.append(
                    executor.submit(
                        _sanity_task,
                        directory,
                        "permutation",
                        train,
                        specs,
                        architecture,
                        config,
                        fit,
                        valid,
                        ids,
                        1.0,
                        permutation,
                    )
                )
        for count, future in enumerate(as_completed(futures), 1):
            print(f"Sanity {count}/{len(futures)}: {future.result()}", flush=True)
    learning = pd.concat([pd.read_csv(p / "folds.csv") for p in learning_paths], ignore_index=True)
    learning.to_csv(output / "learning_curves.csv", index=False)
    permutation_scores = pd.concat(
        [pd.read_csv(p / "folds.csv") for p in permutation_paths], ignore_index=True
    )
    permutation_scores.to_csv(output / "permutation_folds.csv", index=False)
    null = permutation_scores.groupby(["target", "permutation"]).r2.mean()
    observed = (
        scores.loc[
            scores.family.eq(family)
            & scores.architecture.eq(architecture)
            & scores.scheme.eq("repeated_group")
            & scores.repeat.eq(0)
        ]
        .groupby("target")
        .r2.mean()
    )
    permutation_table = []
    for target in TARGETS:
        values = null.loc[target].to_numpy()
        permutation_table.append(
            dict(
                target=target,
                observed_r2=observed[target],
                shuffled_r2_mean=values.mean(),
                shuffled_r2_sd=values.std(ddof=1),
                p_value=(1 + np.sum(values >= observed[target])) / (len(values) + 1),
                n_permutations=len(values),
                strongly_positive_null=bool(values.mean() > 0.1),
            )
        )
    pd.DataFrame(permutation_table).to_csv(output / "permutation.csv", index=False)
    repeated = summary.loc[summary.scheme.eq("repeated_group")]
    time = summary.loc[summary.scheme.eq("temporal")]
    sensitivity = repeated.merge(
        time, on=["family", "architecture", "target"], suffixes=("_cv", "_temporal")
    )
    sensitivity["temporal_minus_cv_mae"] = sensitivity.mae_mean_temporal - sensitivity.mae_mean_cv
    sensitivity.to_csv(output / "temporal_sensitivity.csv", index=False)
    _plot_learning(learning, output)
    (output / "complete").write_text("ok\n")
    return decision


def _plot_learning(learning, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for target, ax in zip(TARGETS, axes, strict=True):
        part = learning.loc[learning.target.eq(target)].groupby("fraction")
        for column, label in (("mae", "CV"), ("train_mae", "Entrenamiento")):
            ax.errorbar(
                part.train_n.mean(),
                part[column].mean(),
                yerr=part[column].std(),
                marker="o",
                label=label,
            )
        ax.set(title=target, xlabel="Etiquetas de entrenamiento", ylabel="MAE (unidad original)")
        ax.legend()
    fig.savefig(output / "learning_curves.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--finalists", type=Path, default=DEFAULT_FINALISTS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_stage3(
        load_train(), output=args.output, finalists_path=args.finalists, workers=args.workers
    )


if __name__ == "__main__":
    main()
