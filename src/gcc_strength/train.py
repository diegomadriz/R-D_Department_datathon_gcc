"""Execute frozen Phase 3 finalists on training groups and log complete MLflow runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin
from threadpoolctl import threadpool_limits

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT, load_yaml
from gcc_strength.cv import load_model_config, repeated_group_splits, temporal_group_splits
from gcc_strength.eda import DEFAULT_GOLD, assert_train, load_train
from gcc_strength.explain import explain_model, physics_consistency_report
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.metrics import compare_oof, paired_mae_bootstrap, within_type_r2_bootstrap
from gcc_strength.models import build_model, design_size
from gcc_strength.multioutput import FixedArchitecture
from gcc_strength.provenance import train_fingerprint
from gcc_strength.split import GROUP_COL
from gcc_strength.stage3 import _group_subset, _score_frame, permute_group_labels
from gcc_strength.uncertainty import coverage_table, cvplus_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/fase3/final"


class ProjectedTarget(RegressorMixin, BaseEstimator):
    """Expose one output of a fitted physically coherent architecture."""

    def __init__(self, architecture, output_index):
        self.architecture = architecture
        self.output_index = output_index

    def predict(self, X):
        return self.architecture.predict(X)[:, self.output_index]


def _fit(frame, specs):
    profile = tuple(s["params"] for s in specs)
    return FixedArchitecture(specs, "a", profile).fit(frame[list(INPUT_COLUMNS)], frame)


def _evaluate(frame, specs, fit, valid, ids):
    model = _fit(frame.iloc[fit], specs)
    oof, scores = _score_frame(
        frame.iloc[valid], model.predict(frame.iloc[valid][list(INPUT_COLUMNS)]), **ids
    )
    _, training = _score_frame(frame.iloc[fit], model.predict(frame.iloc[fit][list(INPUT_COLUMNS)]))
    scores["train_mae"] = training.mae.to_numpy()
    scores["overfit_gap"] = training.r2.to_numpy() - scores.r2.to_numpy()
    scores["train_n"] = training.n.to_numpy()
    return model, oof, scores


def _cvplus(frame, future, specs, splits, alpha):
    """Calibrate only inside an outer training fold, using frozen parameter profiles."""
    centers, residuals = [[] for _ in TARGETS], [[] for _ in TARGETS]
    for _, _, fit, calibration in repeated_group_splits(frame, n_splits=splits, n_repeats=1):
        model = _fit(frame.iloc[fit], specs)
        held = frame.iloc[calibration]
        prediction = model.predict(held[list(INPUT_COLUMNS)])
        next_prediction = model.predict(future[list(INPUT_COLUMNS)])
        for j, target in enumerate(TARGETS):
            scores = pd.DataFrame(
                {"group": held[GROUP_COL].to_numpy(), "score": abs(held[target] - prediction[:, j])}
            ).dropna()
            scores = scores.groupby("group", sort=True).score.max().to_numpy()
            residuals[j].extend(scores)
            centers[j].extend(np.tile(next_prediction[:, j], (len(scores), 1)))
    rows = []
    for j, target in enumerate(TARGETS):
        low, high = cvplus_bounds(centers[j], residuals[j], alpha=alpha)
        mask = future[target].notna().to_numpy()
        part = future.loc[mask, ["_row_id", GROUP_COL, "ProductType"]].copy()
        part["target"] = target
        part["observed"] = future.loc[mask, target].to_numpy()
        part["lower"], part["upper"] = low[mask], high[mask]
        part["covered"] = part.observed.between(part.lower, part.upper)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _permutation_task(args):
    frame, specs, folds, permutation = args
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shuffled = permute_group_labels(
            frame, np.random.default_rng(np.random.SeedSequence([42, permutation]))
        )
        scores = []
        for repeat, fold, fit, valid in folds:
            _, _, result = _evaluate(shuffled, specs, fit, valid, dict(repeat=repeat, fold=fold))
            scores.append(result)
        means = pd.concat(scores).groupby("target").r2.mean()
        return [dict(target=t, permutation=permutation, r2=float(means[t])) for t in TARGETS]


def _savefig(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _table_plot(table, path):
    columns = ["family", "target", "cv_r2_mean", "cv_mae_mean", "cv_mape_pct_mean"]
    display = table[columns].copy()
    for col in columns[2:]:
        display[col] = display[col].map(lambda value: f"{value:.4f}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    grid = ax.table(cellText=display.values, colLabels=display.columns, loc="center")
    grid.auto_set_font_size(False)
    grid.set_fontsize(9)
    grid.scale(1, 1.5)
    _savefig(fig, path)


def _artifacts(oof, learning, shap, physics, directory):
    """Render actual diagnostics, with a panel per target for the multi-output run."""
    directory.mkdir(parents=True, exist_ok=True)
    oof.to_csv(directory / "oof.csv", index=False)
    learning.to_csv(directory / "learning_curve.csv", index=False)
    shap.to_csv(directory / "shap.csv", index=False)
    physics.to_csv(directory / "physics.csv", index=False)
    targets = [t for t in TARGETS if t in set(oof.target)]
    for name in ("oof_actual_vs_predicted", "residuals", "residuals_by_type", "learning_curve"):
        fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
        for target, ax in zip(targets, axes[0], strict=True):
            part = oof.loc[oof.target.eq(target)]
            error = part.predicted - part.observed
            if name == "oof_actual_vs_predicted":
                ax.scatter(part.observed, part.predicted, s=5, alpha=0.3)
                lo, hi = part.observed.min(), part.observed.max()
                ax.plot([lo, hi], [lo, hi], color="black")
                ax.set(xlabel="Observed", ylabel="OOF predicted")
            elif name == "residuals":
                ax.hist(error, bins=25)
                ax.set(xlabel="OOF predicted - observed", ylabel="Count")
            elif name == "residuals_by_type":
                labels = sorted(part.ProductType.unique())
                ax.boxplot([error.loc[part.ProductType.eq(t)] for t in labels], tick_labels=labels)
                ax.set(ylabel="OOF predicted - observed")
            else:
                curve = learning.loc[learning.target.eq(target)].groupby("fraction")
                for col, label in (("mae", "Validation"), ("train_mae", "Train")):
                    ax.plot(curve.train_n.mean(), curve[col].mean(), "o-", label=label)
                ax.set(xlabel="Observed training labels", ylabel="MAE")
                ax.legend()
            ax.set_title(target)
        fig.tight_layout()
        _savefig(fig, directory / f"{name}.png")
    importance = (
        shap.assign(importance=lambda f: f.shap.abs())
        .groupby(["target", "branch", "feature"], as_index=False)
        .importance.mean()
    )
    importance.to_csv(directory / "importance.csv", index=False)
    for name in ("importance", "shap_global", "shap_local"):
        fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5), squeeze=False)
        for target, ax in zip(targets, axes[0], strict=True):
            if name == "shap_local":
                part = shap.loc[shap.target.eq(target)]
                part = part.loc[part.row.eq(part.row.iloc[0])].copy()
                part["importance"] = part.shap
            else:
                part = importance.loc[importance.target.eq(target)].copy()
            part["label"] = part.branch + "/" + part.feature
            top = part.loc[part.importance.abs().nlargest(12).index].iloc[::-1]
            ax.barh(top.label, top.importance)
            ax.set(
                title=target, xlabel="SHAP contribution" if name == "shap_local" else "Mean |SHAP|"
            )
        fig.tight_layout()
        _savefig(fig, directory / f"{name}.png")


def _log_child(name, target, fitted, n_features, metrics, artifacts, example, dataset, params):
    with mlflow.start_run(run_name=f"{name}/{target}", nested=True):
        mlflow.log_params(
            {**params, "model_type": name, "target": target, "n_features": n_features}
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        mlflow.log_input(dataset, context="training")
        mlflow.log_artifacts(str(artifacts))
        logged = mlflow.sklearn.log_model(
            fitted,
            name="model",
            signature=infer_signature(example, fitted.predict(example)),
            input_example=example,
            pip_requirements=[
                "scikit-learn==1.7.2",
                "pandas==2.3.3",
                "numpy==2.3.4",
                "lightgbm==4.6.0",
                "scipy==1.16.3",
                "statsmodels==0.14.5",
                "matplotlib==3.10.6",
                "mlflow-skinny==3.12.0",
                "pyyaml==6.0.3",
            ],
            code_paths=[str(ROOT / "src/gcc_strength")],
        )
        return logged.model_uri


def _rounded(value):
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def source_revision():
    """Use Git locally and a content digest in synced workspace files without .git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return "source-sha256:" + digest.hexdigest()


def run_training(
    *,
    quick=False,
    tracking_uri="file:./mlruns",
    experiment=None,
    source=DEFAULT_GOLD,
    output=DEFAULT_OUTPUT,
    permutation_workers=2,
):
    """Compute every diagnostic afresh; the sole data entry point is load_train."""
    started = time.monotonic()
    config = load_model_config()
    final = config["final"]
    if final["architecture"] != "a":
        raise ValueError("Only independent models with isotonic projection are frozen")
    train = load_train(source).sort_values("_row_id", kind="stable").reset_index(drop=True)
    assert_train(train)
    source_train_size = len(train)
    if source_train_size != 621:
        raise ValueError("The frozen training population must contain 621 rows")
    fitting_train = train.copy()
    if quick:
        train = _group_subset(train, 0.15).reset_index(drop=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n_splits, n_repeats = (2, 2) if quick else (config["cv"]["n_splits"], config["cv"]["n_repeats"])
    partitions = list(repeated_group_splits(train, n_splits=n_splits, n_repeats=n_repeats))
    diagnostic_folds = [p for p in partitions if p[0] == 0]
    temporal = list(temporal_group_splits(train, n_splits=2 if quick else 5))
    n_bootstrap = 100 if quick else config["bootstrap"]["n_resamples"]
    families = {f: [s[t] for t in TARGETS] for f, s in final["families"].items()}
    selected = final["selected_family"]
    params = dict(
        train_size=source_train_size,
        test_size=train.attrs["lockbox_size"],
        seed=42,
        config_hash=hashlib.sha256((ROOT / "conf/models.yaml").read_bytes()).hexdigest(),
        git_commit=source_revision(),
        lockbox_sha256=(ROOT / "conf/lockbox.sha256").read_text().strip(),
        quick=quick,
        fit_size=len(train),
        model_fit_size=len(fitting_train),
        model_train_sha256=train_fingerprint(fitting_train),
        cv_splits=n_splits,
        cv_repeats=n_repeats,
    )
    provenance = dict(
        **params,
        train_sha256=hashlib.sha256(
            pd.util.hash_pandas_object(train, index=True).values.tobytes()
        ).hexdigest(),
        source_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    tables, coverages, permutation_table = [], [], []
    all_oof, all_folds, all_temporal, all_learning = [], [], [], []
    fitted_models, explanations, physics_reports = {}, {}, {}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment or ("gcc-phase3-final-quick" if quick else "gcc-phase3-final"))
    with mlflow.start_run(run_name="final-training") as parent, threadpool_limits(limits=1):
        parent_id = parent.info.run_id
        mlflow.log_params(params)
        dataset = mlflow.data.from_pandas(
            fitting_train[[*INPUT_COLUMNS, *TARGETS]],
            name="gcc_gold_train",
        )
        mlflow.log_input(dataset, context="training")
        for family, specs in families.items():
            print(f"[{time.monotonic() - started:.1f}s] {family}: CV and CV+", flush=True)
            oof_parts, fold_parts, intervals, baselines = [], [], [], []
            for repeat, fold, fit, valid in partitions:
                ids = dict(family=family, repeat=repeat, fold=fold)
                _, oof, scores = _evaluate(train, specs, fit, valid, ids)
                oof_parts.append(oof)
                fold_parts.append(scores)
                interval = _cvplus(
                    train.iloc[fit], train.iloc[valid], specs, 2 if quick else 5, 0.1
                )
                interval["repeat"], interval["fold"], interval["family"] = repeat, fold, family
                intervals.append(interval)
                prediction = []
                for target in TARGETS:
                    known = train.iloc[fit].loc[train.iloc[fit][target].notna()]
                    base = build_model("L1").fit(known[list(INPUT_COLUMNS)], known[target])
                    prediction.append(base.predict(train.iloc[valid][list(INPUT_COLUMNS)]))
                base_oof, _ = _score_frame(train.iloc[valid], np.column_stack(prediction), **ids)
                baselines.append(base_oof)
            oof, folds, baseline = map(
                lambda parts: pd.concat(parts, ignore_index=True),
                (oof_parts, fold_parts, baselines),
            )
            intervals = pd.concat(intervals, ignore_index=True)
            intervals.to_csv(output / f"{family}_intervals.csv", index=False)
            coverage = coverage_table(intervals)
            coverage["family"] = family
            coverages.append(coverage)
            timed, learning = [], []
            for repeat, fold, fit, valid in temporal:
                _, _, scores = _evaluate(
                    train, specs, fit, valid, dict(family=family, repeat=repeat, fold=fold)
                )
                timed.append(scores)
            for repeat, fold, fit, valid in diagnostic_folds:
                for fraction in [0.5, 1.0] if quick else [0.25, 0.5, 0.75, 1.0]:
                    subset = _group_subset(train.iloc[fit], fraction)
                    combined = pd.concat([subset, train.iloc[valid]])
                    combined.attrs = train.attrs.copy()
                    _, _, scores = _evaluate(
                        combined,
                        specs,
                        np.arange(len(subset)),
                        np.arange(len(subset), len(combined)),
                        dict(family=family, repeat=repeat, fold=fold, fraction=fraction),
                    )
                    learning.append(scores)
            timed, learning = (
                pd.concat(timed, ignore_index=True),
                pd.concat(learning, ignore_index=True),
            )
            model = _fit(fitting_train, specs)
            fitted_models[family] = model
            explanation_parts, physics_parts = [], []
            for j, target in enumerate(TARGETS):
                part = oof.loc[oof.target.eq(target)]
                scores = folds.loc[folds.target.eq(target)]
                row = dict(family=family, target=target)
                for metric in ("r2", "mae", "mape_pct", "rmse", "r2_type1", "mae_type1"):
                    row.update(
                        {
                            f"cv_{metric}_mean": scores[metric].mean(),
                            f"cv_{metric}_sd": scores[metric].std(ddof=1),
                        }
                    )
                row["overfit_gap"] = scores.overfit_gap.mean()
                row["coverage"] = coverage.loc[
                    coverage.target.eq(target) & coverage.scope.eq("Todos"), "coverage"
                ].iloc[0]
                row["temporal_minus_cv_mae"] = (
                    timed.loc[timed.target.eq(target)].mae.mean() - scores.mae.mean()
                )
                delta = compare_oof(
                    part, baseline.loc[baseline.target.eq(target)], n_bootstrap=n_bootstrap
                )
                row.update(
                    {
                        f"delta_{key}" if key != "delta_mae" else key: value
                        for key, value in delta.items()
                    }
                )
                ci = paired_mae_bootstrap(
                    part.observed,
                    part.predicted,
                    part.observed,
                    part[GROUP_COL],
                    n_bootstrap=n_bootstrap,
                )
                row["mae_ci_low"], row["mae_ci_high"] = ci["ci_low"], ci["ci_high"]
                within = within_type_r2_bootstrap(part, n_bootstrap=n_bootstrap)
                for number, values in within.items():
                    for key, value in values.items():
                        row[f"oof_r2_type{number}_{key}"] = value
                row["rule_within_type_pass"] = bool(within[1]["r2"] > 0 and within[1]["ci_low"] > 0)
                tables.append(row)
                explanation = explain_model(
                    model.models_[j],
                    train,
                    max_samples=4 if quick else 32,
                    background_size=4 if quick else 16,
                )
                physics = physics_consistency_report(explanation, target)
                explanation["target"] = target
                explanation_parts.append(explanation)
                physics_parts.append(physics)
            explanations[family] = pd.concat(explanation_parts, ignore_index=True)
            physics_reports[family] = pd.concat(physics_parts, ignore_index=True)
            all_oof.append(oof)
            all_folds.append(folds)
            all_temporal.append(timed)
            all_learning.append(learning)
        print(f"[{time.monotonic() - started:.1f}s] Permutations", flush=True)
        n_permutations = 2 if quick else final["permutations"]
        arguments = [
            (train, families[selected], diagnostic_folds, b) for b in range(n_permutations)
        ]
        if quick or permutation_workers == 1:
            null = [_permutation_task(args) for args in arguments]
        else:
            with ProcessPoolExecutor(max_workers=permutation_workers) as executor:
                null = []
                for count, result in enumerate(executor.map(_permutation_task, arguments), 1):
                    null.append(result)
                    if count % 10 == 0:
                        print(
                            f"[{time.monotonic() - started:.1f}s] "
                            f"Permutations {count}/{n_permutations}",
                            flush=True,
                        )
        null = pd.DataFrame([r for batch in null for r in batch])
        null.to_csv(output / "permutation_null.csv", index=False)
        folds = pd.concat(all_folds, ignore_index=True)
        observed = (
            folds.loc[folds.family.eq(selected) & folds.repeat.eq(0)].groupby("target").r2.mean()
        )
        for target in TARGETS:
            values = null.loc[null.target.eq(target)].r2.to_numpy()
            permutation_table.append(
                dict(
                    target=target,
                    observed_r2=observed[target],
                    shuffled_r2_mean=values.mean(),
                    shuffled_r2_sd=values.std(ddof=1),
                    p_value=(1 + (values >= observed[target]).sum()) / (n_permutations + 1),
                    n_permutations=n_permutations,
                )
            )
        table = pd.DataFrame(tables)
        floors = load_yaml(ROOT / "conf/thresholds.yaml")["noise_floor_mape_pct_upper_bound"]
        table["noise_floor_alarm"] = table.cv_mape_pct_mean < table.target.map(floors)
        coverage = pd.concat(coverages, ignore_index=True)
        oof = pd.concat(all_oof, ignore_index=True)
        learning = pd.concat(all_learning, ignore_index=True)
        for frame, name in (
            (table, "cv_3x3"),
            (coverage, "coverage"),
            (folds, "folds"),
            (oof, "oof"),
            (learning, "learning"),
            (pd.concat(all_temporal, ignore_index=True), "temporal"),
            (pd.DataFrame(permutation_table), "permutation"),
        ):
            frame.to_csv(output / f"{name}.csv", index=False)
        _table_plot(table, output / "cv_3x3.png")
        metrics = _rounded(
            dict(
                protocol=dict(
                    architecture="a", cv_splits=n_splits, cv_repeats=n_repeats, quick=quick
                ),
                cv=tables,
                coverage=coverage.to_dict("records"),
                permutation=permutation_table,
            )
        )
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        example = train[list(INPUT_COLUMNS)].head(5)
        model_uris = {}
        for family, model in fitted_models.items():
            for j, target in enumerate(TARGETS):
                directory = output / "artifacts" / family / target
                _artifacts(
                    oof.loc[oof.family.eq(family) & oof.target.eq(target)],
                    learning.loc[learning.family.eq(family) & learning.target.eq(target)],
                    explanations[family].loc[explanations[family].target.eq(target)],
                    physics_reports[family].loc[physics_reports[family].target.eq(target)],
                    directory,
                )
                for name in ("cv_3x3.csv", "cv_3x3.png"):
                    shutil.copyfile(output / name, directory / name)
                row = table.loc[table.family.eq(family) & table.target.eq(target)].iloc[0]
                numeric = {
                    k: v for k, v in row.items() if isinstance(v, (float, int, np.number, np.bool_))
                }
                _log_child(
                    family,
                    target,
                    ProjectedTarget(model, j),
                    design_size(model.models_[j]),
                    numeric,
                    directory,
                    example,
                    dataset,
                    {
                        **params,
                        **families[family][j]["params"],
                        "candidate": families[family][j]["candidate"],
                        "representation": families[family][j]["representation"],
                        "physical_metric_a": families[family][j]["physical_metric_a"],
                        "architecture": "a",
                    },
                )
        directory = output / "artifacts" / "multioutput"
        _artifacts(
            oof.loc[oof.family.eq(selected)],
            learning.loc[learning.family.eq(selected)],
            explanations[selected],
            physics_reports[selected],
            directory,
        )
        for name in ("cv_3x3.csv", "cv_3x3.png"):
            shutil.copyfile(output / name, directory / name)
        numeric = (
            table.loc[table.family.eq(selected)].select_dtypes(include="number").mean().to_dict()
        )
        champion_uri = _log_child(
            "multioutput_a",
            "all",
            fitted_models[selected],
            sum(design_size(m) for m in fitted_models[selected].models_),
            numeric,
            directory,
            example,
            dataset,
            params,
        )
        # Every fitted family is transferred through MLflow, never through task-local /tmp.
        for family, model in fitted_models.items():
            if family == selected:
                model_uris[family] = champion_uri
                continue
            logged = mlflow.sklearn.log_model(
                model,
                name=f"frozen_{family}",
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                code_paths=[str(ROOT / "src/gcc_strength")],
                pip_requirements=[
                    "scikit-learn==1.7.2",
                    "pandas==2.3.3",
                    "numpy==2.3.4",
                    "scipy==1.16.3",
                    "statsmodels==0.14.5",
                    "matplotlib==3.10.6",
                    "lightgbm==4.6.0",
                    "mlflow-skinny==3.12.0",
                    "pyyaml==6.0.3",
                ],
            )
            model_uris[family] = logged.model_uri
        mlflow.log_dict(dict(families=model_uris, champion=champion_uri), "models.json")
        mlflow.log_artifacts(str(output), artifact_path="training")
        for name in (
            "metrics.json",
            "manifest.json",
            "cv_3x3.csv",
            "cv_3x3.png",
            "coverage.csv",
            "permutation.csv",
            "permutation_null.csv",
            "temporal.csv",
        ):
            mlflow.log_artifact(str(output / name))
    (output / "run.json").write_text(
        json.dumps(
            dict(
                parent_id=parent_id,
                tracking_uri=tracking_uri,
                elapsed_seconds=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Finished parent={parent_id} in {time.monotonic() - started:.1f}s", flush=True)
    return parent_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment")
    parser.add_argument("--gold", dest="source", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == "__main__":
    main()
