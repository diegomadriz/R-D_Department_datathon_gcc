"""One-shot authorized lockbox evaluation; development must use synthetic Gold only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.spatial.distance import cdist
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from gcc_strength.audit import FEATURES, TARGETS
from gcc_strength.config import ROOT, load_yaml
from gcc_strength.eda import DEFAULT_GOLD, train_only
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.metrics import paired_mae_bootstrap, regression_metrics
from gcc_strength.models import build_model
from gcc_strength.split import GROUP_COL, lockbox_check, lockbox_digest, read_lockbox
from gcc_strength.train import _cvplus, _fit
from gcc_strength.uncertainty import coverage_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/fase4/lockbox"
METRICS = ("mae", "mape_pct", "rmse", "r2")


def require_authorization():
    """Fail before opening data or creating output, unless explicitly authorized."""
    if os.environ.get("GCC_LOCKBOX_AUTHORIZED") != "1":
        raise PermissionError("Lockbox evaluation requires explicit human authorization")


def check_git_clean():
    """Check tracked and untracked changes; ignored runtime outputs are allowed."""
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("A clean git tree is required")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    # Exclusive creation also protects concurrent invocations and completed results.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _feature_preprocessor():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                FEATURES,
            ),
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ProductType"]),
        ]
    )


def adversarial_validation(train, future, *, near_duplicate_distance, n_splits=5):
    """OOF AUC by assay group and cross-partition quality-pair counts only.

    Distances are Euclidean after train-fitted median/standardization and nominal
    one-hot encoding. Exact duplicates compare raw quality including missingness;
    near duplicates exclude exact pairs. Counts are pairs, not unique rows.
    """
    quality = pd.concat(
        [train[list(INPUT_COLUMNS)], future[list(INPUT_COLUMNS)]], ignore_index=True
    )
    labels = np.r_[np.zeros(len(train), dtype=int), np.ones(len(future), dtype=int)]
    groups = pd.concat([train[GROUP_COL], future[GROUP_COL]], ignore_index=True)
    classifier = Pipeline(
        [
            ("preprocess", _feature_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(cv.split(quality, labels, groups))
    if any(len(np.unique(labels[v])) != 2 or len(np.unique(labels[f])) != 2 for f, v in folds):
        raise ValueError("Adversarial folds require both partitions")
    probability = cross_val_predict(classifier, quality, labels, cv=folds, method="predict_proba")[
        :, 1
    ]
    # Merge matches missing values as equal and counts the full Cartesian multiplicity.
    exact_pairs = len(train[list(INPUT_COLUMNS)].merge(future[list(INPUT_COLUMNS)], how="inner"))
    preprocess = Pipeline([("quality", _feature_preprocessor())])
    fitted = preprocess.fit_transform(train[list(INPUT_COLUMNS)])
    next_features = preprocess.transform(future[list(INPUT_COLUMNS)])
    near_pairs = 0
    for start in range(0, len(next_features), 256):
        distance = cdist(fitted, next_features[start : start + 256])
        near_pairs += int((distance <= near_duplicate_distance).sum())
    return dict(
        auc=float(roc_auc_score(labels, probability)),
        exact_duplicate_pairs=exact_pairs,
        near_duplicate_pairs=max(0, near_pairs - exact_pairs),
        near_duplicate_distance=near_duplicate_distance,
    )


def bootstrap_metrics(y, predicted, types, groups, *, n_resamples):
    """Row-weighted regression metrics with percentile CIs from whole-group draws."""
    y, predicted, types, groups = map(np.asarray, (y, predicted, types, groups))
    point = regression_metrics(y, predicted, types)
    if n_resamples < 1 or pd.isna(groups).any():
        raise ValueError("Positive resamples and complete groups are required")
    error = predicted - y
    nonzero = y != 0
    percentage = np.zeros(len(y))
    percentage[nonzero] = 100 * abs(error[nonzero] / y[nonzero])
    clusters = (
        pd.DataFrame(
            dict(
                group=groups,
                n=1.0,
                ae=abs(error),
                se=error**2,
                ape=percentage,
                n_mape=nonzero.astype(float),
                sy=y,
                sy2=y**2,
            )
        )
        .groupby("group", sort=True)
        .sum()
        .to_numpy()
    )
    draws = {metric: [] for metric in METRICS}
    rng = np.random.default_rng(42)
    for start in range(0, n_resamples, 256):
        indices = rng.integers(0, len(clusters), (min(256, n_resamples - start), len(clusters)))
        n, ae, se, ape, n_mape, sy, sy2 = clusters[indices].sum(axis=1).T
        sst = sy2 - sy * sy / n
        draws["mae"].extend(ae / n)
        draws["rmse"].extend(np.sqrt(se / n))
        draws["mape_pct"].extend(
            np.divide(ape, n_mape, out=np.full_like(ape, np.nan), where=n_mape > 0)
        )
        draws["r2"].extend(
            1 - np.divide(se, sst, out=np.full_like(se, np.nan), where=(n >= 2) & (sst > 1e-10))
        )
    return {
        **point,
        "ci95": {
            metric: dict(
                zip(("lower", "upper"), np.nanquantile(draws[metric], [0.025, 0.975]), strict=True)
            )
            for metric in METRICS
        },
        "n_groups": len(clusters),
        "n_resamples": n_resamples,
    }


def _table_artifacts(rows, output):
    table = pd.DataFrame(
        [
            {
                **{k: row[k] for k in ("family", "target", *METRICS)},
                **{
                    f"{metric}_ci95_{bound}": row["ci95"][metric][bound]
                    for metric in METRICS
                    for bound in ("lower", "upper")
                },
            }
            for row in rows
        ]
    )
    table.to_csv(output / "test_3x3.csv", index=False)
    display = (
        table.assign(
            scores=table.apply(
                lambda row: "\n".join(
                    f"{m}: {row[m]:.4f} [{row[f'{m}_ci95_lower']:.4f}, "
                    f"{row[f'{m}_ci95_upper']:.4f}]"
                    for m in METRICS
                ),
                axis=1,
            )
        )
        .pivot(index="family", columns="target", values="scores")
        .reindex(columns=TARGETS)
    )
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")
    ax.set_title("Test metrics [95% group bootstrap CI]; MAPE in %")
    grid = ax.table(
        cellText=display.values, rowLabels=display.index, colLabels=display.columns, loc="center"
    )
    grid.auto_set_font_size(False)
    grid.set_fontsize(8)
    grid.scale(1, 4.5)
    fig.savefig(output / "test_3x3.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _log_metrics(value, prefix="test"):
    if isinstance(value, dict):
        for key, child in value.items():
            _log_metrics(child, f"{prefix}_{str(key).replace(' ', '_')}")
    elif isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                ids = [
                    str(row[k]).replace(" ", "_") for k in ("family", "target", "scope") if k in row
                ]
                _log_metrics(
                    {k: v for k, v in row.items() if k not in {"family", "target", "scope"}},
                    "_".join([prefix, *ids]),
                )
    elif isinstance(value, (int, float, np.number)) and np.isfinite(value):
        mlflow.log_metric(prefix, float(value))


def evaluate_lockbox(
    *,
    tracking_uri="file:./mlruns",
    parent_run_id=None,
    gold=DEFAULT_GOLD,
    output=DEFAULT_OUTPUT,
    models_path=ROOT / "conf/models.yaml",
    thresholds_path=ROOT / "conf/thresholds.yaml",
    lockbox_file=ROOT / "conf/lockbox.sha256",
    cv_metrics_path=ROOT / "outputs/fase3/final/metrics.json",
    git_check=None,
):
    """Evaluate once; injectable paths and git checker support synthetic tests only.

    No override/retry path exists. Any error after reserving attempt.json consumes
    the attempt, even when no result or MLflow run can be written.
    """
    require_authorization()
    output = Path(output)
    if (output / "result.json").exists() or (output / "attempt.json").exists():
        raise FileExistsError("Lockbox result or attempt already exists")
    git_commit = (git_check or check_git_clean)()
    config_hash = hashlib.sha256(Path(models_path).read_bytes()).hexdigest()
    config = load_yaml(models_path)
    limits = load_yaml(thresholds_path)["lockbox"]
    stored = read_lockbox(lockbox_file)
    if config["random_state"] != 42 or config["final"]["architecture"] != "a":
        raise ValueError("Only frozen architecture a with seed 42 is allowed")
    if (
        not 0 < limits["coverage_target"] < 1
        or limits["bootstrap_resamples"] < 1
        or limits["near_duplicate_distance"] < 0
    ):
        raise ValueError("Invalid lockbox thresholds")
    client = MlflowClient(tracking_uri=tracking_uri)
    page_token = None
    while True:
        experiments = client.search_experiments(view_type=ViewType.ALL, page_token=page_token)
        for experiment in experiments:
            if client.search_runs(
                [experiment.experiment_id],
                filter_string="tags.mlflow.runName = 'lockbox_evaluation'",
                run_view_type=ViewType.ALL,
                max_results=1,
            ):
                raise FileExistsError("A lockbox evaluation run already exists")
        page_token = experiments.token
        if not page_token:
            break
    experiment_id = None
    if parent_run_id:
        parent = client.get_run(parent_run_id)
        for key, expected in (("config_hash", config_hash), ("lockbox_sha256", stored)):
            if parent.data.params.get(key) != expected:
                raise ValueError(f"Parent run {key} mismatch")
        if parent.info.status != "FINISHED" or parent.data.params.get("quick") != "False":
            raise ValueError("Parent must be a finished full training run")
        experiment_id = parent.info.experiment_id
    cv_metrics = json.loads(Path(cv_metrics_path).read_text())
    if cv_metrics["protocol"]["quick"]:
        raise ValueError("CV reference must be a full training run")
    families = config["final"]["families"]
    cv_rows = {(r["family"], r["target"]): r for r in cv_metrics["cv"]}
    for family in families:
        for target in TARGETS:
            for metric in METRICS:
                if not np.isfinite(cv_rows[family, target][f"cv_{metric}_mean"]):
                    raise ValueError("Incomplete CV reference")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "attempt.json",
        dict(
            timestamp=datetime.now(UTC).isoformat(),
            git_commit=git_commit,
            config_hash=config_hash,
            thresholds_sha256=hashlib.sha256(Path(thresholds_path).read_bytes()).hexdigest(),
            parent_run_id=parent_run_id,
        ),
    )
    # Only partition metadata is opened to verify the split, never test quality/labels.
    metadata = pd.read_parquet(gold, columns=["_row_id", "split", GROUP_COL])
    if (
        metadata.empty
        or metadata.isna().any().any()
        or not metadata._row_id.is_unique
        or not metadata["split"].isin(["train", "test"]).all()
    ):
        raise ValueError("Invalid Gold partition metadata")
    held = metadata.loc[metadata["split"].eq("test")]
    fitting = metadata.loc[metadata["split"].eq("train")]
    lockbox_check(lockbox_digest(held._row_id), stored, len(held))
    if held.empty or fitting.empty or set(held[GROUP_COL]) & set(fitting[GROUP_COL]):
        raise ValueError("Nonempty disjoint train/lockbox groups are required")
    training = train_only(pd.read_parquet(gold, filters=[("split", "==", "train")]))
    training = training.sort_values("_row_id", kind="stable").reset_index(drop=True)
    future = (
        pd.read_parquet(
            gold, columns=[*INPUT_COLUMNS, GROUP_COL, "_row_id"], filters=[("split", "==", "test")]
        )
        .sort_values("_row_id", kind="stable")
        .reset_index(drop=True)
    )
    mlflow.set_tracking_uri(tracking_uri)
    if experiment_id is None:
        experiment_id = mlflow.set_experiment("gcc-phase3-final").experiment_id
    with (
        mlflow.start_run(run_name="lockbox_evaluation", experiment_id=experiment_id) as run,
        threadpool_limits(limits=1),
    ):
        mlflow.log_params(
            dict(
                git_commit=git_commit,
                config_hash=config_hash,
                lockbox_sha256=stored,
                train_size=len(training),
                test_size=len(future),
                thresholds_sha256=hashlib.sha256(Path(thresholds_path).read_bytes()).hexdigest(),
            )
        )
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)
        adversarial = adversarial_validation(
            training,
            future,
            near_duplicate_distance=limits["near_duplicate_distance"],
            n_splits=config["cv"]["n_splits"],
        )
        _log_metrics(adversarial, "test_adversarial")
        if (
            adversarial["auc"] > limits["adversarial_auc_max"]
            or adversarial["exact_duplicate_pairs"]
        ):
            result = dict(status="aborted", adversarial=adversarial)
        else:
            # Labels are opened only after the adversarial gate has passed.
            labels = pd.read_parquet(
                gold, columns=["_row_id", *TARGETS], filters=[("split", "==", "test")]
            )
            future = (
                future.merge(labels, on="_row_id", validate="one_to_one")
                .sort_values("_row_id")
                .reset_index(drop=True)
            )
            result = _evaluate_models(training, future, config, limits, cv_rows, output)
            result.update(status="ok", adversarial=adversarial, run_id=run.info.run_id)
            _log_metrics(result)
            for name in ("test_3x3.csv", "test_3x3.png"):
                mlflow.log_artifact(str(output / name))
        _write_json(output / "result.json", result)
        mlflow.log_artifact(str(output / "result.json"))
        mlflow.log_artifact(str(output / "attempt.json"))
    return result


def _evaluate_models(training, future, config, limits, cv_rows, output):
    X = future[list(INPUT_COLUMNS)]
    baseline = {}
    for target in TARGETS:
        known = training.loc[training[target].notna()]
        baseline[target] = (
            build_model("L1").fit(known[list(INPUT_COLUMNS)], known[target]).predict(X)
        )
    rows, coverage, monotonicity = [], [], []
    evaluated_models = {}
    for family, settings in config["final"]["families"].items():
        specs = [settings[target] for target in TARGETS]
        model = _fit(training, specs)
        raw = np.column_stack([component.predict(X) for component in model.models_])
        prediction = model.predict(X)
        monotonicity.append(
            dict(
                family=family,
                before_pct=float(100 * (np.diff(raw, axis=1) >= 0).all(axis=1).mean()),
                after_pct=float(100 * (np.diff(prediction, axis=1) >= 0).all(axis=1).mean()),
            )
        )
        intervals = _cvplus(
            training,
            future,
            specs,
            config["stage3"]["conformal_splits"],
            1 - limits["coverage_target"],
        )
        intervals["repeat"] = 0
        table = coverage_table(intervals)
        table["family"] = family
        coverage.extend(table.to_dict("records"))
        for j, target in enumerate(TARGETS):
            mask = future[target].notna().to_numpy()
            part = future.loc[mask]
            if part.empty:
                raise ValueError("Each lockbox target needs observed labels")
            observed, pred = part[target].to_numpy(), prediction[mask, j]
            scores = bootstrap_metrics(
                observed,
                pred,
                part.ProductType,
                part[GROUP_COL],
                n_resamples=limits["bootstrap_resamples"],
            )
            by_type = {}
            for number in (1, 2, 3):
                selected = part.ProductType.eq(f"Type {number}").to_numpy()
                by_type[f"Type {number}"] = (
                    regression_metrics(
                        observed[selected], pred[selected], part.ProductType[selected]
                    )
                    if selected.any()
                    else dict(n=0, mae=None, r2=None)
                )
            rows.append(
                dict(
                    family=family,
                    target=target,
                    **scores,
                    by_type=by_type,
                    delta_mae_vs_l1=paired_mae_bootstrap(
                        observed,
                        pred,
                        baseline[target][mask],
                        part[GROUP_COL],
                        n_bootstrap=limits["bootstrap_resamples"],
                    ),
                    test_minus_cv={
                        metric: scores[metric] - cv_rows[family, target][f"cv_{metric}_mean"]
                        for metric in METRICS
                    },
                )
            )
        # Preserve the exact fitted models for later registration without refitting.
        example = training[list(INPUT_COLUMNS)].head(5)
        logged = mlflow.sklearn.log_model(
            model,
            name=f"evaluated_{family}",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            code_paths=[str(ROOT / "src/gcc_strength")],
        )
        evaluated_models[family] = logged.model_uri
    _table_artifacts(rows, output)
    return dict(
        models=evaluated_models,
        metrics=rows,
        coverage=coverage,
        monotonicity=monotonicity,
        coverage_target=limits["coverage_target"],
        architecture="a",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--parent-run-id", required=True)
    args = parser.parse_args()
    try:
        result = evaluate_lockbox(**vars(args))
    except Exception:
        print(f"refused {DEFAULT_OUTPUT / 'result.json'}")
        raise SystemExit(1) from None
    print(f"{result['status']} {DEFAULT_OUTPUT / 'result.json'}")


if __name__ == "__main__":
    main()
