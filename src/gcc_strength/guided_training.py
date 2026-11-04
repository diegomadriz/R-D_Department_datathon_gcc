"""Fit frozen finalists using verified historical, train-only diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from threadpoolctl import threadpool_limits

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT
from gcc_strength.cv import load_model_config
from gcc_strength.eda import DEFAULT_GOLD, assert_train, load_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.models import design_size
from gcc_strength.provenance import train_fingerprint
from gcc_strength.train import ProjectedTarget, _fit, _log_child, source_revision
from gcc_strength.training_evidence import DEFAULT_EVIDENCE, DEFAULT_PIN, verify_evidence

DEFAULT_OUTPUT = ROOT / "outputs/fase3/guided"
PROVENANCE_KEYS = (
    "train_mode",
    "diagnostics_origin",
    "diagnostics_source_run_id",
    "evidence_manifest_sha256",
    "fitting_origin",
    "diagnostics_metrics_status",
)


def _tag_model_run(uri, tags):
    client = mlflow.tracking.MlflowClient()
    run_id = client.get_logged_model(uri.removeprefix("models:/")).source_run_id
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def run_guided_training(
    *,
    source=DEFAULT_GOLD,
    evidence=DEFAULT_EVIDENCE,
    tracking_uri="file:./mlruns",
    experiment=None,
    output=DEFAULT_OUTPUT,
    pin_path=DEFAULT_PIN,
):
    """Verify local evidence before fitting or logging; never recompute diagnostics."""
    started = time.monotonic()
    train = load_train(source).sort_values("_row_id", kind="stable").reset_index(drop=True)
    assert_train(train)
    if len(train) != 621:
        raise ValueError("The frozen training population must contain 621 rows")
    evidence = Path(evidence)
    manifest = verify_evidence(evidence, train, pin_path=pin_path)
    config = load_model_config()
    final = config["final"]
    if final["architecture"] != "a" or final["selected_family"] != "gaussian_process":
        raise ValueError("Guided fitting requires the frozen GPR architecture a selection")
    provenance = dict(
        train_mode="guided",
        diagnostics_origin="local",
        diagnostics_source_run_id=manifest["source_run_id"],
        evidence_manifest_sha256=hashlib.sha256(
            (evidence / "manifest.json").read_bytes()
        ).hexdigest(),
        fitting_origin="databricks" if tracking_uri.startswith("databricks") else "local",
        diagnostics_metrics_status="historical_imported",
    )
    params = dict(
        **provenance,
        train_size=len(train),
        test_size=train.attrs["lockbox_size"],
        seed=42,
        config_hash=hashlib.sha256((ROOT / "conf/models.yaml").read_bytes()).hexdigest(),
        git_commit=source_revision(),
        lockbox_sha256=(ROOT / "conf/lockbox.sha256").read_text().strip(),
        quick=False,
        fit_size=len(train),
        model_fit_size=len(train),
        model_train_sha256=train_fingerprint(train),
    )
    # Preserve producer metadata verbatim; current code/version metadata describes fitting only.
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Guided output must be empty to avoid mixing historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(evidence / "diagnostics", output, dirs_exist_ok=True)
    (output / "manifest.json").rename(output / "diagnostic_manifest.json")
    shutil.copyfile(evidence / "manifest.json", output / "evidence_manifest.json")
    fitting_manifest = dict(
        **params,
        diagnostics_manifest="diagnostic_manifest.json",
        fitting_source_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        fitting_versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "mlflow-skinny")
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(fitting_manifest, indent=2, sort_keys=True) + "\n"
    )
    table = pd.read_csv(output / "cv_3x3.csv")
    families = {f: [s[t] for t in TARGETS] for f, s in final["families"].items()}
    example = train[list(INPUT_COLUMNS)].head(5)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment or "gcc-phase3-guided")
    with mlflow.start_run(run_name="guided-final-fitting") as parent, threadpool_limits(limits=1):
        parent_id = parent.info.run_id
        mlflow.log_params(params)
        mlflow.set_tags(provenance)
        dataset = mlflow.data.from_pandas(train[[*INPUT_COLUMNS, *TARGETS]], name="gcc_gold_train")
        mlflow.log_input(dataset, context="training")
        fitted = {family: _fit(train, specs) for family, specs in families.items()}
        for family, model in fitted.items():
            for j, target in enumerate(TARGETS):
                row = table.loc[table.family.eq(family) & table.target.eq(target)].iloc[0]
                metrics = {
                    k: v for k, v in row.items() if isinstance(v, (float, int, np.number, np.bool_))
                }
                uri = _log_child(
                    family,
                    target,
                    ProjectedTarget(model, j),
                    design_size(model.models_[j]),
                    metrics,
                    output / "artifacts" / family / target,
                    example,
                    dataset,
                    {
                        **params,
                        **families[family][j]["params"],
                        **{
                            k: families[family][j][k]
                            for k in ("candidate", "representation", "physical_metric_a")
                        },
                        "architecture": "a",
                    },
                )
                _tag_model_run(uri, provenance)
        selected = final["selected_family"]
        numeric = (
            table.loc[table.family.eq(selected)].select_dtypes(include="number").mean().to_dict()
        )
        champion_uri = _log_child(
            "multioutput_a",
            "all",
            fitted[selected],
            sum(design_size(m) for m in fitted[selected].models_),
            numeric,
            output / "artifacts/multioutput",
            example,
            dataset,
            {**params, "architecture": "a"},
        )
        _tag_model_run(champion_uri, provenance)
        model_uris = {selected: champion_uri}
        for family, model in fitted.items():
            if family != selected:
                logged = mlflow.sklearn.log_model(
                    model,
                    name=f"frozen_{family}",
                    signature=infer_signature(example, model.predict(example)),
                    input_example=example,
                    code_paths=[str(ROOT / "src/gcc_strength")],
                    pip_requirements=[
                        f"{name}=={importlib.metadata.version(name)}"
                        for name in (
                            "scikit-learn",
                            "pandas",
                            "numpy",
                            "scipy",
                            "statsmodels",
                            "matplotlib",
                            "lightgbm",
                            "mlflow-skinny",
                            "pyyaml",
                        )
                    ],
                )
                model_uris[family] = logged.model_uri
        mlflow.log_dict(dict(families=model_uris, champion=champion_uri), "models.json")
        mlflow.log_artifacts(str(evidence), artifact_path="evidence")
        mlflow.log_artifacts(str(output), artifact_path="training")
        for path in sorted(output.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))
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
    return parent_id
