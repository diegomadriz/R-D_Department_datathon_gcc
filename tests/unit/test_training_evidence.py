"""Synthetic, train-only contracts for portable diagnostic evidence."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from gcc_strength import training_evidence as evidence
from gcc_strength.audit import TARGETS
from gcc_strength.eda import train_only
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.metrics import regression_metrics
from gcc_strength.provenance import train_fingerprint
from gcc_strength.split import GROUP_COL
from gcc_strength.uncertainty import coverage_table

RUN_ID = "a" * 32


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(evidence._bytes(value))


class Page(list):
    token = None


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path / "project"
    code = root / "src/gcc_strength"
    code.mkdir(parents=True)
    for name in evidence.ANALYTIC_COMPONENTS:
        (code / name).write_text("# frozen synthetic component\n")
    (root / "conf").mkdir()
    (root / "conf/models.yaml").write_text("seed: 42\n")
    (root / "conf/lockbox.sha256").write_text("b" * 64 + "\n")
    monkeypatch.setattr(evidence, "ROOT", root)
    frame = pd.DataFrame({name: np.arange(10, dtype=float) + 1 for name in INPUT_COLUMNS})
    frame["ProductType"] = "Type 1"
    frame["_row_id"] = np.arange(10)
    frame[GROUP_COL] = [f"group-{i}" for i in range(10)]
    frame["split"] = "train"
    for j, target in enumerate(TARGETS):
        frame[target] = np.arange(10, dtype=float) + 100 * (j + 1)
    train = train_only(frame)
    monkeypatch.setattr(evidence, "load_train", lambda _: train.copy())
    hashes = {name: evidence._sha(code / name) for name in evidence.ANALYTIC_COMPONENTS}
    config_hash = evidence._sha(root / "conf/models.yaml")
    protocol = {
        "models_sha256": config_hash,
        "lockbox_sha256": "b" * 64,
        "train_sha256": train_fingerprint(train),
        "source_sha256": hashes,
    }
    write_json(root / "conf/lockbox_protocol_frozen.json", protocol)
    source = tmp_path / "diagnostics"
    for name in evidence._required():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic archived image")
        else:
            path.write_text("synthetic\n")
    producer = {
        "quick": False,
        "seed": 42,
        "cv_splits": 5,
        "cv_repeats": 5,
        "config_hash": config_hash,
        "lockbox_sha256": "b" * 64,
        "source_hashes": hashes,
        "train_size": 10,
        "fit_size": 10,
        "git_commit": "c" * 40,
        "train_sha256": "d" * 64,
    }
    write_json(source / "manifest.json", producer)
    write_json(source / "run.json", {"parent_id": RUN_ID})
    write_json(source / "metrics.json", {"protocol": {"quick": False}})
    oofs, intervals, folds, learning, comparisons = [], [], [], [], []
    for family in evidence.FAMILIES:
        for target in TARGETS:
            rows = []
            for repeat in range(5):
                for index, row in train.iterrows():
                    rows.append(
                        {
                            "_row_id": row._row_id,
                            GROUP_COL: row[GROUP_COL],
                            "ProductType": row.ProductType,
                            "target": target,
                            "observed": row[target],
                            "predicted": row[target] + 0.5,
                            "family": family,
                            "repeat": repeat,
                            "fold": index % 5,
                        }
                    )
            oof = pd.DataFrame(rows)
            oofs.append(oof)
            interval = oof.drop(columns="predicted").assign(
                lower=oof.observed - 1,
                upper=oof.observed + 1,
                covered=True,
            )
            intervals.append(interval)
            for (repeat, fold), part in oof.groupby(["repeat", "fold"]):
                scores = regression_metrics(part.observed, part.predicted, part.ProductType)
                scores.update(
                    target=target,
                    family=family,
                    repeat=repeat,
                    fold=fold,
                    train_mae=0.3,
                    overfit_gap=0.1,
                    train_n=8,
                )
                folds.append(scores)
                if repeat == 0:
                    for fraction in (0.25, 0.5, 0.75, 1.0):
                        learning.append(dict(scores, fraction=fraction))
            subset = pd.DataFrame(folds)
            subset = subset.loc[subset.family.eq(family) & subset.target.eq(target)]
            comparison = {"family": family, "target": target, "overfit_gap": 0.1}
            for metric in ("r2", "mae", "mape_pct", "rmse"):
                comparison[f"cv_{metric}_mean"] = subset[metric].mean()
                comparison[f"cv_{metric}_sd"] = subset[metric].std()
            comparisons.append(comparison)
    oof = pd.concat(oofs, ignore_index=True)
    interval = pd.concat(intervals, ignore_index=True)
    fold_frame, learning_frame = pd.DataFrame(folds), pd.DataFrame(learning)
    comparison_frame = pd.DataFrame(comparisons)
    for frame_, name in (
        (oof, "oof.csv"),
        (fold_frame, "folds.csv"),
        (learning_frame, "learning.csv"),
        (comparison_frame, "cv_3x3.csv"),
    ):
        frame_.to_csv(source / name, index=False)
    coverages = []
    for family in evidence.FAMILIES:
        part = interval.loc[interval.family.eq(family)]
        part.to_csv(source / f"{family}_intervals.csv", index=False)
        coverages.append(coverage_table(part).assign(family=family))
    pd.concat(coverages).to_csv(source / "coverage.csv", index=False)
    children, models = [], []
    params = {
        key: str(producer[key])
        for key in (
            "config_hash",
            "lockbox_sha256",
            "train_size",
            "fit_size",
            "seed",
            "cv_splits",
            "cv_repeats",
        )
    }
    parent_artifacts = tmp_path / "parent"
    parent_artifacts.mkdir()
    for name in evidence.ARCHIVED_ROOT:
        shutil.copyfile(source / name, parent_artifacts / name)
    for index, (family, target) in enumerate(
        [*((f, t) for f in evidence.FAMILIES for t in TARGETS), ("multioutput_a", "all")]
    ):
        child_id = f"{index + 1:032x}"
        prefix = "multioutput" if target == "all" else f"{family}/{target}"
        child_dir = source / "artifacts" / prefix
        selection = (
            oof if target == "all" else oof.loc[oof.family.eq(family) & oof.target.eq(target)]
        )
        selection.to_csv(child_dir / "oof.csv", index=False)
        part = (
            learning_frame
            if target == "all"
            else learning_frame.loc[
                learning_frame.family.eq(family) & learning_frame.target.eq(target)
            ]
        )
        part.to_csv(child_dir / "learning_curve.csv", index=False)
        artifact_dir = tmp_path / "archived" / child_id
        shutil.copytree(child_dir, artifact_dir)
        children.append(
            NS(
                info=NS(run_id=child_id, status="FINISHED", artifact_uri=artifact_dir.as_uri()),
                data=NS(params=dict(params, model_type=family, target=target)),
            )
        )
        environment = tmp_path / "models" / child_id
        environment.mkdir(parents=True)
        (environment / "requirements.txt").write_text(
            "numpy==2.3.4\npandas==2.3.3\nscikit-learn==1.7.2\nlightgbm==4.6.0\n"
        )
        (environment / "python_env.yaml").write_text("python: 3.12.3\n")
        models.append(
            NS(
                source_run_id=child_id,
                model_id="m-" + child_id,
                artifact_location=environment.as_uri(),
            )
        )
    parent = NS(
        info=NS(status="FINISHED", experiment_id="123", artifact_uri=parent_artifacts.as_uri()),
        data=NS(params=dict(params, quick="False")),
    )
    client = NS(
        get_run=lambda _: parent,
        search_runs=lambda *a, **k: Page(children),
        search_logged_models=lambda **k: Page(models),
    )
    monkeypatch.setattr(evidence, "_local_client", lambda uri: client)
    return NS(
        root=root,
        source=source,
        train=train,
        parent=parent,
        children=children,
        output=tmp_path / "package",
        pin=tmp_path / "pin.json",
        client=client,
    )


def export(synthetic):
    return evidence.export_evidence(
        synthetic.source, synthetic.output, pin_path=synthetic.pin, gold="synthetic"
    )


def reseal(synthetic, manifest):
    write_json(synthetic.output / "manifest.json", manifest)
    pin = evidence._json(synthetic.pin)
    pin["manifest_sha256"] = evidence._sha(synthetic.output / "manifest.json")
    write_json(synthetic.pin, pin)


def test_valid_export_preserves_producer_and_readonly_verification(synthetic):
    manifest = export(synthetic)
    assert len(manifest["files"]) == 154
    assert manifest["producer_train_sha256_legacy"] != manifest["train_fingerprint"]
    assert (synthetic.output / "diagnostics/manifest.json").read_bytes() == (
        synthetic.source / "manifest.json"
    ).read_bytes()
    before = evidence._inventory(synthetic.output)
    pin_before = synthetic.pin.read_bytes()
    assert evidence.verify_evidence_files(synthetic.output, pin_path=synthetic.pin) == manifest
    assert (
        evidence.verify_evidence(synthetic.output, synthetic.train, pin_path=synthetic.pin)
        == manifest
    )
    assert export(synthetic) == manifest
    assert evidence._inventory(synthetic.output) == before
    assert synthetic.pin.read_bytes() == pin_before
    assert {v["kind"] for v in manifest["source"]["source_verification"].values()} == {
        "mlflow_bytes",
        "reconstructed_from_archived_children",
        "local_snapshot_crosschecked_aggregates",
    }
    assert str(synthetic.root) not in json.dumps(manifest)


@pytest.mark.parametrize("mode", ["tampered", "missing", "extra", "symlink", "directory_symlink"])
def test_reject_package_inventory_changes(synthetic, mode):
    export(synthetic)
    path = synthetic.output / "diagnostics/cv_3x3.png"
    if mode == "tampered":
        path.write_bytes(b"tampered")
    elif mode == "missing":
        path.unlink()
    elif mode == "extra":
        (synthetic.output / "unexpected.pkl").write_bytes(b"payload")
    elif mode == "symlink":
        path.unlink()
        path.symlink_to(synthetic.source / "cv_3x3.png")
    else:
        (synthetic.output / "extra").symlink_to(synthetic.source, target_is_directory=True)
    with pytest.raises(ValueError):
        evidence.verify_evidence(synthetic.output, synthetic.train, pin_path=synthetic.pin)


def test_symlinked_ancestor_of_package_is_host_layout_not_tampering(synthetic, tmp_path):
    export(synthetic)
    host_alias = tmp_path / "host-alias"
    host_alias.symlink_to(synthetic.output.parent, target_is_directory=True)
    package = host_alias / synthetic.output.name
    assert evidence.verify_evidence_files(package, pin_path=synthetic.pin)
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(synthetic.output, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        evidence.verify_evidence_files(root_alias, pin_path=synthetic.pin)


def test_reject_changed_manifest_and_refuse_overwrite(synthetic):
    export(synthetic)
    path = synthetic.output / "manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="pin"):
        evidence.verify_evidence_files(synthetic.output, pin_path=synthetic.pin)
    with pytest.raises(ValueError, match="overwrite"):
        export(synthetic)


@pytest.mark.parametrize("field", ["quick", "train", "config", "component", "traversal"])
def test_reject_incompatible_or_unsafe_resealed_package(synthetic, field):
    manifest = export(synthetic)
    if field == "quick":
        manifest["source"]["quick"] = True
    elif field == "train":
        manifest["train_fingerprint"] = "f" * 64
    elif field == "config":
        (synthetic.root / "conf/models.yaml").write_text("# changed\n")
    elif field == "component":
        (synthetic.root / "src/gcc_strength/features.py").write_text("# changed\n")
    else:
        manifest["files"]["../escaped.csv"] = "f" * 64
    reseal(synthetic, manifest)
    with pytest.raises(ValueError):
        evidence.verify_evidence(synthetic.output, synthetic.train, pin_path=synthetic.pin)


def test_reject_different_train_or_unmarked_frame(synthetic):
    export(synthetic)
    changed = synthetic.train.copy()
    changed.loc[0, "Strength1"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        evidence.verify_evidence(synthetic.output, changed, pin_path=synthetic.pin)
    changed = synthetic.train.copy()
    changed.attrs.clear()
    with pytest.raises(ValueError, match="load_train"):
        evidence.verify_evidence(synthetic.output, changed, pin_path=synthetic.pin)


@pytest.mark.parametrize("mode", ["quick", "unfinished", "missing", "altered", "unexpected"])
def test_reject_unverifiable_origin(synthetic, mode):
    if mode == "quick":
        synthetic.parent.data.params["quick"] = "True"
    elif mode == "unfinished":
        synthetic.parent.info.status = "RUNNING"
    elif mode == "missing":
        (Path(synthetic.parent.info.artifact_uri.removeprefix("file://")) / "metrics.json").unlink()
    elif mode == "altered":
        (synthetic.source / "artifacts/linear/Strength1/shap.csv").write_text("tampered\n")
    else:
        (synthetic.source / "models.pkl").write_text("forbidden")
    with pytest.raises(ValueError):
        export(synthetic)
    assert not synthetic.pin.exists()
    assert not synthetic.output.exists()


@pytest.mark.parametrize("mode", ["row", "observed", "group", "ProductType", "coverage", "folds"])
def test_reject_invalid_local_snapshots(synthetic, mode):
    name = "folds.csv" if mode == "folds" else "linear_intervals.csv"
    path = synthetic.source / name
    frame = pd.read_csv(path)
    if mode == "row":
        frame.loc[0, "_row_id"] = 9999
    elif mode == "observed":
        frame.loc[0, "observed"] += 100
    elif mode == "group":
        frame.loc[0, GROUP_COL] = "foreign"
    elif mode == "ProductType":
        frame.loc[0, "ProductType"] = "Type 2"
    elif mode == "coverage":
        frame.loc[0, "upper"] += 10
    else:
        frame.loc[0, "train_mae"] = 999
        # The first repeat is also archived in the full-fraction learning curve.
        frame.loc[0, "mae"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        export(synthetic)
    assert not synthetic.pin.exists()


@pytest.mark.parametrize(
    "uri", ["https://remote", "databricks", "file://remote/path", "sqlite:///x"]
)
def test_reject_nonlocal_tracking_before_client(synthetic, monkeypatch, uri):
    def forbidden(_):
        pytest.fail("Client must not be constructed")

    monkeypatch.setattr(evidence, "_local_client", forbidden)
    with pytest.raises(ValueError, match="local file"):
        evidence.export_evidence(
            synthetic.source, synthetic.output, tracking_uri=uri, pin_path=synthetic.pin
        )


def test_different_pin_is_never_repaired(synthetic):
    synthetic.pin.write_text('{"different":true}\n')
    with pytest.raises(ValueError, match="overwrite"):
        export(synthetic)
    assert synthetic.pin.read_text() == '{"different":true}\n'
    assert not synthetic.output.exists()
