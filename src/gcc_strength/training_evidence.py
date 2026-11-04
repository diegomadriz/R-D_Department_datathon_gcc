"""Export and verify pinned, train-only diagnostics without refitting predictors.

Only explicit export may create a pin. Verification never repairs or writes evidence.
The pin attests bytes from export onward, not their historical creation time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import yaml

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT
from gcc_strength.eda import DEFAULT_GOLD, assert_train, load_train
from gcc_strength.metrics import regression_metrics
from gcc_strength.provenance import train_fingerprint
from gcc_strength.split import GROUP_COL
from gcc_strength.uncertainty import coverage_table

DEFAULT_EVIDENCE = ROOT / "data/training_evidence"
DEFAULT_PIN = ROOT / "conf/training_evidence_frozen.json"
ANALYTIC_COMPONENTS = (
    "features.py",
    "models.py",
    "multioutput.py",
    "uncertainty.py",
    "cv.py",
    "metrics.py",
    "explain.py",
    "stage3.py",
)
INTEGRATION_COMPONENTS = frozenset(
    {"catalog.py", "eda.py", "gold.py", "mlflow_audit.py", "pipeline_local.py", "train.py"}
)
FAMILIES = ("linear", "lightgbm", "gaussian_process")
ROOT_FILES = frozenset(
    {
        "coverage.csv",
        "cv_3x3.csv",
        "cv_3x3.png",
        "folds.csv",
        "learning.csv",
        "manifest.json",
        "metrics.json",
        "oof.csv",
        "permutation.csv",
        "permutation_null.csv",
        "temporal.csv",
        *(f"{family}_intervals.csv" for family in FAMILIES),
    }
)
CHILD_FILES = frozenset(
    {
        "cv_3x3.csv",
        "cv_3x3.png",
        "importance.csv",
        "importance.png",
        "learning_curve.csv",
        "learning_curve.png",
        "oof.csv",
        "oof_actual_vs_predicted.png",
        "physics.csv",
        "residuals.png",
        "residuals_by_type.png",
        "shap.csv",
        "shap_global.png",
        "shap_local.png",
    }
)
ARCHIVED_ROOT = ROOT_FILES - {
    "folds.csv",
    "learning.csv",
    "oof.csv",
    *(f"{f}_intervals.csv" for f in FAMILIES),
}
LIMITATION = (
    "The original run did not archive root folds or interval CSVs. Those files are local "
    "snapshots pinned at export, checked against archived CV/coverage aggregates and train "
    "identities. Their hashes are not historical MLflow attestations. The legacy producer "
    "train_sha256 includes ingestion metadata; train_fingerprint is independently recomputed."
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    _no_symlink(Path(path))
    return json.loads(Path(path).read_text())


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _no_symlink(path):
    # Only the entry itself: _inventory visits every entry below the package root, while
    # ancestors of the root belong to the host (macOS /var, workspace mounts, temp folders).
    if Path(path).is_symlink():
        raise ValueError("Symlinks are forbidden in evidence paths")


def _inventory(directory):
    directory = Path(directory)
    _no_symlink(directory)
    if not directory.is_dir():
        raise ValueError("Evidence directory is missing")
    result = {}
    for path in directory.rglob("*"):
        _no_symlink(path)
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = _sha(path)
        elif not path.is_dir():
            raise ValueError("Only regular evidence files and directories are allowed")
    return result


def _required():
    children = [f"{f}/{t}" for f in FAMILIES for t in TARGETS] + ["multioutput"]
    return ROOT_FILES | {f"artifacts/{child}/{name}" for child in children for name in CHILD_FILES}


def _local_path(uri):
    parsed = urlparse(str(uri))
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Only local file: tracking and artifact URIs are allowed")
    path = Path(unquote(parsed.path))
    _no_symlink(path)
    return path


def _local_client(tracking_uri):
    # Validate before importing/constructing a client; ambient remote defaults are ignored.
    _local_path(tracking_uri)
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _context(train=None):
    protocol = _json(ROOT / "conf/lockbox_protocol_frozen.json")
    config_hash = _sha(ROOT / "conf/models.yaml")
    lockbox = (ROOT / "conf/lockbox.sha256").read_text().strip()
    if config_hash != protocol["models_sha256"] or lockbox != protocol["lockbox_sha256"]:
        raise ValueError("Current config/lockbox metadata differs from frozen protocol")
    hashes = {n: _sha(ROOT / "src/gcc_strength" / n) for n in ANALYTIC_COMPONENTS}
    for name, value in protocol["source_sha256"].items():
        if hashes.get(name) != value:
            raise ValueError("Frozen predictor component differs: " + name)
    if train is not None:
        assert_train(train)
        if train["_row_id"].duplicated().any():
            raise ValueError("Train row identities must be unique")
        if train_fingerprint(train) != protocol["train_sha256"]:
            raise ValueError("Train fingerprint differs from frozen protocol")
    return protocol, config_hash, lockbox, hashes


def _producer_checks(producer, config_hash, lockbox, hashes):
    if producer.get("quick") is not False or producer.get("seed") != 42:
        raise ValueError("Evidence must come from final, non-quick seed-42 diagnostics")
    if producer.get("cv_splits") != 5 or producer.get("cv_repeats") != 5:
        raise ValueError("Evidence must retain the frozen 5x5 CV protocol")
    if producer.get("config_hash") != config_hash or producer.get("lockbox_sha256") != lockbox:
        raise ValueError("Producer config/lockbox metadata differs")
    historical = producer["source_hashes"]
    if any(historical.get(n) != value for n, value in hashes.items()):
        raise ValueError("Producer analytical components differ from current components")
    changes = {}
    for name, value in historical.items():
        if not re.fullmatch(r"[a-z_][a-z_0-9]*\.py", name):
            raise ValueError("Unsafe producer source path")
        path = ROOT / "src/gcc_strength" / name
        actual = _sha(path)
        if actual == value:
            continue
        if name == "split.py":
            normalized = path.read_bytes().replace(b'PARTITION_LABELS = ("train", "test")\n', b"")
            if hashlib.sha256(normalized).hexdigest() != value:
                raise ValueError("split.py differs beyond the known partition-label constant")
        elif name not in INTEGRATION_COMPONENTS:
            raise ValueError("Unrecognized historical source change: " + name)
        changes[name] = {"producer_sha256": value, "export_sha256": actual}
    return changes


def _train_rows(directory, train):
    """Check every OOF/interval label and identity against already filtered train."""
    lookup = train.set_index("_row_id")
    for name in sorted(_required()):
        if not (name.endswith("/oof.csv") or name == "oof.csv" or name.endswith("_intervals.csv")):
            continue
        frame = pd.read_csv(Path(directory) / name, float_precision="round_trip")
        needed = {"_row_id", GROUP_COL, "ProductType", "target", "observed"}
        if not needed.issubset(frame) or frame.empty:
            raise ValueError("Missing train identity columns: " + name)
        if not frame["_row_id"].isin(lookup.index).all() or not frame.target.isin(TARGETS).all():
            raise ValueError("Non-train row or unknown target in " + name)
        expected = lookup.loc[frame["_row_id"]]
        for col in (GROUP_COL, "ProductType"):
            if not np.array_equal(frame[col].to_numpy(), expected[col].to_numpy()):
                raise ValueError("Train metadata differs in " + name)
        observed = np.array(
            [
                lookup.at[row, target]
                for row, target in zip(frame["_row_id"], frame.target, strict=True)
            ]
        )
        if not np.allclose(frame.observed, observed, rtol=0, atol=1e-10, equal_nan=False):
            raise ValueError("Observed train target differs in " + name)
        if frame.duplicated(["_row_id", "target", "family", "repeat"]).any():
            raise ValueError("Duplicate OOF/interval identities in " + name)


def _same_frame(left, right, *, keys=None):
    if keys:
        left = left.sort_values(keys).reset_index(drop=True)
        right = right.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_like=True,
            rtol=1e-9,
            atol=1e-9,
        )
    except AssertionError as exc:
        raise ValueError("Diagnostic aggregate/reconstruction differs") from exc


def _snapshot_checks(directory):
    directory = Path(directory)

    def read(name):
        return pd.read_csv(directory / name, float_precision="round_trip")

    for family in FAMILIES:
        intervals = read(f"{family}_intervals.csv")
        if not intervals.family.eq(family).all() or (intervals.lower > intervals.upper).any():
            raise ValueError("Invalid interval family or bounds")
        covered = intervals.observed.between(intervals.lower, intervals.upper)
        if not np.array_equal(covered, intervals.covered):
            raise ValueError("Interval covered flag differs")
        archived = read("coverage.csv")
        archived = archived.loc[archived.family.eq(family)].drop(columns="family")
        _same_frame(coverage_table(intervals), archived, keys=["target", "scope"])
    oof = read("oof.csv")
    folds = read("folds.csv")
    keys = ["family", "target", "repeat", "fold"]
    for ids, frame in oof.groupby(keys):
        selected = folds
        for key, value in zip(keys, ids, strict=True):
            selected = selected.loc[selected[key].eq(value)]
        if len(selected) != 1:
            raise ValueError("Fold inventory differs from archived OOF")
        row = selected.iloc[0]
        scores = regression_metrics(frame.observed, frame.predicted, frame.ProductType)
        for name, value in scores.items():
            if name in row and not np.isclose(row[name], value, rtol=1e-9, atol=1e-9):
                raise ValueError("Fold validation metrics differ from archived OOF")
    if len(folds) != oof.groupby(keys).ngroups:
        raise ValueError("Unexpected folds")
    comparison = read("cv_3x3.csv")
    for (family, target), frame in folds.groupby(["family", "target"]):
        row = comparison.loc[comparison.family.eq(family) & comparison.target.eq(target)]
        if len(row) != 1:
            raise ValueError("CV comparison inventory differs")
        for metric in ("r2", "mae", "mape_pct", "rmse"):
            for stat, actual in (("mean", frame[metric].mean()), ("sd", frame[metric].std())):
                if not np.isclose(row.iloc[0][f"cv_{metric}_{stat}"], actual, atol=1e-9):
                    raise ValueError("CV aggregate differs from local folds snapshot")
        if not np.isclose(row.iloc[0].overfit_gap, frame.overfit_gap.mean(), atol=1e-9):
            raise ValueError("CV overfit gap differs from local folds snapshot")


def _origin_versions(client, experiment_id, run_ids):
    """Read only environment metadata attached to this training run's logged models."""
    records = []
    token = None
    while True:
        page = client.search_logged_models(
            experiment_ids=[experiment_id], max_results=50, page_token=token
        )
        for model in page:
            if model.source_run_id not in run_ids:
                continue
            path = _local_path(model.artifact_location)
            requirements = path / "requirements.txt"
            environment = path / "python_env.yaml"
            _no_symlink(requirements)
            _no_symlink(environment)
            packages = {}
            for line in requirements.read_text().splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line)
                if match:
                    packages[match[1]] = match[2]
            if not {"numpy", "scikit-learn", "pandas", "lightgbm"}.issubset(packages):
                raise ValueError("Missing source model dependency versions")
            records.append(
                {
                    "model_id": model.model_id,
                    "source_run_id": model.source_run_id,
                    "requirements_sha256": _sha(requirements),
                    "python_env_sha256": _sha(environment),
                    "python": str(yaml.safe_load(environment.read_text())["python"]),
                    "packages": packages,
                }
            )
        token = page.token
        if not token:
            break
    if {r["source_run_id"] for r in records} != set(run_ids):
        raise ValueError("Missing source model environment provenance")
    return sorted(records, key=lambda r: r["source_run_id"])


def _origin(source, run_id, client, producer):
    parent = client.get_run(run_id)
    if parent.info.status != "FINISHED" or parent.data.params.get("quick", "").lower() != "false":
        raise ValueError("Source run must be FINISHED and non-quick")
    for key in (
        "config_hash",
        "lockbox_sha256",
        "train_size",
        "fit_size",
        "seed",
        "cv_splits",
        "cv_repeats",
    ):
        if parent.data.params.get(key) != str(producer[key]):
            raise ValueError("Source run metadata differs: " + key)
    proofs = {}
    parent_path = _local_path(parent.info.artifact_uri)
    for name in sorted(ARCHIVED_ROOT):
        archived = parent_path / name
        _no_symlink(archived)
        if not archived.is_file() or _sha(archived) != _sha(source / name):
            raise ValueError("Source differs from archived parent artifact: " + name)
        proofs[name] = {"kind": "mlflow_bytes", "run_id": run_id, "artifact": name}
    children = []
    token = None
    while True:
        page = client.search_runs(
            [parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            max_results=100,
            page_token=token,
        )
        children.extend(page)
        token = page.token
        if not token:
            break
    expected = {(f, t) for f in FAMILIES for t in TARGETS} | {("multioutput_a", "all")}
    found = [(r.data.params.get("model_type"), r.data.params.get("target")) for r in children]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError("Source run must have exactly the frozen 9+1 children")
    for child in children:
        if child.info.status != "FINISHED":
            raise ValueError("Source child is not FINISHED")
        for key in ("config_hash", "lockbox_sha256", "seed", "train_size"):
            if child.data.params.get(key) != str(producer[key]):
                raise ValueError("Source child metadata differs")
        family, target = child.data.params["model_type"], child.data.params["target"]
        prefix = "artifacts/multioutput" if target == "all" else f"artifacts/{family}/{target}"
        root = _local_path(child.info.artifact_uri)
        for name in sorted(CHILD_FILES):
            archived = root / name
            _no_symlink(archived)
            if not archived.is_file() or _sha(archived) != _sha(source / prefix / name):
                raise ValueError(
                    "Source differs from archived child artifact: " + prefix + "/" + name
                )
            proofs[f"{prefix}/{name}"] = {
                "kind": "mlflow_bytes",
                "run_id": child.info.run_id,
                "artifact": name,
            }
    for name, child_name in (("oof.csv", "oof.csv"), ("learning.csv", "learning_curve.csv")):
        combined = pd.concat(
            [
                pd.read_csv(source / "artifacts" / f / t / child_name, float_precision="round_trip")
                for f in FAMILIES
                for t in TARGETS
            ],
            ignore_index=True,
        )
        keys = ["family", "target", "repeat", "fold"]
        keys += ["_row_id"] if name == "oof.csv" else ["fraction"]
        _same_frame(pd.read_csv(source / name, float_precision="round_trip"), combined, keys=keys)
        proofs[name] = {"kind": "reconstructed_from_archived_children", "artifact": child_name}
    for name in ("folds.csv", *(f"{f}_intervals.csv" for f in FAMILIES)):
        proofs[name] = {"kind": "local_snapshot_crosschecked_aggregates", "historical_hash": False}
    ids = sorted(r.info.run_id for r in children)
    return {
        "run_id": run_id,
        "status": "FINISHED",
        "quick": False,
        "git_commit": producer["git_commit"],
        "child_runs": ids,
        "source_verification": proofs,
        "versions": _origin_versions(client, parent.info.experiment_id, ids),
    }


def verify_evidence_files(directory, *, pin_path=DEFAULT_PIN):
    """Verify the complete pinned package and code compatibility without loading Gold."""
    directory = Path(directory)
    pin = _json(pin_path)
    manifest_path = directory / "manifest.json"
    _no_symlink(manifest_path)
    if _sha(manifest_path) != pin["manifest_sha256"]:
        raise ValueError("Evidence manifest differs from frozen pin")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("source_run_id") != pin["source_run_id"]:
        raise ValueError("Invalid evidence schema or source identity")
    files = manifest["files"]
    for name, digest in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name:
            raise ValueError("Unsafe evidence inventory path")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid evidence digest")
    required = {"diagnostics/" + name for name in _required()}
    if set(files) != required:
        raise ValueError("Evidence diagnostic inventory differs from the explicit allowlist")
    actual = _inventory(directory)
    expected = {**files, "manifest.json": pin["manifest_sha256"]}
    if actual != expected:
        raise ValueError("Evidence file inventory or hashes differ")
    protocol, config_hash, lockbox, hashes = _context()
    for key, value in (
        ("config_hash", config_hash),
        ("lockbox_sha256", lockbox),
        ("train_fingerprint", protocol["train_sha256"]),
        ("analytic_source_hashes", hashes),
    ):
        if manifest.get(key) != value:
            raise ValueError("Evidence compatibility differs: " + key)
    producer = _json(directory / "diagnostics/manifest.json")
    _producer_checks(producer, config_hash, lockbox, hashes)
    if producer["train_sha256"] != manifest["producer_train_sha256_legacy"]:
        raise ValueError("Producer legacy fingerprint differs")
    if manifest["producer_manifest_sha256"] != files["diagnostics/manifest.json"]:
        raise ValueError("Producer manifest identity differs")
    source = manifest["source"]
    if source["run_id"] != manifest["source_run_id"] or source["status"] != "FINISHED":
        raise ValueError("Invalid original source run")
    if source.get("quick") is not False or not source.get("versions"):
        raise ValueError("Missing final source environment provenance")
    if set(source["source_verification"]) != _required():
        raise ValueError("Missing per-file source provenance")
    return manifest


def verify_evidence(directory, train, *, pin_path=DEFAULT_PIN):
    """Strict read-only verification, including current train identity and labels."""
    _context(train)
    manifest = verify_evidence_files(directory, pin_path=pin_path)
    if manifest["train_fingerprint"] != train_fingerprint(train) or manifest["train_size"] != len(
        train
    ):
        raise ValueError("Evidence train identity differs")
    _train_rows(Path(directory) / "diagnostics", train)
    _snapshot_checks(Path(directory) / "diagnostics")
    return manifest


def export_evidence(
    source,
    destination=DEFAULT_EVIDENCE,
    *,
    tracking_uri="file:./mlruns",
    source_run_id=None,
    gold=DEFAULT_GOLD,
    pin_path=DEFAULT_PIN,
):
    """Explicitly export local source diagnostics and pin their immutable package bytes."""
    _local_path(tracking_uri)
    source, destination, pin_path = Path(source), Path(destination), Path(pin_path)
    for path in (source, destination, pin_path):
        _no_symlink(path)
    inventory = _inventory(source)
    if set(inventory) - {"run.json"} != _required():
        raise ValueError("Source inventory differs from the explicit diagnostic allowlist")
    if source_run_id is None:
        source_run_id = _json(source / "run.json")["parent_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", source_run_id):
        raise ValueError("Invalid source run identity")
    train = load_train(gold)
    protocol, config_hash, lockbox, hashes = _context(train)
    producer = _json(source / "manifest.json")
    changes = _producer_checks(producer, config_hash, lockbox, hashes)
    if producer["train_size"] != len(train) or producer["fit_size"] != len(train):
        raise ValueError("Producer training population differs")
    _train_rows(source, train)
    _snapshot_checks(source)
    origin = _origin(source, source_run_id, _local_client(tracking_uri), producer)
    manifest = {
        "schema_version": 1,
        "source_run_id": source_run_id,
        "source": origin,
        "config_hash": config_hash,
        "lockbox_sha256": lockbox,
        "train_fingerprint": protocol["train_sha256"],
        "train_size": len(train),
        "producer_manifest_sha256": inventory["manifest.json"],
        "producer_train_sha256_legacy": producer["train_sha256"],
        "analytic_source_hashes": hashes,
        "known_integration_changes": changes,
        "limitations": [LIMITATION],
        "lockbox_accessed": False,
        "files": {"diagnostics/" + n: inventory[n] for n in sorted(_required())},
    }
    content = _bytes(manifest)
    pin = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "source_run_id": source_run_id,
    }
    pin_content = _bytes(pin)
    if pin_path.exists() and pin_path.read_bytes() != pin_content:
        raise ValueError("Refusing to overwrite a different frozen evidence pin")
    if destination.exists():
        actual = _inventory(destination)
        if actual != {**manifest["files"], "manifest.json": pin["manifest_sha256"]}:
            raise ValueError("Refusing to overwrite a different evidence package")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination.parent) as temp:
            staging = Path(temp) / "package"
            for name in sorted(_required()):
                target = staging / "diagnostics" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            (staging / "manifest.json").write_bytes(content)
            if _inventory(staging) != {
                **manifest["files"],
                "manifest.json": pin["manifest_sha256"],
            }:
                raise ValueError("Source changed during export")
            staging.rename(destination)
    if not pin_path.exists():
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        with pin_path.open("xb") as handle:
            handle.write(pin_content)
    return verify_evidence(destination, train, pin_path=pin_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    export.add_argument("--tracking-uri", default="file:./mlruns")
    export.add_argument("--source-run-id")
    export.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    export.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    verify = sub.add_parser("verify")
    verify.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    verify.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    verify.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    args = parser.parse_args()
    if args.command == "export":
        manifest = export_evidence(
            args.source,
            args.output,
            tracking_uri=args.tracking_uri,
            source_run_id=args.source_run_id,
            gold=args.gold,
            pin_path=args.pin,
        )
    else:
        manifest = verify_evidence(args.evidence, load_train(args.gold), pin_path=args.pin)
    print(
        json.dumps(
            {
                "status": "verified",
                "source_run_id": manifest["source_run_id"],
                "files": len(manifest["files"]),
                "train_size": manifest["train_size"],
            }
        )
    )


if __name__ == "__main__":
    main()
