"""Pure parameter validation shared by the Databricks ML notebook wrappers."""

from __future__ import annotations

from dataclasses import dataclass

from gcc_strength.catalog import quote_ident

ML_TASKS = ("eda", "train", "evaluate", "register", "publish")


@dataclass(frozen=True)
class JobParameters:
    catalog: str
    schema: str
    experiment_path: str
    train_mode: str
    lockbox_mode: str
    train_run_id: str = ""
    evaluate_run_id: str = ""
    register_run_id: str = ""


def parse_notebook_parameters(task: str, values: dict[str, str]) -> JobParameters:
    """Reject invalid controls before reading data or opening an MLflow run."""
    if task not in ML_TASKS:
        raise ValueError(f"Unknown ML task: {task}")
    catalog, schema = values.get("catalog", "datathon"), values.get("schema", "gcc")
    quote_ident(catalog)
    quote_ident(schema)
    experiment = values.get("experiment_path", "")
    if not experiment.startswith("/") or "${" in experiment:
        raise ValueError("experiment_path must be a resolved absolute workspace path")
    mode = values.get("train_mode", "full")
    if mode not in {"full", "quick"}:
        raise ValueError("train_mode must be full or quick")
    if values.get("lockbox_mode", "replay") != "replay":
        raise ValueError("The job permits lockbox_mode=replay only")
    ids = {k: values.get(k, "") for k in ("train_run_id", "evaluate_run_id", "register_run_id")}
    required = {
        "eda": (),
        "train": (),
        "evaluate": ("train_run_id",),
        "register": ("train_run_id", "evaluate_run_id"),
        "publish": tuple(ids),
    }[task]
    if any(not ids[k] or not ids[k].isalnum() for k in required):
        raise ValueError(f"{task} requires upstream MLflow run IDs")
    return JobParameters(catalog, schema, experiment, mode, "replay", **ids)
