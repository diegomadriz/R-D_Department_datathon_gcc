"""Data-quality results (``dq_results``) and the mandatory count reconciliation.

Every layer returns a list of rule results built with :func:`dq_row`. The orchestrator stamps
them with ``run_id`` and ``ts`` (:func:`to_frame`) and appends them to ``dq_results``.

Severity semantics:

- ``info``: descriptive count, never fails.
- ``warn``: rows affected by a deterministic rule (quarantine, flag, set to null, drop).
- ``error``: a gate. Any ``error`` with ``rows_failed > 0`` fails the run
  (:func:`assert_no_errors`); gates that cannot be evaluated row by row raise directly.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime

import pandas as pd

DQ_COLUMNS = [
    "run_id",
    "ts",
    "layer",
    "rule",
    "severity",
    "rows_evaluated",
    "rows_failed",
    "action",
]
SEVERITIES = ("info", "warn", "error")
LAYERS = ("bronze", "silver", "gold", "dq")


class DataQualityError(RuntimeError):
    """A data-quality gate failed."""


def dq_row(
    layer: str,
    rule: str,
    severity: str,
    rows_evaluated: int,
    rows_failed: int,
    action: str,
) -> dict:
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return {
        "layer": layer,
        "rule": rule,
        "severity": severity,
        "rows_evaluated": int(rows_evaluated),
        "rows_failed": int(rows_failed),
        "action": action,
    }


def new_run_id() -> str:
    return uuid.uuid4().hex


def to_frame(rows: Iterable[Mapping], run_id: str, ts: datetime | None = None) -> pd.DataFrame:
    """Stamp rule results with the run id and timestamp, in the ``dq_results`` schema."""
    stamp = pd.Timestamp(ts or datetime.now(UTC))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    df = pd.DataFrame(list(rows), columns=DQ_COLUMNS[2:])
    df.insert(0, "ts", pd.Series([stamp] * len(df), dtype="datetime64[ns, UTC]"))
    df.insert(0, "run_id", str(run_id))
    df = df.astype({"rows_evaluated": "int64", "rows_failed": "int64"})
    return df[DQ_COLUMNS]


def rows_failed(rows: Iterable[Mapping], layer: str, rule: str) -> int:
    """``rows_failed`` of a single rule; raises if the rule is missing or repeated."""
    hits = [r for r in rows if r["layer"] == layer and r["rule"] == rule]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one dq row for {layer}.{rule}, found {len(hits)}")
    return int(hits[0]["rows_failed"])


def assert_no_errors(rows: Iterable[Mapping]) -> None:
    failed = [
        f"{r['layer']}.{r['rule']} ({r['rows_failed']} rows)"
        for r in rows
        if r["severity"] == "error" and r["rows_failed"] > 0
    ]
    if failed:
        raise DataQualityError("failed data-quality gates: " + ", ".join(failed))


def reconcile(
    n_bronze: int, n_silver: int, n_quarantine: int, duplicates_removed: int
) -> list[dict]:
    """``bronze = silver + quarantine + duplicates_removed``; raises if it does not balance."""
    accounted = n_silver + n_quarantine + duplicates_removed
    gap = abs(n_bronze - accounted)
    row = dq_row(
        "dq", "reconciliation_bronze_eq_silver_quarantine_dups", "error", n_bronze, gap, "fail"
    )
    if gap:
        raise DataQualityError(
            f"reconciliation failed: bronze={n_bronze} != silver={n_silver} + "
            f"quarantine={n_quarantine} + duplicates_removed={duplicates_removed}"
        )
    return [row]


def check_expected_counts(actual: Mapping, expected: Mapping, file_hash: str) -> list[dict]:
    """Compare the run counts with the counts documented by the Phase 0 audit.

    ``actual`` and ``expected`` hold ``bronze``, ``silver``, ``duplicates_removed`` and a
    ``quarantine`` mapping of reason -> rows. The check only applies to the audited file
    (``expected['source_sha256']``); for any other file it is recorded as skipped.
    """
    if file_hash != expected.get("source_sha256"):
        return [dq_row("dq", "expected_counts_skipped_other_file", "info", 0, 0, "none")]
    mismatches = []
    for key in ("bronze", "silver", "duplicates_removed"):
        if int(actual[key]) != int(expected[key]):
            mismatches.append(f"{key}: expected {expected[key]}, got {actual[key]}")
    exp_q = {k: int(v) for k, v in expected["quarantine"].items()}
    act_q = {k: int(v) for k, v in actual["quarantine"].items() if int(v)}
    if act_q != exp_q:
        mismatches.append(f"quarantine: expected {exp_q}, got {act_q}")
    row = dq_row(
        "dq",
        "expected_counts_phase0_audit",
        "error",
        int(actual["bronze"]),
        len(mismatches),
        "fail",
    )
    if mismatches:
        raise DataQualityError("counts differ from the Phase 0 audit: " + "; ".join(mismatches))
    return [row]


def run_reconciliation(
    n_bronze: int,
    n_silver: int,
    quarantine_by_reason: Mapping[str, int],
    duplicates_removed: int,
    file_hash: str,
    expected: Mapping | None,
) -> list[dict]:
    """Reconciliation plus the Phase 0 expected-count check (shared by local and Databricks)."""
    n_quarantine = int(sum(quarantine_by_reason.values()))
    rows = reconcile(n_bronze, n_silver, n_quarantine, duplicates_removed)
    if expected:
        actual = {
            "bronze": n_bronze,
            "silver": n_silver,
            "duplicates_removed": duplicates_removed,
            "quarantine": dict(quarantine_by_reason),
        }
        rows += check_expected_counts(actual, expected, file_hash)
    return rows
