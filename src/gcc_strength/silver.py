"""Silver layer: validated types and deterministic cleaning rules (ADR 0002, rules 1-7).

Order of operations (each step works on the output of the previous one):

1. Schema check: the CSV must have exactly the expected columns.
2. ``ProductType`` normalisation (strip, collapse spaces, case and variants -> ``Type N``).
3. Numeric cast: the only null token is the empty string; any other value that does not parse
   to a finite float is a ``cast_error``.
4. Quarantine, one reason per row, by priority: ``empty_row`` > ``cast_error`` >
   ``unknown_product_type`` > ``no_features`` > ``no_target``.
5. Exact duplicates (same normalised type and same 15 numeric values) are dropped, keeping the
   lowest ``_row_id``. This runs after the quarantine, so empty rows are never counted as
   duplicates.
6. Deterministic rules on the remaining rows:
   - ``flag_phases_anomalous``: >= 7 of the 8 phases exactly at the scaled minimum (0). All
     8 phases are set to null.
   - ``flag_zero_Mineral_Phase_F`` / ``flag_zero_Mineral_Phase_H``: exact zeros, flagged and
     kept (possible "not measured"). Evaluated after the previous rule.
   - ``flag_r1_suspect``: ``Strength1 == Strength7``. ``Strength1`` is set to null.

No statistic of the dataset is used: no outlier flags, no imputation (ADR 0002, rules 5 and 7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcc_strength.audit import (
    CATEGORICAL,
    EXPECTED_COLUMNS,
    EXPECTED_TYPES,
    FEATURES,
    PHASES,
    TARGETS,
)
from gcc_strength.bronze import LINEAGE_COLUMNS
from gcc_strength.dq import DataQualityError, dq_row

NUMERIC = FEATURES + TARGETS
REASONS = ("empty_row", "cast_error", "unknown_product_type", "no_features", "no_target")
ZERO_FLAG_COLUMNS = ["Mineral_Phase_F", "Mineral_Phase_H"]
FLAG_COLUMNS = [
    "flag_phases_anomalous",
    *[f"flag_zero_{c}" for c in ZERO_FLAG_COLUMNS],
    "flag_r1_suspect",
]
MIN_ZERO_PHASES = len(PHASES) - 1
SILVER_COLUMNS = [
    "_row_id",
    CATEGORICAL,
    *FEATURES,
    *TARGETS,
    *FLAG_COLUMNS,
    "_source_file",
    "_file_hash",
    "_ingest_ts",
]
QUARANTINE_COLUMNS = [
    "_row_id",
    "reason",
    *EXPECTED_COLUMNS,
    "_source_file",
    "_file_hash",
    "_ingest_ts",
]

_PRODUCT_TYPE_RE = r"(?i)^(?:type|tipo)[\s_-]*0*([0-9]+)$"


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
def check_bronze_schema(bronze: pd.DataFrame) -> None:
    data_cols = [c for c in bronze.columns if c not in LINEAGE_COLUMNS]
    if data_cols != EXPECTED_COLUMNS:
        raise DataQualityError(
            f"unexpected CSV columns: expected {EXPECTED_COLUMNS}, got {data_cols}"
        )
    missing = [c for c in LINEAGE_COLUMNS if c not in bronze.columns]
    if missing:
        raise DataQualityError(f"bronze is missing lineage columns {missing}")


def normalize_product_type(values: pd.Series) -> pd.Series:
    """Map raw values to the ``Type N`` catalog; values outside ``Type 1..5`` become null.

    Accepts surrounding/repeated whitespace, any letter case, ``tipo`` and ``_``/``-``
    separators, and leading zeros: ``'  type   03'`` -> ``'Type 3'``.
    """
    clean = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    number = clean.str.extract(_PRODUCT_TYPE_RE, expand=False)
    out = ("Type " + number).astype("object")
    out = out.where(out.isin(EXPECTED_TYPES), None)
    return out.rename(CATEGORICAL)


def cast_numeric(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Cast string columns to float64. Returns (values, cast_error row mask).

    Empty (or whitespace-only) cells are null. Any other token that is not a finite number
    (``'abc'``, ``'NaN'``, ``'inf'``) is a cast error and its value is null.
    """
    values = pd.DataFrame(index=raw.index)
    errors = pd.Series(False, index=raw.index)
    for col in raw.columns:
        text = raw[col].astype("string").str.strip()
        blank = text.isna() | (text == "")
        parsed = pd.to_numeric(text.where(~blank), errors="coerce").astype("float64")
        bad = ~blank & ~np.isfinite(parsed)
        values[col] = parsed.where(~bad)
        errors |= bad
    return values, errors


def quarantine_reasons(
    raw: pd.DataFrame, product_type: pd.Series, values: pd.DataFrame, cast_error: pd.Series
) -> pd.Series:
    """One quarantine reason per row (or null if the row stays in Silver)."""
    blank = raw[NUMERIC].apply(lambda s: s.astype("string").str.strip().fillna("") == "")
    no_features = values[FEATURES].isna().all(axis=1)
    no_target = values[TARGETS].isna().all(axis=1)
    conditions = [
        blank.all(axis=1),
        cast_error,
        product_type.isna(),
        no_features,
        no_target,
    ]
    reason = pd.Series(np.select(conditions, list(REASONS), default=""), index=raw.index)
    return reason.where(reason != "", None)


def apply_phase_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 3: >= 7 phases exactly at 0 -> flag and set all 8 phases to null."""
    out = df.copy()
    flag = (out[PHASES] == 0).sum(axis=1) >= MIN_ZERO_PHASES
    out["flag_phases_anomalous"] = flag
    out.loc[flag, PHASES] = np.nan
    return out


def apply_zero_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 4: flag exact zeros of Mineral_Phase_F and H (values are kept)."""
    out = df.copy()
    for col in ZERO_FLAG_COLUMNS:
        out[f"flag_zero_{col}"] = out[col] == 0
    return out


def apply_r1_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 4: Strength1 == Strength7 is implausible -> flag and set Strength1 to null."""
    out = df.copy()
    flag = out["Strength1"].notna() & (out["Strength1"] == out["Strength7"])
    out["flag_r1_suspect"] = flag
    out.loc[flag, "Strength1"] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------------------
def build_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Bronze -> (silver, quarantine, dq rows)."""
    check_bronze_schema(bronze)
    b = bronze.sort_values("_row_id", kind="stable").reset_index(drop=True)
    n_bronze = len(b)
    dq: list[dict] = [dq_row("silver", "schema_expected_columns", "error", n_bronze, 0, "fail")]

    product_type = normalize_product_type(b[CATEGORICAL])
    values, cast_error = cast_numeric(b[NUMERIC])
    reason = quarantine_reasons(b, product_type, values, cast_error)

    changed = product_type.notna() & (product_type != b[CATEGORICAL])
    dq.append(
        dq_row("silver", "product_type_normalized", "info", n_bronze, changed.sum(), "normalize")
    )
    for r in REASONS:
        dq.append(
            dq_row("silver", f"quarantine_{r}", "warn", n_bronze, (reason == r).sum(), "quarantine")
        )

    in_quarantine = reason.notna()
    quarantine = b.loc[in_quarantine].assign(reason=reason[in_quarantine])
    quarantine = quarantine[QUARANTINE_COLUMNS].reset_index(drop=True)

    typed = pd.concat(
        [b[["_row_id"]], product_type, values, b[["_source_file", "_file_hash", "_ingest_ts"]]],
        axis=1,
    ).loc[~in_quarantine]
    duplicated = typed.duplicated(subset=[CATEGORICAL, *NUMERIC], keep="first")
    dq.append(dq_row("silver", "exact_duplicate", "warn", len(typed), duplicated.sum(), "drop"))
    typed = typed.loc[~duplicated]

    silver = apply_r1_rule(apply_zero_flags(apply_phase_rule(typed)))
    silver = silver[SILVER_COLUMNS].reset_index(drop=True)
    n = len(silver)
    dq += [
        dq_row(
            "silver",
            "phases_anomalous_ge7_at_zero",
            "warn",
            n,
            silver["flag_phases_anomalous"].sum(),
            "set_null_and_flag",
        ),
        *[
            dq_row("silver", f"zero_{c}", "info", n, silver[f"flag_zero_{c}"].sum(), "flag")
            for c in ZERO_FLAG_COLUMNS
        ],
        dq_row(
            "silver",
            "r1_equals_r7",
            "warn",
            n,
            silver["flag_r1_suspect"].sum(),
            "set_null_and_flag",
        ),
        *report_only_checks(silver),
    ]
    return silver, quarantine, dq


def report_only_checks(silver: pd.DataFrame) -> list[dict]:
    """Deterministic checks that only report (no rule acts on them today)."""
    n = len(silver)
    t = silver[TARGETS]
    non_positive = (t <= 0).any(axis=1)
    order = (
        (t["Strength1"] > t["Strength7"])
        | (t["Strength7"] > t["Strength28"])
        | (t["Strength1"] > t["Strength28"])
    )
    return [
        dq_row("silver", "target_non_positive", "warn", n, non_positive.sum(), "none"),
        dq_row("silver", "target_order_violation", "warn", n, order.sum(), "none"),
        dq_row(
            "silver",
            "no_features_after_rules",
            "warn",
            n,
            silver[FEATURES].isna().all(axis=1).sum(),
            "none",
        ),
        dq_row("silver", "no_target_after_rules", "warn", n, t.isna().all(axis=1).sum(), "none"),
    ]
