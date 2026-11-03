"""Gold layer: modelling table with the persisted split (ADR 0002, rules 8-11).

Columns: ``_row_id``, ``ProductType``, the 12 raw features, the 3 targets (separate columns),
the Silver flags, ``strength_group_id`` and ``split``. Derived features are decided in
Phase 2 and are not part of this table yet.

``strength_group_id`` and ``split`` are partition metadata and ``flag_r1_suspect`` is computed
from the targets: none of them may be a model input (:data:`NON_FEATURE_COLUMNS`).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from gcc_strength.audit import CATEGORICAL, FEATURES, TARGETS, strength_groups
from gcc_strength.dq import dq_row
from gcc_strength.silver import FLAG_COLUMNS
from gcc_strength.split import (
    GROUP_COL,
    SPLIT_COL,
    assign_split,
    assign_strata,
    lockbox_digest,
    viability_checks,
)

GOLD_COLUMNS = [
    "_row_id",
    CATEGORICAL,
    *FEATURES,
    *TARGETS,
    *FLAG_COLUMNS,
    GROUP_COL,
    SPLIT_COL,
]
# Candidate model inputs (raw). Which of them enter each model is decided by CV.
FEATURE_COLUMNS = [CATEGORICAL, *FEATURES]
# Never model inputs: targets, partition metadata, row order and target-derived flags.
NON_FEATURE_COLUMNS = [*TARGETS, "_row_id", GROUP_COL, SPLIT_COL, "flag_r1_suspect"]
DERIVED_RAW_FEATURES = {"Mineral_Phase_A": "Mineral_Phase_B + Mineral_Phase_C"}


def build_gold(silver: pd.DataFrame, split_conf: Mapping) -> tuple[pd.DataFrame, list[dict]]:
    """Silver -> (gold, dq rows). Raises ``SplitViabilityError`` if the split is not viable."""
    df = silver.sort_values("_row_id", kind="stable").reset_index(drop=True)
    df[GROUP_COL] = strength_groups(df).astype("int64")

    strata = assign_strata(
        df,
        n_bins=split_conf["n_bins"],
        rare_types=split_conf["rare_types"],
        min_groups=int(split_conf["min_groups_per_stratum"]),
    )
    df[SPLIT_COL] = assign_split(
        strata,
        df[GROUP_COL],
        n_splits=int(split_conf["n_splits"]),
        test_fold=int(split_conf["test_fold"]),
        random_state=int(split_conf["random_state"]),
    )
    n = len(df)
    sizes = df[GROUP_COL].map(df[GROUP_COL].value_counts())
    dq = [
        dq_row("gold", "strength_groups_multi_row", "info", n, (sizes > 1).sum(), "group"),
        *viability_checks(
            df,
            df[SPLIT_COL],
            strata,
            test_fraction_min=float(split_conf["test_fraction_min"]),
            test_fraction_max=float(split_conf["test_fraction_max"]),
            required_types_both_sides=split_conf["required_types_both_sides"],
            min_groups_per_stratum=int(split_conf["min_groups_per_stratum"]),
        ),
    ]
    return df[GOLD_COLUMNS], dq


def gold_lockbox_digest(gold: pd.DataFrame) -> str:
    return lockbox_digest(gold.loc[gold[SPLIT_COL] == "test", "_row_id"])


def feature_frame(gold: pd.DataFrame) -> pd.DataFrame:
    """Raw candidate inputs only; partition metadata and targets are excluded by construction."""
    cols = [c for c in FEATURE_COLUMNS if c not in NON_FEATURE_COLUMNS]
    return gold[cols]
