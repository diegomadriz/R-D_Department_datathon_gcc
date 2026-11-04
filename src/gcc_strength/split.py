"""Persisted 80/20 group split (ADR 0002, rules 8, 10-12) and the lockbox hash.

- Rows are sorted by ``_row_id`` before anything else.
- Strata (only used to stratify; the real ``ProductType`` never changes):
  - Types listed as rare (4 and 5) share the single stratum ``rare``.
  - Other types: ``Strength28`` quantile bins within the type (quartiles for Type 1, median
    for Types 2 and 3). Rows without ``Strength28`` use the ``Strength7`` bins of their type;
    rows with neither get the type-only stratum ``<type>|na``.
  - Every stratum needs ``min_groups`` distinct groups. A smaller bin is merged with its
    adjacent bin of the same type (the smaller neighbour; ties go to the lower bin) until it
    complies. A small ``<type>|na`` stratum joins the type bin with the fewest groups.
- ``StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)``; the configured
  ``test_fold`` is the test set.
- Viability checks raise :class:`SplitViabilityError`.

The quantile cut points come from the per-type target values already exposed in the Phase 0
profile; they are partition metadata and are never persisted or used to model. Nothing in this
module reports target values.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from gcc_strength.audit import CATEGORICAL
from gcc_strength.dq import dq_row

GROUP_COL = "strength_group_id"
SPLIT_COL = "split"
PARTITION_LABELS = ("train", "test")
RARE = "rare"
NA_BIN = "na"


class SplitViabilityError(RuntimeError):
    """The split does not satisfy the viability checks of ADR 0002 (rule 11)."""


# --------------------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------------------
def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Bin index 0..n_bins-1 from the empirical quantiles of the non-null ``values``.

    A value equal to a cut point goes to the lower bin. Nulls stay null.
    """
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    known = values.dropna()
    if known.empty:
        return out
    cuts = np.quantile(known.to_numpy(), np.linspace(0, 1, n_bins + 1)[1:-1])
    out.loc[known.index] = np.searchsorted(cuts, known.to_numpy(), side="left")
    return out


def _bin_label(product_type: str, members: Sequence[int]) -> str:
    lo, hi = min(members), max(members)
    return f"{product_type}|q{lo}" if lo == hi else f"{product_type}|q{lo}-{hi}"


def _merge_small_bins(
    bins: pd.Series, groups: pd.Series, min_groups: int
) -> dict[int, tuple[int, ...]]:
    """Merge adjacent bins until each has >= min_groups groups. Returns bin -> merged block."""
    blocks: list[tuple[int, ...]] = [(b,) for b in sorted(bins.dropna().unique())]

    def n_groups(block: tuple[int, ...]) -> int:
        return groups[bins.isin(block)].nunique()

    while len(blocks) > 1:
        sizes = [n_groups(b) for b in blocks]
        small = [i for i, s in enumerate(sizes) if s < min_groups]
        if not small:
            break
        i = min(small, key=lambda k: (sizes[k], k))
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(blocks)]
        j = min(neighbours, key=lambda k: (sizes[k], k))
        lo, hi = sorted((i, j))
        blocks[lo : hi + 1] = [blocks[lo] + blocks[hi]]
    return {b: block for block in blocks for b in block}


def assign_strata(
    df: pd.DataFrame,
    *,
    n_bins: Mapping[str, int],
    rare_types: Iterable[str],
    min_groups: int,
    group_col: str = GROUP_COL,
) -> pd.Series:
    """Stratum label per row (see the module docstring for the rules)."""
    rare_types = set(rare_types)
    strata = pd.Series(df[CATEGORICAL].astype(str), index=df.index, name="stratum")
    strata[df[CATEGORICAL].isin(rare_types)] = RARE
    for ptype, k in n_bins.items():
        mask = df[CATEGORICAL] == ptype
        if not mask.any():
            continue
        sub = df.loc[mask]
        bins = quantile_bins(sub["Strength28"], int(k))
        fallback = quantile_bins(sub["Strength7"], int(k))
        bins = bins.fillna(fallback)
        groups = sub[group_col]
        blocks = _merge_small_bins(bins, groups, min_groups)
        labels = bins.map(
            lambda b, bl=blocks, t=ptype: _bin_label(t, bl[b]) if pd.notna(b) else f"{t}|{NA_BIN}"
        )
        na = labels == f"{ptype}|{NA_BIN}"
        if na.any() and groups[na].nunique() < min_groups and (~na).any():
            counts = groups[~na].groupby(labels[~na]).nunique()
            target = min(counts.index, key=lambda lab: (counts[lab], lab))
            labels[na] = target
        strata.loc[mask] = labels
    return strata


# --------------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------------
def assign_split(
    strata: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    test_fold: int,
    random_state: int,
) -> pd.Series:
    """``'train'``/``'test'`` per row from a fixed fold of ``StratifiedGroupKFold``."""
    if not 0 <= test_fold < n_splits:
        raise ValueError(f"test_fold must be in [0, {n_splits}), got {test_fold}")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(np.zeros(len(strata)), strata.to_numpy(), groups.to_numpy()))
    split = pd.Series("train", index=strata.index, name=SPLIT_COL, dtype="object")
    split.iloc[folds[test_fold][1]] = "test"
    return split


def viability_checks(
    df: pd.DataFrame,
    split: pd.Series,
    strata: pd.Series,
    *,
    test_fraction_min: float,
    test_fraction_max: float,
    required_types_both_sides: Iterable[str],
    min_groups_per_stratum: int,
    group_col: str = GROUP_COL,
) -> list[dict]:
    """Structural checks of ADR 0002 rule 11. Raises :class:`SplitViabilityError` on failure.

    Only counts, proportions, type presence and group separation are inspected.
    """
    n = len(df)
    groups = df[group_col]
    failures: list[str] = []
    rows: list[dict] = []

    per_stratum = groups.groupby(strata).nunique()
    small = per_stratum[per_stratum < min_groups_per_stratum]
    rows.append(
        dq_row(
            "gold",
            "split_min_groups_per_stratum",
            "error",
            n,
            strata.isin(small.index).sum(),
            "fail",
        )
    )
    if len(small):
        failures.append(f"strata with < {min_groups_per_stratum} groups: {small.to_dict()}")

    is_test = split == "test"
    frac = is_test.mean() if n else 0.0
    ok_frac = test_fraction_min <= frac <= test_fraction_max
    rows.append(
        dq_row("gold", "split_test_fraction_in_bounds", "error", n, 0 if ok_frac else n, "fail")
    )
    if not ok_frac:
        failures.append(
            f"test fraction {frac:.3f} outside [{test_fraction_min}, {test_fraction_max}]"
        )

    missing = [
        t
        for t in required_types_both_sides
        for side in (is_test, ~is_test)
        if not (df.loc[side, CATEGORICAL] == t).any()
    ]
    rows.append(dq_row("gold", "split_required_types_both_sides", "error", n, len(missing), "fail"))
    if missing:
        failures.append(f"types missing on one side: {sorted(set(missing))}")

    shared = set(groups[is_test]) & set(groups[~is_test])
    rows.append(
        dq_row("gold", "split_groups_disjoint", "error", n, groups.isin(shared).sum(), "fail")
    )
    if shared:
        failures.append(f"{len(shared)} strength groups on both sides")

    if failures:
        raise SplitViabilityError("; ".join(failures))
    return rows


# --------------------------------------------------------------------------------------
# Lockbox
# --------------------------------------------------------------------------------------
def lockbox_digest(test_row_ids: Iterable[int]) -> str:
    """SHA-256 of the sorted test ``_row_id`` values, one per line (``\\n``-terminated)."""
    text = "".join(f"{i}\n" for i in sorted(int(x) for x in test_row_ids))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def read_lockbox(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split()[0]
    return None


def write_lockbox(path: Path | str, digest: str) -> None:
    Path(path).write_text(f"{digest}\n", encoding="utf-8")


def lockbox_check(digest: str, stored: str | None, n_test: int) -> list[dict]:
    """The split must reproduce the stored lockbox hash exactly; raises otherwise."""
    if stored is None:
        raise SplitViabilityError("no stored lockbox hash: run the local pipeline first")
    if digest != stored:
        raise SplitViabilityError(
            "the test split differs from conf/lockbox.sha256: the lockbox must not change"
        )
    return [dq_row("gold", "lockbox_hash_matches", "error", n_test, 0, "fail")]
