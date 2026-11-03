"""Gold table contract: columns, metadata vs features, groups."""

from gcc_strength.audit import FEATURES, TARGETS
from gcc_strength.gold import (
    FEATURE_COLUMNS,
    GOLD_COLUMNS,
    NON_FEATURE_COLUMNS,
    feature_frame,
)
from gcc_strength.split import GROUP_COL, SPLIT_COL


def test_gold_columns(medallion):
    g = medallion.gold
    assert list(g.columns) == GOLD_COLUMNS
    assert len(g) == len(medallion.silver)
    assert g["_row_id"].is_monotonic_increasing and g["_row_id"].is_unique
    assert g[GROUP_COL].dtype == "int64"


def test_no_derived_features_yet():
    assert set(FEATURE_COLUMNS) == {"ProductType", *FEATURES}


def test_metadata_and_targets_are_never_features(medallion):
    x = feature_frame(medallion.gold)
    forbidden = {*TARGETS, GROUP_COL, SPLIT_COL, "_row_id", "flag_r1_suspect"}
    assert forbidden <= set(NON_FEATURE_COLUMNS)
    assert not forbidden & set(x.columns)
    assert not any(t in c for c in x.columns for t in TARGETS)


def test_strength_groups_match_audit(medallion):
    sizes = medallion.gold[GROUP_COL].value_counts()
    assert (sizes > 1).sum() == 21
    assert sizes[sizes > 1].sum() == 44
