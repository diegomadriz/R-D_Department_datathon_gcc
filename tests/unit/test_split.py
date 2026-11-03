"""Split rules of ADR 0002 (8, 10-11). Only structure is checked: counts, proportions,
group separation, type presence and determinism. No target value of the test is inspected."""

import numpy as np
import pandas as pd
import pytest

from gcc_strength.config import LOCKBOX_FILE, load_medallion_conf
from gcc_strength.gold import build_gold, gold_lockbox_digest
from gcc_strength.split import (
    GROUP_COL,
    SPLIT_COL,
    SplitViabilityError,
    assign_split,
    assign_strata,
    lockbox_check,
    lockbox_digest,
    quantile_bins,
    read_lockbox,
    viability_checks,
    write_lockbox,
)

CONF = load_medallion_conf()["split"]


def frame(types, r28, r7=None):
    n = len(types)
    return pd.DataFrame(
        {
            "ProductType": types,
            "Strength28": r28,
            "Strength7": r7 if r7 is not None else [np.nan] * n,
            GROUP_COL: range(n),
        }
    )


def test_quantile_bins():
    b = quantile_bins(pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan]), 4)
    assert b.iloc[:8].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert pd.isna(b.iloc[8])


def test_strata_rules():
    types = ["Type 1"] * 40 + ["Type 2"] * 10 + ["Type 4"] * 3 + ["Type 5"] * 2
    r28 = list(range(40)) + list(range(10)) + [1, 2, 3, 4, 5]
    df = frame(types, [float(x) for x in r28])
    s = assign_strata(df, n_bins=CONF["n_bins"], rare_types=CONF["rare_types"], min_groups=5)
    assert set(s[df["ProductType"].isin(["Type 4", "Type 5"])]) == {"rare"}
    assert set(s[df["ProductType"] == "Type 1"]) == {f"Type 1|q{i}" for i in range(4)}
    assert set(s[df["ProductType"] == "Type 2"]) == {"Type 2|q0", "Type 2|q1"}
    assert df["ProductType"].tolist() == types  # the real type is untouched


def test_r7_fallback_and_type_only():
    r28 = [float(x) for x in range(20)] + [np.nan] * 7
    r7 = [np.nan] * 20 + [0.0, 1.0, 19.0, np.nan, np.nan, np.nan, np.nan]
    df = frame(["Type 1"] * 27, r28, r7)
    s = assign_strata(df, n_bins={"Type 1": 2}, rare_types=[], min_groups=4)
    assert s.iloc[20] == "Type 1|q0" and s.iloc[22] == "Type 1|q1"
    assert set(s.iloc[23:]) == {"Type 1|na"}


def test_small_type_only_stratum_joins_smallest_bin():
    r28 = [float(x) for x in range(20)] + [np.nan] * 2
    df = frame(["Type 1"] * 22, r28)
    s = assign_strata(df, n_bins={"Type 1": 2}, rare_types=[], min_groups=5)
    assert "Type 1|na" not in set(s)
    assert s.iloc[20] in {"Type 1|q0", "Type 1|q1"}


def test_small_bins_are_merged_with_adjacent():
    df = frame(["Type 1"] * 12, [float(x) for x in range(12)])
    s = assign_strata(df, n_bins={"Type 1": 4}, rare_types=[], min_groups=5)
    groups = df[GROUP_COL].groupby(s).nunique()
    assert (groups >= 5).all()
    assert all(label.startswith("Type 1|q") for label in groups.index)


def test_merged_rows_of_a_group_count_once():
    df = frame(["Type 1"] * 10, [float(x) for x in range(10)])
    df[GROUP_COL] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    s = assign_strata(df, n_bins={"Type 1": 2}, rare_types=[], min_groups=5)
    assert s.nunique() == 1


def test_viability_failures_raise():
    df = frame(["Type 1"] * 10, [float(x) for x in range(10)])
    strata = pd.Series("Type 1", index=df.index)
    kwargs = dict(
        test_fraction_min=0.18,
        test_fraction_max=0.22,
        required_types_both_sides=["Type 1"],
        min_groups_per_stratum=5,
    )
    ok = pd.Series(["test", "test"] + ["train"] * 8)
    assert viability_checks(df, ok, strata, **kwargs)
    with pytest.raises(SplitViabilityError, match="fraction"):
        viability_checks(df, pd.Series(["test"] * 5 + ["train"] * 5), strata, **kwargs)
    with pytest.raises(SplitViabilityError, match="types missing"):
        viability_checks(df, ok, strata, **{**kwargs, "required_types_both_sides": ["Type 2"]})
    shared = df.copy()
    shared.loc[2, GROUP_COL] = 0
    with pytest.raises(SplitViabilityError, match="both sides"):
        viability_checks(shared, ok, strata, **kwargs)
    with pytest.raises(SplitViabilityError, match="groups"):
        viability_checks(df, ok, strata, **{**kwargs, "min_groups_per_stratum": 11})


def test_assign_split_keeps_groups_together():
    groups = pd.Series(np.repeat(np.arange(50), 2))
    strata = pd.Series(np.where(groups % 2 == 0, "a", "b"))
    split = assign_split(strata, groups, n_splits=5, test_fold=0, random_state=42)
    assert (split.groupby(groups).nunique() == 1).all()
    assert (split == "test").mean() == pytest.approx(0.2)


def test_lockbox_digest_and_file(tmp_path):
    assert lockbox_digest([3, 1, 2]) == lockbox_digest([1, 2, 3])
    assert lockbox_digest([1, 2]) != lockbox_digest([1, 2, 3])
    path = tmp_path / "lockbox.sha256"
    assert read_lockbox(path) is None
    write_lockbox(path, "abc")
    assert read_lockbox(path) == "abc"
    assert lockbox_check("abc", "abc", 1)
    with pytest.raises(SplitViabilityError):
        lockbox_check("abc", "abd", 1)
    with pytest.raises(SplitViabilityError):
        lockbox_check("abc", None, 1)


# --------------------------------------------------------------------------------------
# Real data: structure only
# --------------------------------------------------------------------------------------
def test_real_split_structure(medallion):
    g = medallion.gold
    is_test = g[SPLIT_COL] == "test"
    assert set(g[SPLIT_COL]) == {"train", "test"}
    assert 0.18 <= is_test.mean() <= 0.22
    assert not set(g.loc[is_test, GROUP_COL]) & set(g.loc[~is_test, GROUP_COL])
    for t in CONF["required_types_both_sides"]:
        assert (g.loc[is_test, "ProductType"] == t).any()
        assert (g.loc[~is_test, "ProductType"] == t).any()


def test_real_split_is_deterministic_and_order_free(medallion):
    silver = medallion.silver
    again, _ = build_gold(silver.sample(frac=1, random_state=42), CONF)
    pd.testing.assert_series_equal(again[SPLIT_COL], medallion.gold[SPLIT_COL])
    assert gold_lockbox_digest(again) == medallion.lockbox_digest


def test_real_split_matches_committed_lockbox(medallion):
    assert medallion.lockbox_digest == read_lockbox(LOCKBOX_FILE)
