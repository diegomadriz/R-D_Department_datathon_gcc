"""Silver rules 1-7 of ADR 0002 on synthetic rows and on the real CSV."""

import numpy as np
import pandas as pd
import pytest
from _synthetic import make_bronze

from gcc_strength.audit import FEATURES, PHASES, TARGETS
from gcc_strength.dq import DataQualityError, rows_failed
from gcc_strength.silver import (
    FLAG_COLUMNS,
    QUARANTINE_COLUMNS,
    SILVER_COLUMNS,
    build_silver,
    cast_numeric,
    normalize_product_type,
)

FULL = {c: 0.5 for c in FEATURES} | {"Strength1": 2000, "Strength7": 5000, "Strength28": 7000}


def row(**kw):
    return {"ProductType": "Type 1", **FULL, **kw}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Type 1", "Type 1"),
        ("  type   3 ", "Type 3"),
        ("TYPE2", "Type 2"),
        ("tipo_04", "Type 4"),
        ("Type-5", "Type 5"),
        ("Type 6", None),
        ("Cement", None),
        ("", None),
    ],
)
def test_normalize_product_type(raw, expected):
    assert normalize_product_type(pd.Series([raw])).iloc[0] == expected


def test_cast_numeric_flags_bad_tokens_only():
    values, err = cast_numeric(pd.DataFrame({"a": ["1.5", " ", "", "abc", "inf", "NaN", "-0"]}))
    assert err.tolist() == [False, False, False, True, True, True, False]
    assert values["a"].iloc[0] == 1.5 and values["a"].iloc[1:6].isna().all()


def test_quarantine_reasons_and_priority():
    empty = {c: "" for c in FEATURES + TARGETS}
    rows = [
        row(),  # 0 kept
        {"ProductType": "Type 1", **empty},  # 1 empty_row
        {"ProductType": "Unknown", **empty},  # 2 empty_row wins over unknown type
        row(Strength1="abc"),  # 3 cast_error
        row(ProductType="Type 9"),  # 4 unknown_product_type
        {"ProductType": "Type 1", **{c: "" for c in FEATURES}, "Strength7": 5000},  # 5
        {**row(), **{t: "" for t in TARGETS}},  # 6 no_target
    ]
    silver, q, dq = build_silver(make_bronze(rows))
    assert silver["_row_id"].tolist() == [0]
    assert dict(zip(q["_row_id"], q["reason"], strict=True)) == {
        1: "empty_row",
        2: "empty_row",
        3: "cast_error",
        4: "unknown_product_type",
        5: "no_features",
        6: "no_target",
    }
    assert list(q.columns) == QUARANTINE_COLUMNS
    assert q.loc[q["_row_id"] == 3, "Strength1"].iloc[0] == "abc"  # raw value kept
    assert rows_failed(dq, "silver", "quarantine_empty_row") == 2


def test_duplicates_dropped_after_quarantine():
    empty = {c: "" for c in FEATURES + TARGETS}
    rows = [row(), row(), {"ProductType": "Type 1", **empty}, {"ProductType": "Type 1", **empty}]
    rows[1]["ProductType"] = " type 1"  # identical after normalisation
    silver, q, dq = build_silver(make_bronze(rows))
    assert silver["_row_id"].tolist() == [0]
    assert len(q) == 2
    assert rows_failed(dq, "silver", "exact_duplicate") == 1


def test_phase_rule_nulls_all_phases():
    phases7 = {p: 0 for p in PHASES[:-1]} | {PHASES[-1]: 1}
    phases6 = {p: 0 for p in PHASES[:6]}
    silver, _, _ = build_silver(make_bronze([row(**phases7), row(**phases6)]))
    anomalous = silver.set_index("_row_id")
    assert anomalous.loc[0, "flag_phases_anomalous"]
    assert anomalous.loc[0, PHASES].isna().all()
    assert not anomalous.loc[1, "flag_phases_anomalous"]
    assert (anomalous.loc[1, PHASES[:6]] == 0).all()
    # zero flags are evaluated after the phase rule
    assert not anomalous.loc[0, "flag_zero_Mineral_Phase_F"]
    assert anomalous.loc[1, "flag_zero_Mineral_Phase_F"]


def test_zero_flags_keep_value():
    silver, _, _ = build_silver(make_bronze([row(Mineral_Phase_H=0), row()]))
    assert silver["flag_zero_Mineral_Phase_H"].tolist() == [True, False]
    assert silver["Mineral_Phase_H"].iloc[0] == 0


def test_r1_rule():
    silver, _, _ = build_silver(make_bronze([row(Strength1=5000, Strength7=5000), row()]))
    assert silver["flag_r1_suspect"].tolist() == [True, False]
    assert np.isnan(silver["Strength1"].iloc[0])
    assert silver["Strength7"].iloc[0] == 5000 and silver["Strength28"].iloc[0] == 7000


def test_schema_mismatch_raises():
    b = make_bronze([row()]).drop(columns=["Strength28"])
    with pytest.raises(DataQualityError):
        build_silver(b)


def test_input_order_does_not_matter():
    rows = [row(Strength1=1000 + i) for i in range(5)]
    b = make_bronze(rows)
    a, _, _ = build_silver(b)
    c, _, _ = build_silver(b.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(a, c)


# --------------------------------------------------------------------------------------
# Real CSV (counts documented in the Phase 0 audit)
# --------------------------------------------------------------------------------------
def test_real_counts(medallion):
    assert len(medallion.silver) == 773
    assert medallion.quarantine["reason"].value_counts().to_dict() == {
        "empty_row": 8,
        "no_features": 5,
        "no_target": 5,
    }
    q = medallion.quarantine.groupby("reason")["_row_id"].apply(sorted).to_dict()
    assert q["empty_row"] == [128, 262, 264, 300, 368, 783, 789, 790]
    assert q["no_features"] == [169, 263, 329, 549, 601]
    assert q["no_target"] == [220, 314, 339, 519, 591]
    assert medallion.counts["duplicates_removed"] == 0


def test_real_flags(medallion):
    s = medallion.silver.set_index("_row_id")
    assert s.index[s["flag_phases_anomalous"]].tolist() == [619]
    assert s.loc[619, PHASES].isna().all()
    assert s.index[s["flag_r1_suspect"]].tolist() == [265]
    assert np.isnan(s.loc[265, "Strength1"])
    assert s.index[s["flag_zero_Mineral_Phase_F"]].tolist() == [312, 491]
    assert s.index[s["flag_zero_Mineral_Phase_H"]].tolist() == [241, 242, 243, 246, 491]


def test_silver_schema(medallion):
    s = medallion.silver
    assert list(s.columns) == SILVER_COLUMNS
    assert all(s[c].dtype == "float64" for c in FEATURES + TARGETS)
    assert all(s[c].dtype == bool for c in FLAG_COLUMNS)
    assert set(s["ProductType"]) <= {f"Type {i}" for i in range(1, 6)}


def test_no_statistical_columns(medallion):
    banned = ("iqr", "mad", "zscore", "outlier", "imputed")
    assert not any(b in c.lower() for c in medallion.silver.columns for b in banned)
