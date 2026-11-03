"""dq_results schema, gates and the bronze = silver + quarantine + duplicates reconciliation."""

import pytest

from gcc_strength import dq
from gcc_strength.config import load_medallion_conf

EXPECTED = load_medallion_conf()["expected_counts"]


def test_dq_frame_schema():
    rows = [dq.dq_row("silver", "r", "warn", 10, 2, "flag")]
    df = dq.to_frame(rows, "run1")
    assert list(df.columns) == dq.DQ_COLUMNS
    assert df["rows_failed"].dtype == "int64"
    assert str(df["ts"].dt.tz) == "UTC"


def test_dq_row_validates():
    with pytest.raises(ValueError):
        dq.dq_row("platinum", "r", "warn", 1, 0, "none")
    with pytest.raises(ValueError):
        dq.dq_row("silver", "r", "fatal", 1, 0, "none")


def test_reconcile():
    assert dq.reconcile(10, 7, 2, 1)[0]["rows_failed"] == 0
    with pytest.raises(dq.DataQualityError):
        dq.reconcile(10, 7, 2, 0)


def test_expected_counts():
    actual = {
        "bronze": 791,
        "silver": 773,
        "duplicates_removed": 0,
        "quarantine": {"empty_row": 8, "no_features": 5, "no_target": 5, "cast_error": 0},
    }
    h = EXPECTED["source_sha256"]
    assert dq.check_expected_counts(actual, EXPECTED, h)[0]["rows_failed"] == 0
    with pytest.raises(dq.DataQualityError):
        dq.check_expected_counts({**actual, "silver": 772}, EXPECTED, h)
    skipped = dq.check_expected_counts({**actual, "silver": 1}, EXPECTED, "other")
    assert skipped[0]["rule"] == "expected_counts_skipped_other_file"


def test_assert_no_errors():
    dq.assert_no_errors([dq.dq_row("gold", "x", "warn", 5, 5, "flag")])
    with pytest.raises(dq.DataQualityError):
        dq.assert_no_errors([dq.dq_row("gold", "x", "error", 5, 1, "fail")])


def test_real_run_reconciles(medallion):
    c = medallion.counts
    assert c["bronze"] == c["silver"] + sum(c["quarantine"].values()) + c["duplicates_removed"]
    d = medallion.dq_results
    assert list(d.columns) == dq.DQ_COLUMNS
    assert d["run_id"].eq("test").all()
    rec = d[d["rule"] == "reconciliation_bronze_eq_silver_quarantine_dups"]
    assert rec["rows_failed"].tolist() == [0]
    assert not ((d["severity"] == "error") & (d["rows_failed"] > 0)).any()
