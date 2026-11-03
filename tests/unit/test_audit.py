"""Tests for the Phase 0 data audit: the key facts the cleaning rules will rely on."""

import numpy as np
import pandas as pd
import pytest

from gcc_strength import audit


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return audit.load_raw()[1]


def test_schema_matches_expected(df):
    assert list(df.columns) == audit.EXPECTED_COLUMNS
    assert len(df) == 791


def test_product_type_values_are_canonical(df):
    assert set(df[audit.CATEGORICAL]) == set(audit.EXPECTED_TYPES)


def test_features_are_prescaled(df):
    scaling = audit.scaling_diagnosis(df).set_index("column")["inferred_scaling"]
    assert scaling["Physical_Metric_B"] == "z-score (ddof=0)"
    assert (scaling.drop("Physical_Metric_B") == "min-max [0,1]").all()


def test_phase_a_is_linear_combination_of_b_and_c(df):
    rel = audit.fit_linear_relation(df, "Mineral_Phase_A", ["Mineral_Phase_B", "Mineral_Phase_C"])
    assert rel.r2 > 0.99999
    assert rel.max_abs_residual < 1e-3
    assert abs(rel.intercept) < 1e-3


def test_no_other_exact_phase_dependency(df):
    others = [p for p in audit.PHASES if p != "Mineral_Phase_A"]
    x = df[others].dropna()
    sv = np.linalg.svd((x - x.mean()).to_numpy(), compute_uv=False)
    assert sv.min() > 0.1


def test_empty_rows_and_duplicates(df):
    assert list(audit.empty_rows(df)) == [128, 262, 264, 300, 368, 783, 789, 790]
    summary = audit.duplicate_summary(df)
    assert summary["exact_duplicates_among_non_empty_rows"] == 0


def test_target_triple_groups_are_consecutive(df):
    groups = audit.target_triple_groups(df)
    assert len(groups) == 21
    assert (groups["max_row_gap"] == 1).all()


def test_targets_are_monotone_in_time(df):
    cons = audit.target_consistency(df).set_index("group")
    assert cons.loc["all", "pct_R1>R7"] == 0
    assert cons.loc["all", "pct_R7>R28"] == 0


def test_row_619_is_degenerate(df):
    flags = audit.suspicious_rows(df)
    assert flags.loc[619, "phases_at_scaled_min_ge7"]


def test_physical_metric_a_starts_late(df):
    assert df["Physical_Metric_A"].first_valid_index() == 586


def test_report_builds():
    report = audit.build_report()
    assert "Relación casi exacta" in report


def test_strength_groups_consecutive_and_partial_labels():
    df = pd.DataFrame(
        {
            "Strength1": [1.0, 1.0, 1.0, 5.0, 1.0],
            "Strength7": [2.0, 2.0, 2.0, 6.0, 2.0],
            "Strength28": [3.0, np.nan, 3.0, 7.0, 3.0],
        }
    )
    gid = audit.strength_groups(df)
    # rows 0-2 share all observed targets (row 1 lacks R28); row 4 repeats but is not consecutive
    assert list(gid) == [0, 0, 0, 1, 2]


def test_report_survives_edge_cases():
    df = audit.load_raw()[1].dropna().reset_index(drop=True)
    df = df.loc[~df.duplicated(subset=audit.TARGETS)].reset_index(drop=True)
    assert audit.target_triple_groups(df).empty
    tests = audit.missingness_vs_type_tests(df)
    assert tests.empty and "p_value" in tests.columns


def test_robust_outliers_with_zero_mad():
    df = audit.load_raw()[1].copy()
    df["Mineral_Phase_G"] = 0.0
    df.loc[0, "Mineral_Phase_G"] = 1.0
    out = audit.robust_outliers(df).set_index("column")
    assert out.loc["Mineral_Phase_G", "robust_z>3.5"] == 1


def test_noise_conversion_factor():
    assert audit.NOISE_MEDIAN_TO_MAE == pytest.approx(0.8365, abs=1e-3)


def test_phase_a_is_integer_sum_of_b_and_c(df):
    ph = df.dropna(subset=["Mineral_Phase_A"])
    k = {c: ph[f"Mineral_Phase_{c}"] * n for c, n in [("A", 7328), ("B", 3947), ("C", 5993)]}
    for v in k.values():
        assert (v - v.round()).abs().max() < 1e-6
    diff = k["A"].round() - k["B"].round() - k["C"].round()
    assert set(diff.unique()) <= {-1.0, 0.0, 1.0}
