"""Train boundary, deterministic statistics and physics diagnostic contracts."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from gcc_strength import eda
from gcc_strength.audit import FEATURES, TARGETS


@pytest.fixture
def train():
    rng = np.random.default_rng(42)
    n = 160
    frame = pd.DataFrame({c: rng.uniform(0.1, 1, n) for c in FEATURES + TARGETS})
    frame["ProductType"] = "Type 1"
    frame["_row_id"] = np.arange(n)
    frame["strength_group_id"] = np.arange(n) // 2
    frame["split"] = "train"
    return eda.train_only(frame)


def test_loader_pushes_down_train_filter_and_never_exposes_test(tmp_path, monkeypatch):
    path = tmp_path / "gold.parquet"
    pd.DataFrame(
        {
            "split": ["train", "test", "train"],
            "_row_id": [1, 2, 3],
            "Strength1": [2.0, 999999.0, 4.0],
        }
    ).to_parquet(path)
    original = pd.read_parquet
    calls = []

    def reader(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", reader)
    result = eda.load_train(path)
    assert calls == [{"columns": ["split"]}, {"filters": [("split", "==", "train")]}]
    assert result._row_id.tolist() == [1, 3]
    assert result.Strength1.tolist() == [2.0, 4.0]
    assert "split" not in result
    eda.assert_train(result)


@pytest.mark.parametrize("labels", [["train", "test"], ["test"], ["validation"], [None], []])
def test_train_only_rejects_any_nontrain_input(labels):
    with pytest.raises(ValueError, match="only split == train"):
        eda.train_only(pd.DataFrame({"split": labels}))


def test_loader_rejects_unknown_labels(tmp_path):
    path = tmp_path / "gold.parquet"
    pd.DataFrame({"split": ["train", None]}).to_parquet(path)
    with pytest.raises(ValueError, match="partition"):
        eda.load_train(path)


@pytest.mark.parametrize(
    "function,args",
    [
        (eda.descriptive_statistics, ()),
        (eda.null_report, ()),
        (eda.outlier_report, ()),
        (eda.temporal_blocks, ()),
        (eda.correlation_report, ()),
        (eda.target_correlations, ()),
        (eda.rank_block_partial, ("Strength1", "Strength28", [], 8)),
        (eda.blaine_sensitivity, ()),
        (eda.compositional_features, ()),
        (eda.vif_report, (FEATURES,)),
        (eda.reduce_vif, (FEATURES,)),
        (eda.bootstrap_partial, ("Physical_Metric_B", ["Strength1"], 2)),
        (eda.physics_tests, (2,)),
        (eda.drift_report, (TARGETS,)),
        (eda.analyze, (2,)),
        (eda.write_report, ()),
        (eda.make_figures, ({}, None)),
    ],
)
def test_analysis_entrypoints_reject_mixed_gold_before_any_analysis(function, args):
    frame = pd.DataFrame({"split": ["train", "test"]})
    frame.attrs[eda.PROVENANCE] = True  # even an incorrectly marked mixed frame must fail
    with pytest.raises(ValueError, match="test rows are forbidden"):
        function(frame, *args)


def test_only_loader_reads_parquet():
    tree = ast.parse(inspect.getsource(eda))
    readers = []
    for definition in tree.body:
        if isinstance(definition, ast.FunctionDef):
            for node in ast.walk(definition):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"read_parquet", "read_csv", "read_table"}:
                        readers.append(definition.name)
    assert readers == ["load_train", "load_train"]


def test_statistics_missing_outliers_and_no_mutation(train):
    frame = train.copy()
    frame["Physical_Metric_B"] = 2.0
    frame.loc[0, "Physical_Metric_B"] = 9.0
    frame.loc[1, "Physical_Metric_A"] = np.nan
    before = frame.copy(deep=True)
    descriptive = eda.descriptive_statistics(frame)
    row = descriptive.query("scope == 'Todos' and column == 'Physical_Metric_A'").iloc[0]
    assert row.n == len(frame) - 1
    assert row["mean"] == pytest.approx(frame.Physical_Metric_A.mean())
    assert row["skew"] == pytest.approx(frame.Physical_Metric_A.skew())
    assert row.kurtosis_excess == pytest.approx(frame.Physical_Metric_A.kurt())
    outliers = eda.outlier_report(frame).set_index("column")
    assert outliers.loc["Physical_Metric_B", "mad_zero"]
    assert outliers.loc["Physical_Metric_B", "robust_z_outliers"] == 1
    assert outliers.loc["Physical_Metric_B", "iqr_outliers"] == 1
    nulls = eda.null_report(frame).set_index("column")
    assert nulls.loc["Physical_Metric_A", "n_null"] == 1
    pd.testing.assert_frame_equal(before, frame)


def test_empty_and_constant_outlier_columns(train):
    train["Physical_Metric_A"] = np.nan
    train["Physical_Metric_B"] = 1.0
    result = eda.outlier_report(train).set_index("column")
    assert result.loc["Physical_Metric_A", "n"] == 0
    assert result.loc["Physical_Metric_A", "robust_z_outliers"] == 0
    assert result.loc["Physical_Metric_B", "robust_z_outliers"] == 0


def test_ratios_zero_and_missing_are_flagged_without_epsilon(train):
    train.loc[0, "Mineral_Phase_F"] = 0
    train.loc[1, "Mineral_Phase_B"] = np.nan
    train.loc[2, "Mineral_Phase_A"] = 0
    result = eda.compositional_features(train)
    assert np.isnan(result.loc[0, "log_F_over_E"])
    assert result.loc[0, "flag_invalid_log_F_over_E"]
    assert result.loc[0, "F_over_E_plus_F"] == 0
    assert not result.loc[0, "flag_invalid_F_over_E_plus_F"]
    assert np.isnan(result.loc[0, "log_H_over_F"])
    assert np.isnan(result.loc[2, "log_A"])
    assert np.isnan(result.loc[2, "log_H_over_A"])
    assert result.loc[3, "log_A"] == pytest.approx(np.log(train.loc[3, "Mineral_Phase_A"]))
    assert np.isnan(result.loc[1, "log_C_over_B"])
    assert np.isnan(result.loc[2, "C_over_A"])
    assert result.loc[3, "log_C_over_B"] == pytest.approx(
        np.log(train.loc[3, "Mineral_Phase_C"] / train.loc[3, "Mineral_Phase_B"])
    )
    assert set(eda.RATIOS).isdisjoint(TARGETS)
    assert not any("Strength" in name for name in eda.RATIOS)
    pd.testing.assert_frame_equal(result[train.columns], train)


def test_temporal_blocks_are_order_invariant_and_keep_assays_intact(train):
    blocks = eda.temporal_blocks(train)
    assert set(blocks) == set(range(8))
    assert blocks.groupby(train.strength_group_id).nunique().max() == 1
    shuffled = train.sample(frac=1, random_state=42)
    pd.testing.assert_series_equal(eda.temporal_blocks(shuffled).sort_index(), blocks)
    train.loc[20, "strength_group_id"] = train.loc[19, "strength_group_id"]
    assert eda.temporal_blocks(train).loc[20] == eda.temporal_blocks(train).loc[19]


def test_partial_spearman_matches_explicit_dummy_projection(train):
    feature, target = "Physical_Metric_B", "Strength1"
    train.loc[:2, feature] = np.nan
    pair = train[[feature, target]].dropna()
    b = eda.temporal_blocks(train).loc[pair.index]
    design = np.column_stack([np.ones(len(pair)), pd.get_dummies(b, drop_first=True)])
    ranks = rankdata(pair.to_numpy(), axis=0)
    residual = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    expected = np.corrcoef(residual.T)[0, 1]
    table = eda.correlation_report(train, [feature])
    row = table.query("method == 'partial_spearman' and target == @target").iloc[0]
    assert row.n == len(pair)
    assert row.rho == pytest.approx(expected)


def test_bootstrap_is_reproducible_and_uses_paired_complete_cases(train):
    train["Strength1"] = train.Physical_Metric_B * 10
    train["Strength28"] = -train.Physical_Metric_B * 3
    train.loc[0, "Strength28"] = np.nan
    first = eda.bootstrap_partial(train, "Physical_Metric_B", ["Strength1", "Strength28"], 40)
    second = eda.bootstrap_partial(train, "Physical_Metric_B", ["Strength1", "Strength28"], 40)
    assert first == second
    assert first["n"] == len(train) - 1
    assert first["valid_bootstraps"] == 40
    assert first["effect"] == pytest.approx(2)
    assert first["ci_low"] == pytest.approx(2)


@pytest.mark.parametrize(
    "low,high,expected,verdict",
    [
        (0.1, 0.3, "+", "apoya"),
        (-0.3, -0.1, "+", "no apoya"),
        (-0.3, 0.1, "+", "no concluyente"),
        (-0.3, -0.1, "-", "apoya"),
        (-0.09, 0.09, "neutral", "apoya"),
        (-0.3, 0.3, "neutral", "no concluyente"),
        (0.11, 0.3, "neutral", "no apoya"),
        (np.nan, np.nan, "+", "no concluyente"),
    ],
)
def test_physics_verdict_requires_direction_or_equivalence(low, high, expected, verdict):
    assert eda.direction_verdict(low, high, expected) == verdict


def test_vif_detects_exact_redundancy_and_reduces_target_blind(train):
    train["Mineral_Phase_A"] = train.Mineral_Phase_B + train.Mineral_Phase_C
    columns = ["Mineral_Phase_A", "Mineral_Phase_B", "Mineral_Phase_C", "Mineral_Phase_D"]
    table = eda.vif_report(train, columns)
    assert (table.vif[:3] > 1e10).all()
    selected, history = eda.reduce_vif(train, columns)
    assert len(history) >= 1
    assert eda.vif_report(train, selected).vif.max() < 10
    altered = train.copy()
    altered[TARGETS] = 12345
    assert eda.reduce_vif(altered, columns)[0] == selected
    with pytest.raises(ValueError, match="quality features"):
        eda.vif_report(train, ["Strength1"])


def test_product_type_vif_uses_nominal_dummy_columns(train):
    train.loc[::2, "ProductType"] = "Type 2"
    result = eda.vif_report(train, ["Physical_Metric_B", "ProductType"])
    assert result.feature.tolist() == ["Physical_Metric_B", "ProductType_Type 2"]


def test_vif_excludes_rare_types_before_complete_case_design(train):
    baseline = eda.vif_report(train, ["Mineral_Phase_D", "Physical_Metric_B", "ProductType"])
    rare = train.iloc[:4].copy()
    rare["ProductType"] = ["Type 4", "Type 4", "Type 5", "Type 5"]
    rare["Mineral_Phase_D"] = 999
    combined = pd.concat([train, rare], ignore_index=True)
    pd.testing.assert_frame_equal(
        baseline, eda.vif_report(combined, ["Mineral_Phase_D", "Physical_Metric_B", "ProductType"])
    )


def test_rank_block_partial_matches_review_projection_with_controls(train):
    train.loc[0, "Mineral_Phase_D"] = np.nan
    controls = ["Mineral_Phase_D"]
    columns = ["Physical_Metric_B", "Strength1", *controls]
    part = train.dropna(subset=columns)
    blocks = pd.qcut(part._row_id.rank(), 24, labels=False)
    ranks = part[columns].rank().to_numpy()
    design = np.column_stack([pd.get_dummies(blocks, dtype=float), ranks[:, 2:]])
    residuals = ranks[:, :2] - design @ np.linalg.lstsq(design, ranks[:, :2], rcond=None)[0]
    n, rho = eda.rank_block_partial(train, *columns[:2], controls, 24)
    assert n == len(part)
    assert rho == pytest.approx(np.corrcoef(residuals.T)[0, 1])
    shuffled = train.sample(frac=1, random_state=42)
    assert eda.rank_block_partial(shuffled, *columns[:2], controls, 24)[1] == pytest.approx(rho)


def test_target_correlation_is_descriptive_and_reports_n(train):
    train["Strength7"] = train.Strength1 * 2
    train.loc[0, "Strength7"] = np.nan
    table = eda.target_correlations(train)
    row = table.query("scope == 'Todos' and method == 'pearson' and right == 'Strength7'").iloc[0]
    assert row.rho == pytest.approx(1)
    assert row.n == len(train) - 1


def test_ratios_are_invariant_to_multiplicative_scale_up_to_log_shift(train):
    original = eda.compositional_features(train)
    scaled = train.copy()
    scaled["Mineral_Phase_C"] *= 3
    scaled["Mineral_Phase_B"] *= 2
    transformed = eda.compositional_features(scaled)
    np.testing.assert_allclose(transformed.log_C_over_B, original.log_C_over_B + np.log(3 / 2))
    np.testing.assert_allclose(transformed.C_over_A, original.C_over_A * 3)
    # Constant scaling/shifts do not change rank association or VIF with an intercept.
    np.testing.assert_allclose(rankdata(original.log_C_over_B), rankdata(transformed.log_C_over_B))


def test_bootstrap_without_type1_is_explicitly_inconclusive(train):
    train["ProductType"] = "Type 2"
    result = eda.bootstrap_partial(train, "Physical_Metric_B", ["Strength1"], 10)
    assert result["n"] == 0
    assert result["valid_bootstraps"] == 0
    assert np.isnan(result["effect"])


def test_ratio_design_has_no_targets_or_partition_metadata():
    assert set(eda.RATIO_DESIGN_FEATURES) <= set(FEATURES + eda.RATIOS)
    assert set(eda.RATIO_DESIGN_FEATURES).isdisjoint(
        TARGETS + ["split", "_row_id", "strength_group_id"]
    )
