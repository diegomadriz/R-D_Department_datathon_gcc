"""Metric edge cases and row-weighted paired cluster bootstrap."""

import numpy as np
import pandas as pd
import pytest

from gcc_strength.metrics import (
    compare_oof,
    paired_mae_bootstrap,
    regression_metrics,
    within_type_r2_bootstrap,
)


def test_within_type_bootstrap_preserves_group_repeats_and_detects_zero():
    frame = pd.DataFrame(
        {
            "observed": list(range(10)),
            "predicted": list(range(10)),
            "strength_group_id": list(range(10)),
            "ProductType": ["Type 1"] * 10,
        }
    )
    result = within_type_r2_bootstrap(frame)
    repeated = within_type_r2_bootstrap(pd.concat([frame] * 5, ignore_index=True))
    assert result[1] == repeated[1]
    assert result[1]["r2"] == result[1]["ci_low"] == result[1]["ci_high"] == 1
    assert np.isnan(result[2]["r2"])
    with pytest.raises(ValueError, match="positive"):
        within_type_r2_bootstrap(frame, n_bootstrap=0)


def test_scores_are_known_values_and_type1_is_separate():
    result = regression_metrics([1, 3, 7, 9], [2, 2, 7, 7], ["Type 1"] * 2 + ["Type 2"] * 2)
    assert result["mae"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt(1.5))
    assert result["r2"] == pytest.approx(1 - 6 / 40)
    assert result["r2_type1"] == 0
    assert result["mae_type1"] == 1
    assert result["mape_pct"] == pytest.approx(100 * (1 + 1 / 3 + 2 / 9) / 4)


def test_undefined_r2_zero_mape_and_absent_type1_are_explicit():
    result = regression_metrics([0, 0], [1, 1], ["Type 2", "Type 2"])
    assert result["n_mape"] == 0
    assert np.isnan(result["mape_pct"])
    assert np.isnan(result["r2"])
    assert np.isnan(result["r2_type1"])
    assert result["n_type1"] == 0
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([np.nan], [1], ["Type 1"])


def test_bootstrap_resamples_whole_groups_and_preserves_row_weighted_estimand():
    y = np.zeros(4)
    model = [0, 0, 0, 9]
    baseline = np.ones(4)
    groups = [1, 1, 1, 2]
    result = paired_mae_bootstrap(y, model, baseline, groups)
    assert result == paired_mae_bootstrap(y, model, baseline, groups)
    assert result["delta_mae"] == pytest.approx(1.25)
    assert result["ci_low"] == -1
    assert result["ci_high"] == 8
    assert not result["beats_l1"]
    assert result["n_groups"] == 2
    assert result["n_bootstrap"] == 10000
    # Duplicating repeated OOF losses must not shrink the cluster interval.
    repeated = paired_mae_bootstrap(
        np.tile(y, 5), np.tile(model, 5), np.tile(baseline, 5), np.tile(groups, 5)
    )
    assert repeated == result


def test_paired_bootstrap_superiority_requires_entire_ci_below_zero():
    result = paired_mae_bootstrap([1, 2, 3], [1, 2, 3], [2, 3, 4], [1, 2, 3])
    assert result["beats_l1"]
    assert result["delta_mae"] == result["ci_low"] == result["ci_high"] == -1


def test_compare_oof_aligns_rows_and_uses_mean_losses_not_mean_predictions():
    model = pd.DataFrame(
        {
            "target": ["Strength1"] * 4,
            "repeat": [0, 0, 1, 1],
            "_row_id": [1, 2, 1, 2],
            "observed": [10] * 4,
            "strength_group_id": [1, 2, 1, 2],
            "ProductType": ["Type 1"] * 4,
            "fold": [0, 1, 1, 0],
            "predicted": [8, 12, 12, 8],
        }
    )
    baseline = model.assign(predicted=11)
    assert compare_oof(model, baseline.iloc[::-1])["delta_mae"] == 1
    with pytest.raises(ValueError, match="exactly the same"):
        compare_oof(model, baseline.iloc[:-1])
    with pytest.raises(ValueError, match="must agree"):
        compare_oof(model, baseline.assign(observed=0))
