"""Tests of descriptive physical consistency classification."""

import numpy as np
import pandas as pd

from gcc_strength.explain import physics_consistency_report


def test_physics_report_exposes_disagreement_and_unknown_signs():
    frame = pd.DataFrame(
        [
            {"branch": "model", "feature": feature, "value": value, "shap": contribution}
            for feature, contributions in [
                ("Physical_Metric_B", [2.0, 1.0, 0.0]),
                ("Mineral_Phase_H", [0.0, 1.0, 2.0]),
                ("constant", [0.0, 0.0, 0.0]),
            ]
            for value, contribution in enumerate(contributions)
        ]
    )
    report = physics_consistency_report(frame, "Strength1").set_index("feature")
    assert report.loc["Physical_Metric_B", "status"] == "discrepant"
    assert report.loc["Mineral_Phase_H", "status"] == "undetermined"
    assert np.isnan(report.loc["constant", "rho_value_shap"])
    assert physics_consistency_report(frame, "Strength28").status.eq("undetermined").all()


def test_hypothesized_sign_without_eda_support_is_not_a_constraint():
    frame = pd.DataFrame(
        [
            {"branch": "model", "feature": feature, "value": value, "shap": -value}
            for feature in ("Mineral_Phase_D", "Mineral_Phase_F")
            for value in range(3)
        ]
    )
    for target, feature in (("Strength28", "Mineral_Phase_D"), ("Strength1", "Mineral_Phase_F")):
        result = physics_consistency_report(frame, target).set_index("feature").loc[feature]
        assert result.status == "undetermined"
        assert "inconclusive" in result.interpretation


def test_explanations_cover_local_rows_and_tree(tmp_path):
    from _modeling import synthetic_train

    from gcc_strength.explain import explain_model
    from gcc_strength.features import INPUT_COLUMNS
    from gcc_strength.models import build_model

    train = synthetic_train(tmp_path)
    known = train.loc[train.Strength1.notna()]
    x = known[list(INPUT_COLUMNS)]
    for level in ("L2", "L4", "L5"):
        model = build_model(level, target="Strength1").fit(x, known.Strength1)
        result = explain_model(model, x, max_samples=3, background_size=2)
        assert result.row.nunique() == 3
        assert np.isfinite(result.shap).all()
        assert result.groupby("row").size().nunique() == 1


def test_explanation_plots_are_written_per_branch(tmp_path):
    from gcc_strength.explain import _plot_explanation

    explanation = pd.DataFrame(
        [{"branch": "model", "row": "1", "feature": "x", "value": 2.0, "shap": 1.0}]
    )
    importance = pd.DataFrame([{"branch": "model", "feature": "x", "abs_shap": 1.0}])
    _plot_explanation(explanation, importance, tmp_path / "example")
    assert (tmp_path / "example_model_global.png").stat().st_size > 0
    assert (tmp_path / "example_model_local.png").stat().st_size > 0
