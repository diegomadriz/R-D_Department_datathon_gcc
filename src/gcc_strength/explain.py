"""Bounded train-only explanations of fitted single-target model components."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT
from gcc_strength.cv import load_model_config
from gcc_strength.eda import assert_train, load_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.models import PhysicalMetricARegressor
from gcc_strength.multioutput import fit_architecture
from gcc_strength.stage3 import read_finalists


def explain_model(model, X, *, max_samples=32, background_size=16):
    """Explain fitted component predictions before any multi-output projection.

    The fallback and observed-A expert are explained separately. Sampling is fixed;
    transformed features and preprocessing belong to the fitted training pipeline.
    """
    import shap

    X = X.loc[:, list(INPUT_COLUMNS)]
    branches = [("model", model, X)]
    if isinstance(model, PhysicalMetricARegressor):
        branches = [("fallback", model.fallback_, X)]
        if model.expert_ is not None:
            branches.append(("expert", model.expert_, X.loc[X.Physical_Metric_A.notna()]))
    results = []
    for branch, pipe, data in branches:
        if data.empty:
            continue
        data = data.sample(n=min(len(data), max_samples), random_state=42)
        transformed = pipe[:-1].transform(data)
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(
                transformed,
                index=data.index,
                columns=pipe.named_steps["preprocess"].get_feature_names_out(),
            )
        names = list(transformed.columns)
        estimator = pipe.named_steps["model"]
        if hasattr(estimator, "model_") and estimator.model_.__class__.__name__ == "LGBMRegressor":
            explainer = shap.TreeExplainer(estimator.model_)
            values = explainer.shap_values(transformed)
        else:
            background = transformed.iloc[: min(background_size, len(transformed))]
            explainer = shap.Explainer(
                estimator.predict, background, algorithm="permutation", seed=42
            )
            values = explainer(transformed, max_evals=2 * len(names) + 1).values
        for i, row_id in enumerate(data.index):
            for j, feature in enumerate(names):
                results.append(
                    {
                        "branch": branch,
                        "row": str(row_id),
                        "feature": feature,
                        "value": float(transformed.iloc[i, j]),
                        "shap": float(values[i, j]),
                    }
                )
    return pd.DataFrame(results)


def physics_consistency_report(explanations, target):
    """Compare conditional SHAP trends to the one supported candidate sign.

    Signs are descriptive associations, not causal effects or identified chemistry.
    """
    rows = []
    for (branch, feature), part in explanations.groupby(["branch", "feature"], sort=True):
        varying = part.value.nunique() > 1 and part.shap.nunique() > 1
        rho = float(spearmanr(part.value, part.shap).statistic) if varying else np.nan
        expected = (
            "positive"
            if target == "Strength1" and feature == "Physical_Metric_B"
            else "undetermined"
        )
        status = "undetermined"
        if expected == "positive" and np.isfinite(rho):
            status = "consistent" if rho > 0 else "discrepant"
        explanation = (
            "Conditional association; correlated inputs, scaling and "
            "unidentified chemistry limit interpretation."
        )
        if feature == "Physical_Metric_B" and target == "Strength1":
            explanation = (
                "Blaine identity is provisional; a disagreement can reflect correlated "
                "inputs, drift, model interactions or the bounded SHAP sample. "
                "The train-only EDA found a small positive association after adjustment."
            )
        elif feature == "Mineral_Phase_D" and target == "Strength28":
            explanation = (
                "A negative effect is hypothesized if D is limestone, but the adjusted "
                "train-only EDA was inconclusive. Confounding and uncertain identity "
                "prevent treating either SHAP sign as a physical contradiction."
            )
        elif feature == "Mineral_Phase_F" and target == "Strength1":
            explanation = (
                "A positive effect is hypothesized if F is C3A, but the adjusted "
                "train-only EDA was inconclusive; sulfate interactions and uncertain "
                "identity can change the conditional association."
            )
        elif feature == "Physical_Metric_B" and target == "Strength28":
            explanation = (
                "The adjusted train-only EDA found a negative association, sensitive "
                "to controls; grinding, limestone and drift are plausible explanations, "
                "not identified mechanisms or a universal physical sign."
            )
        rows.append(
            {
                "branch": branch,
                "feature": feature,
                "target": target,
                "rho_value_shap": rho,
                "expected": expected,
                "status": status,
                "interpretation": explanation,
            }
        )
    return pd.DataFrame(rows)


def explain_finalists(train, *, output=ROOT / "outputs/fase3/explanation", max_samples=32):
    """Fit the comparable independent architecture and explain its nine components.

    Explanations are in-sample, before isotonic projection; they are descriptive,
    not estimates of generalization or causal effects.
    """
    assert_train(train)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    families = read_finalists(ROOT / "outputs/fase3/stage2/finalists.csv")
    config = load_model_config()
    results = []
    for family, specs in sorted(families.items()):
        with threadpool_limits(limits=1):
            model, selection = fit_architecture(train, specs, "a", config)
            for target, component in zip(TARGETS, model.models_, strict=True):
                explanation = explain_model(component, train, max_samples=max_samples)
                explanation.to_csv(output / f"{family}_{target}_shap.csv", index=False)
                report = physics_consistency_report(explanation, target)
                report.to_csv(output / f"{family}_{target}_physics.csv", index=False)
                importance = (
                    explanation.assign(abs_shap=lambda frame: frame.shap.abs())
                    .groupby(["branch", "feature"], as_index=False)
                    .abs_shap.mean()
                    .sort_values("abs_shap", ascending=False)
                )
                importance.to_csv(output / f"{family}_{target}_global.csv", index=False)
                _plot_explanation(explanation, importance, output / f"{family}_{target}")
                results.append(
                    {
                        "family": family,
                        "target": target,
                        "profile_index": selection["profile_index"],
                    }
                )
    return pd.DataFrame(results)


def _plot_explanation(explanation, importance, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for branch in sorted(explanation.branch.unique()):
        top = importance.loc[importance.branch.eq(branch)].head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.barh(top.feature, top.abs_shap)
        ax.set(title=f"{branch}: mean absolute SHAP", xlabel="Mean absolute SHAP")
        fig.savefig(f"{prefix}_{branch}_global.png", dpi=130)
        plt.close(fig)
        local = explanation.loc[explanation.branch.eq(branch)]
        local = local.loc[local.row.eq(local.row.iloc[0])].copy()
        local = local.reindex(local.shap.abs().sort_values(ascending=False).head(12).index)
        local = local.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.barh(local.feature, local.shap)
        ax.set(title=f"{branch}: local SHAP (one train row)", xlabel="SHAP contribution")
        fig.savefig(f"{prefix}_{branch}_local.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fase3/explanation")
    args = parser.parse_args()
    explain_finalists(load_train(), output=args.output, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
