"""Stage 2 train-only ladder, fixed ensemble gate and declared family selection.

Run with ``uv run python -m gcc_strength.ladder``. Cached results are accepted only
under an identical code/config/train manifest. This module has no evaluation executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT, load_yaml
from gcc_strength.cv import CVResult, cross_validate_model, load_model_config
from gcc_strength.eda import assert_train, load_train
from gcc_strength.metrics import compare_oof, regression_metrics

DEFAULT_OUT = ROOT / "outputs/fase3/stage2"
METRICS = ("r2", "mae", "mape_pct", "rmse", "r2_type1", "mae_type1")


def candidate_specs(config, target):
    """Enumerate a fixed factorial design before receiving any predictions."""
    stage = config["stage2"]
    specs = [
        dict(candidate=level, level=level, representation="raw", a="exclude", monotone=False)
        for level in ("L0", "L1")
    ]
    for representation in stage["representations"]:
        levels = [stage["linear_levels"][representation], *stage["comparison_levels"]]
        for level in levels:
            for a in stage["physical_metric_a"]:
                for monotone in (
                    [False, True]
                    if level == "L4" and target == stage["monotone_target"]
                    else [False]
                ):
                    specs.append(
                        dict(
                            candidate=f"{level}_{representation}_{a}"
                            + ("_monotone" if monotone else ""),
                            level=level,
                            representation=representation,
                            a=a,
                            monotone=monotone,
                        )
                    )
    return specs


def select_finalists(summary, config):
    """Apply the predeclared minimum-MAE rule, with deterministic parsimonious ties."""
    rule = config["selection"]
    repeated = summary.loc[summary.scheme.eq("repeated_group")]
    winners = []
    for target in sorted(repeated.target.unique()):
        for family, levels in rule["families"].items():
            candidates = repeated.loc[
                repeated.target.eq(target) & repeated.level.isin(levels)
            ].copy()
            minimum = candidates[rule["objective"]].min()
            tied = candidates.loc[
                candidates[rule["objective"]] <= minimum + rule["tie_tolerance"]
            ].copy()
            if tied.empty:
                raise ValueError(f"No finite candidate for {target}/{family}")
            tied["level_rank"] = tied.level.map(
                {v: i for i, v in enumerate(rule["complexity_order"])}
            )
            tied["a_rank"] = tied.physical_metric_a.map(
                {v: i for i, v in enumerate(rule["a_complexity_order"])}
            )
            tied["monotone_rank"] = tied.monotone.map(
                {v: i for i, v in enumerate(rule["monotone_complexity_order"])}
            )
            winner = (
                tied.sort_values(
                    ["n_features_mean", "level_rank", "a_rank", "monotone_rank", "candidate"],
                    kind="stable",
                )
                .iloc[0]
                .drop(labels=["level_rank", "a_rank", "monotone_rank"])
                .to_dict()
            )
            winners.append({"family": family, **winner})
    return pd.DataFrame(winners)


def fixed_ensemble(results, members):
    """Average predeclared members on identical OOF rows, without learned weights."""
    first = results[members[0]].oof.copy()
    keys = ["target", "repeat", "fold", "_row_id", "strength_group_id", "ProductType", "observed"]
    first = first.sort_values(["repeat", "_row_id"]).reset_index(drop=True)
    predictions = [first.predicted.to_numpy()]
    for name in members[1:]:
        other = results[name].oof.sort_values(["repeat", "_row_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(first[keys], other[keys], check_exact=True)
        predictions.append(other.predicted.to_numpy())
    first["predicted"] = np.mean(predictions, axis=0)
    first["model"] = "L6"
    rows = []
    for (repeat, fold), part in first.groupby(["repeat", "fold"], sort=True):
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                **regression_metrics(part.observed, part.predicted, part.ProductType),
            }
        )
    selections = results[members[0]].selections[["repeat", "fold"]].copy()
    selections["n_features"] = sum(
        results[name].selections.n_features.to_numpy() for name in members
    )
    selections["members"] = json.dumps(members)
    return CVResult(first, pd.DataFrame(rows), selections)


def ensemble_gate(ensemble, references, *, n_bootstrap=10000):
    """Require negative paired upper bounds against every specified reference."""
    comparisons = {
        name: compare_oof(ensemble.oof, result.oof, n_bootstrap=n_bootstrap)
        for name, result in references.items()
    }
    return all(c["ci_high"] < 0 for c in comparisons.values()), comparisons


def summarize(result, baseline, spec, target, scheme, config):
    """Fold mean/sample SD plus row-weighted paired OOF loss difference and CI."""
    row = dict(
        candidate=spec["candidate"],
        level=spec["level"],
        target=target,
        scheme=scheme,
        representation=spec["representation"],
        physical_metric_a=spec["a"],
        monotone=spec["monotone"],
        n_folds=len(result.metrics),
        n_rows=result.oof._row_id.nunique(),
        n_features_mean=float(result.selections.n_features.mean()),
    )
    for metric in METRICS:
        row[f"{metric}_mean"] = float(result.metrics[metric].mean())
        row[f"{metric}_sd"] = float(result.metrics[metric].std(ddof=1))
    row.update(
        compare_oof(result.oof, baseline.oof, n_bootstrap=config["bootstrap"]["n_resamples"])
    )
    return row


def _evaluate(train, target, spec, config, temporal):
    with threadpool_limits(limits=1):
        return cross_validate_model(
            train,
            target,
            spec["level"],
            config=config,
            temporal=temporal,
            physical_metric_a=spec["a"],
            model_options=dict(representation=spec["representation"], monotone=spec["monotone"]),
        )


def _save(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result.oof.to_csv(directory / "oof.csv", index=False)
    result.metrics.to_csv(directory / "folds.csv", index=False)
    result.selections.to_json(
        directory / "selections.json", orient="records", indent=2, double_precision=15
    )
    (directory / "complete").write_text("ok\n")


def _read(directory):
    return CVResult(
        pd.read_csv(directory / "oof.csv"),
        pd.read_csv(directory / "folds.csv"),
        pd.read_json(directory / "selections.json"),
    )


def _manifest(train, config):
    files = ["ladder.py", "models.py", "features.py", "cv.py", "metrics.py", "eda.py", "split.py"]
    return dict(
        config=config,
        code_hashes={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in files
        },
        train_sha256=hashlib.sha256(
            pd.util.hash_pandas_object(train, index=True).values.tobytes()
        ).hexdigest(),
        train_columns=list(train.columns),
        versions={
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "scipy")
        },
        medallion_config=hashlib.sha256((ROOT / "conf/medallion.yaml").read_bytes()).hexdigest(),
        lockbox_accessed=False,
    )


def run_ladder(train, *, output=DEFAULT_OUT, config=None, workers=2):
    """Persist the complete ladder and select three algorithm families per target."""
    assert_train(train)
    config = load_model_config() if config is None else config
    if config["bootstrap"]["confidence"] != 0.95:
        raise ValueError("The declared paired bootstrap requires confidence 0.95")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(train, config)
    manifest_file = output / "manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise ValueError("Output manifest differs: use a fresh output directory")
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    all_results, specs_by_target = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for target in TARGETS:
            specs_by_target[target] = candidate_specs(config, target)
            for scheme in ("repeated_group", "temporal"):
                for spec in specs_by_target[target]:
                    key = (target, scheme, spec["candidate"])
                    directory = output / "candidates" / target / scheme / spec["candidate"]
                    if (directory / "complete").is_file():
                        all_results[key] = _read(directory)
                    else:
                        future = executor.submit(
                            _evaluate, train, target, spec, config, scheme == "temporal"
                        )
                        pending[future] = (key, directory)
        for future in as_completed(pending):
            key, directory = pending[future]
            result = future.result()
            _save(result, directory)
            all_results[key] = result
            print(f"Completed {len(all_results)}: {' / '.join(key)}", flush=True)
    rows = []
    for target in TARGETS:
        for scheme in ("repeated_group", "temporal"):
            baseline = all_results[(target, scheme, "L1")]
            for spec in specs_by_target[target]:
                rows.append(
                    summarize(
                        all_results[(target, scheme, spec["candidate"])],
                        baseline,
                        spec,
                        target,
                        scheme,
                        config,
                    )
                )
    summary = pd.DataFrame(rows)
    finalists = select_finalists(summary, config)
    gates = []
    for target in TARGETS:
        selected = finalists.loc[finalists.target.eq(target)]
        for scheme in ("repeated_group", "temporal"):
            results = {
                name: result
                for (t, s, name), result in all_results.items()
                if t == target and s == scheme
            }
            ensemble = fixed_ensemble(results, config["stage2"]["ensemble_members"])
            _save(ensemble, output / "candidates" / target / scheme / "L6")
            spec = dict(
                candidate="L6", level="L6", representation="fixed_mix", a="exclude", monotone=False
            )
            row = summarize(ensemble, results["L1"], spec, target, scheme, config)
            references = {name: results[name] for name in ["L1", *selected.candidate]}
            accepted, comparisons = ensemble_gate(
                ensemble, references, n_bootstrap=config["bootstrap"]["n_resamples"]
            )
            row["ensemble_accepted"] = accepted if scheme == "repeated_group" else False
            rows.append(row)
            for reference, comparison in comparisons.items():
                gates.append(
                    dict(
                        target=target,
                        scheme=scheme,
                        reference=reference,
                        accepted=accepted if scheme == "repeated_group" else False,
                        **comparison,
                    )
                )
    summary = pd.DataFrame(rows).sort_values(["target", "scheme", "level", "candidate"])
    thresholds = load_yaml(ROOT / "conf/thresholds.yaml")["noise_floor_mape_pct_upper_bound"]
    summary["noise_floor_alarm"] = summary.mape_pct_mean < summary.target.map(thresholds)
    summary.to_csv(output / "ladder.csv", index=False)
    finalists.to_csv(output / "finalists.csv", index=False)
    pd.DataFrame(gates).to_csv(output / "ensemble_gate.csv", index=False)
    repeated = summary.loc[summary.scheme.eq("repeated_group")]
    temporal = summary.loc[summary.scheme.eq("temporal")]
    sensitivity = repeated.merge(
        temporal, on=["target", "candidate"], suffixes=("_cv", "_temporal")
    )
    sensitivity["temporal_minus_cv_mae"] = sensitivity.mae_mean_temporal - sensitivity.mae_mean_cv
    sensitivity.to_csv(output / "temporal_sensitivity.csv", index=False)
    display = summary[["target", "scheme", "candidate"]].copy()
    for metric in METRICS:
        display[metric] = summary.apply(
            lambda r, m=metric: f"{r[m + '_mean']:.5f} ± {r[m + '_sd']:.5f}", axis=1
        )
    display["delta_mae_vs_l1_95ci"] = summary.apply(
        lambda r: f"{r.delta_mae:.5f} [{r.ci_low:.5f}, {r.ci_high:.5f}]", axis=1
    )
    (output / "ladder.md").write_text(
        "# Escalera de CV sobre train\n\n" + display.to_markdown(index=False) + "\n\n"
        "SD entre folds; repeticiones correlacionadas. IC pareados por grupo, condicionales\n"
        "a OOF; selección de variantes sobre CV externa sujeta a optimismo y multiplicidad.\n"
        "L6 es candidata: consultar ensemble_gate.csv para aceptación por target.\n"
        "Sensibilidad temporal: calentamiento sin OOF; sus filas difieren de CV aleatoria.\n"
        "Unidades del target sin confirmar. Lockbox cerrado.\n"
    )
    return summary, finalists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_ladder(load_train(), output=args.output, workers=args.workers)


if __name__ == "__main__":
    main()
