"""Export train-only aggregates and a one-at-a-time surrogate of the frozen GPR.

The output holds no individual rows: only grid predictions, percentiles, binned
counts above a disclosure minimum and residual quantiles. Lockbox rows are never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gcc_strength.audit import TARGETS
from gcc_strength.config import ROOT
from gcc_strength.cv import load_model_config
from gcc_strength.eda import assert_train, load_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.train import _fit

SEED = 42
MIN_CELL = 5
TYPES = ("Type 1", "Type 2", "Type 3")
METAS = {"Type 1": 7000, "Type 2": 8000, "Type 3": 7400}  # assumption, not a GCC spec
GRID = [1, *range(5, 100, 5), 99]
VARIABLES = {
    "PMB": "Physical_Metric_B",
    "PMA": "Physical_Metric_A",
    "MPC": "Mineral_Phase_C",
    "MPB": "Mineral_Phase_B",
    "MPD": "Mineral_Phase_D",
    "MPF": "Mineral_Phase_F",
    "MPE": "Mineral_Phase_E",
    "MPG": "Mineral_Phase_G",
    "MPH": "Mineral_Phase_H",
    "CCA": "Chemical_Composition_A",
    "CCB": "Chemical_Composition_B",
}
EVIDENCE = ROOT / "data/training_evidence/diagnostics"


def rounded(values, digits=1):
    return [round(float(v), digits) for v in values]


def density(x, y, bins_x=12, bins_y=10):
    """Binned counts; cells under the disclosure minimum are dropped."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < MIN_CELL:
        return None
    if len(frame) < 100:  # Small types need coarser cells to clear the disclosure minimum.
        bins_x, bins_y = 4, 4
    x_edges = np.linspace(frame.x.quantile(0.01), frame.x.quantile(0.99), bins_x + 1)
    y_edges = np.linspace(frame.y.quantile(0.01), frame.y.quantile(0.99), bins_y + 1)
    counts, _, _ = np.histogram2d(frame.x, frame.y, bins=[x_edges, y_edges])
    cells = [
        [int(i), int(j), int(counts[i, j])]
        for i in range(bins_x)
        for j in range(bins_y)
        if counts[i, j] >= MIN_CELL
    ]
    return dict(x=rounded(x_edges, 3), y=rounded(y_edges, 1), cells=cells, n=len(frame))


def main(output):
    train = load_train().sort_values("_row_id", kind="stable").reset_index(drop=True)
    assert_train(train)
    if len(train) != 621:
        raise ValueError("Expected the frozen 621 training rows")
    final = load_model_config()["final"]
    specs = [final["families"]["gaussian_process"][t] for t in TARGETS]
    model = _fit(train, specs)
    columns = list(INPUT_COLUMNS)

    grids = {k: train[c].quantile([p / 100 for p in GRID]).to_numpy() for k, c in VARIABLES.items()}

    def percentile(key, value):
        return float(np.interp(value, grids[key], GRID))

    types = {}
    for name in TYPES:
        part = train.loc[train.ProductType.eq(name)]
        anchor = part[columns].iloc[[0]].copy()
        for column in columns:
            if column != "ProductType":
                anchor[column] = part[column].median()
        anchor["Physical_Metric_A"] = np.nan  # missing in most history; default state
        base = model.predict(anchor[columns])[0]
        deltas, envelope, start = {}, {}, {}
        for key, column in VARIABLES.items():
            rows = pd.concat([anchor] * len(GRID), ignore_index=True)
            rows[column] = grids[key]
            deltas[key] = np.round(model.predict(rows[columns]) - base, 1).T.tolist()
            values = part[column].dropna()
            if len(values):
                envelope[key] = [
                    round(percentile(key, values.quantile(0.01)), 1),
                    round(percentile(key, values.quantile(0.99)), 1),
                ]
                start[key] = round(percentile(key, values.median()), 1)
        types[name] = dict(
            n=len(part),
            base=rounded(base),
            deltas=deltas,
            envelope=envelope,
            start=start,
            meta=METAS[name],
        )

    # Surrogate error against the full GPR, on complete training inputs.
    predicted = model.predict(train[columns])
    surrogate = np.full_like(predicted, np.nan, dtype=float)
    for name in TYPES:
        spec = types[name]
        for i in train.index[train.ProductType.eq(name)]:
            estimate = np.array(spec["base"])
            for key, column in VARIABLES.items():
                value = train.at[i, column]
                if pd.notna(value):
                    estimate = estimate + np.array(
                        [np.interp(value, grids[key], spec["deltas"][key][j]) for j in range(3)]
                    )
            surrogate[i] = np.maximum.accumulate(estimate)
    surrogate_mae = np.nanmean(np.abs(surrogate - predicted), axis=0)

    oof = pd.read_csv(EVIDENCE / "oof.csv")
    oof = oof.loc[oof.family.eq("gaussian_process")]
    if not set(oof._row_id).issubset(set(train._row_id)):
        raise ValueError("OOF must belong to training rows")
    intervals = pd.read_csv(EVIDENCE / "gaussian_process_intervals.csv")
    half_width, quantiles, pit = {}, {}, {}
    levels = np.linspace(0, 1, 41)
    for target in TARGETS:
        block = intervals.loc[intervals.target.eq(target)]
        half_width[target] = {
            scope: round(float(((part.upper - part.lower) / 2).mean()), 1)
            for scope, part in [("all", block), *block.groupby("ProductType")]
            if scope == "all" or scope in TYPES
        }
        errors = oof.loc[oof.target.eq(target)]
        signed = errors.observed - errors.predicted
        quantiles[target] = rounded(signed.quantile(levels))
    # Calibration of the global residual distribution, by type, at 28 days.
    errors = oof.loc[oof.target.eq("Strength28")]
    signed = np.sort((errors.observed - errors.predicted).to_numpy())
    for scope in TYPES:
        part = errors.loc[errors.ProductType.eq(scope)]
        values = np.searchsorted(signed, part.observed - part.predicted) / len(signed)
        pit[scope] = round(
            float(max(abs((values < q).mean() - q) for q in (0.05, 0.1, 0.2, 0.3, 0.5))), 3
        )

    # Aggregated historical status, one prediction per row and per repeat 0.
    first = intervals.loc[intervals.repeat.eq(0) & intervals.target.eq("Strength28")].merge(
        oof.loc[oof.repeat.eq(0) & oof.target.eq("Strength28"), ["_row_id", "predicted"]],
        on="_row_id",
    )
    history = {}
    for name in TYPES:
        part = first.loc[first.ProductType.eq(name)]
        meta = METAS[name]
        history[name] = dict(
            ok=int((part.lower >= meta).sum()),
            warn=int(((part.predicted >= meta) & (part.lower < meta)).sum()),
            bad=int((part.predicted < meta).sum()),
        )

    scopes = [("Todos", train), *((name, train[train.ProductType.eq(name)]) for name in TYPES)]
    distributions, quality = {}, {}
    for scope, part in scopes:
        distributions[scope] = {}
        for target in TARGETS:
            values = part[target].dropna()
            counts, edges = np.histogram(values, bins=14)
            counts = [int(c) if c >= MIN_CELL else 0 for c in counts]
            distributions[scope][target] = dict(
                edges=rounded(edges, 0), counts=counts, n=len(values)
            )
        quality[scope] = {
            key: {t: density(part[column], part[t]) for t in TARGETS}
            for key, column in VARIABLES.items()
        }

    residual_view = {}
    repeat = oof.loc[oof.repeat.eq(0)]
    for scope in ("Todos", *TYPES):
        part = repeat if scope == "Todos" else repeat.loc[repeat.ProductType.eq(scope)]
        residual_view[scope] = {
            t: density(
                part.loc[part.target.eq(t), "observed"],
                part.loc[part.target.eq(t), "predicted"],
            )
            for t in TARGETS
        }

    importance = {}
    for target in TARGETS:
        table = pd.read_csv(EVIDENCE / f"artifacts/gaussian_process/{target}/importance.csv")
        table = table.loc[table.branch.eq("model")].sort_values("importance", ascending=False)
        importance[target] = [
            dict(column=row.feature, value=round(float(row.importance), 2))
            for row in table.head(6).itertuples()
        ]

    snapshot = dict(
        seed=SEED,
        min_cell=MIN_CELL,
        train_rows=len(train),
        grid=GRID,
        columns=VARIABLES,
        values={k: rounded(v, 3) for k, v in grids.items()},
        types=types,
        half_width=half_width,
        residual_quantiles=quantiles,
        residual_levels=rounded(levels, 3),
        calibration_max_deviation=pit,
        surrogate_mae=dict(zip(TARGETS, rounded(surrogate_mae), strict=True)),
        history=history,
        distributions=distributions,
        quality=quality,
        predicted_vs_observed=residual_view,
        importance=importance,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    output.write_text(f"window.SNAPSHOT={text};\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                bytes=output.stat().st_size,
                surrogate_mae=snapshot["surrogate_mae"],
                calibration=pit,
                history=history,
                half_width=half_width,
                base={k: types[k]["base"] for k in TYPES},
                importance={t: [i["column"] for i in importance[t][:4]] for t in TARGETS},
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
