"""Physical curve and descriptive fitting checks on synthetic strengths."""

import numpy as np
import pandas as pd
import pytest

from gcc_strength import kinetics
from gcc_strength.audit import TARGETS
from gcc_strength.eda import train_only


def test_curve_matches_day28_and_is_monotone():
    times = np.linspace(0.01, 28, 300)
    values = kinetics.strength_curve(times, 7000, 0.25, 0.7)
    assert values[-1] == pytest.approx(7000)
    assert np.all(np.diff(values) >= 0)
    assert kinetics.strength_curve(kinetics.AGES, 7000, 0).tolist() == [7000] * 3


@pytest.mark.parametrize("values", [[1, 0, 3], [1, np.nan, 3], [1, 2], [1, -2, 3]])
def test_fit_rejects_incomplete_or_invalid_strengths(values):
    with pytest.raises(ValueError):
        kinetics.fit_curve(values)


def test_curve_rejects_invalid_physical_parameters():
    for args in (([0, 7], 100, 0.2), ([1, 7], -1, 0.2), ([1, 7], 1, -0.2)):
        with pytest.raises(ValueError):
            kinetics.strength_curve(*args)


def test_fixed_fit_recovers_known_parameters():
    observed = kinetics.strength_curve(kinetics.AGES, 7200, 0.28)
    fitted = kinetics.fit_curve(observed)
    assert fitted["f28"] == pytest.approx(7200, rel=1e-6)
    assert fitted["s"] == pytest.approx(0.28, rel=1e-6)


def test_type_fit_recovers_shared_exponent_and_improves_sse():
    values = np.array(
        [
            kinetics.strength_curve(kinetics.AGES, f, s, 0.8)
            for f, s in [(6000, 0.15), (8000, 0.3), (7500, 0.2)]
        ]
    )
    fixed = [kinetics.fit_curve(v) for v in values]
    initial = np.array([[r["f28"], r["s"]] for r in fixed])
    parameters, exponent = kinetics.fit_type_exponent(values, initial)
    assert exponent == pytest.approx(0.8, abs=1e-4)
    predictions = np.array(
        [kinetics.strength_curve(kinetics.AGES, *p, exponent) for p in parameters]
    )
    assert np.max(np.abs(predictions - values)) < 0.1


def test_fit_samples_omits_incomplete_targets_and_rejects_test():
    values = [
        kinetics.strength_curve(kinetics.AGES, 7000, 0.2),
        kinetics.strength_curve(kinetics.AGES, 8000, 0.3),
        [1, np.nan, 3],
    ]
    frame = pd.DataFrame(values, columns=TARGETS)
    frame["ProductType"] = "Type 1"
    frame["split"] = "train"
    fits = kinetics.fit_samples(train_only(frame))
    assert len(fits) == 4
    assert set(fits.model) == {"sqrt", "free"}
    summary = kinetics.summarize_fits(fits)
    assert summary.n_complete.tolist() == [2, 2]
    assert summary.MAPE_Strength1.max() < 0.001
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="test rows are forbidden"):
        kinetics.fit_samples(frame)


def test_r7_anchored_diagnostic_recovers_a_true_sqrt_curve():
    frame = pd.DataFrame([kinetics.strength_curve(kinetics.AGES, 7000, 0.25)], columns=TARGETS)
    frame["ProductType"] = "Type 1"
    frame["split"] = "train"
    result = kinetics.anchored_diagnostic(train_only(frame)).iloc[0]
    assert result.s7_median == pytest.approx(0.25)
    assert result.R1_MAPE_pct < 1e-10
    with pytest.raises(ValueError, match="test rows are forbidden"):
        kinetics.anchored_diagnostic(frame)
