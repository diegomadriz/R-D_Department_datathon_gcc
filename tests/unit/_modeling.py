"""Synthetic train-only quality data for modeling contracts."""

import numpy as np
import pandas as pd

from gcc_strength.audit import FEATURES
from gcc_strength.eda import load_train


def synthetic_train(tmp_path, n=120):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({col: rng.uniform(0.1, 1, n) for col in FEATURES})
    frame["ProductType"] = np.where(np.arange(n) % 4 == 0, "Type 2", "Type 1")
    frame["_row_id"] = np.arange(n) * 2
    frame["strength_group_id"] = np.arange(n) // 2
    frame["split"] = "train"
    frame["Strength1"] = 100 + 30 * frame.Mineral_Phase_C + rng.normal(0, 2, n)
    frame["Strength7"] = 200 + 25 * frame.Physical_Metric_B + rng.normal(0, 3, n)
    frame["Strength28"] = 300 + 20 * frame.Mineral_Phase_G + rng.normal(0, 4, n)
    frame.loc[:79, "Physical_Metric_A"] = np.nan
    frame.loc[::19, "Mineral_Phase_H"] = np.nan
    frame.loc[::23, "Mineral_Phase_F"] = 0
    frame.loc[::17, "Strength1"] = np.nan
    path = tmp_path / "synthetic_train.parquet"
    frame.to_parquet(path)
    return load_train(path)
