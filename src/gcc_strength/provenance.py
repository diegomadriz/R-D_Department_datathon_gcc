"""Stable train input identity independent of ingestion timestamps and Spark dtypes."""

import hashlib

from gcc_strength.audit import TARGETS
from gcc_strength.eda import assert_train
from gcc_strength.features import INPUT_COLUMNS
from gcc_strength.split import GROUP_COL


def train_fingerprint(train):
    """Hash sorted quality, observed labels and partition metadata, never ingestion time.

    Twelve significant digits keep the identity stable across platforms: text-to-float parsing
    in Silver may differ by a few units in the last place between local and serverless runs.
    """
    assert_train(train)
    columns = ["_row_id", GROUP_COL, *INPUT_COLUMNS, *TARGETS]
    text = train.sort_values("_row_id")[columns].to_csv(
        index=False, float_format="%.12g", lineterminator="\n"
    )
    return hashlib.sha256(text.encode()).hexdigest()
