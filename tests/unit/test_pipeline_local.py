"""End-to-end local run: Parquet outputs and a reproducible lockbox hash."""

import pandas as pd
import pytest

from gcc_strength import pipeline_local
from gcc_strength.config import LOCKBOX_FILE
from gcc_strength.split import SplitViabilityError, read_lockbox, write_lockbox


def test_two_runs_same_lockbox(tmp_path):
    lockbox = tmp_path / "lockbox.sha256"
    first = pipeline_local.run(lockbox_path=lockbox, out_dir=tmp_path / "a")
    second = pipeline_local.run(lockbox_path=lockbox, out_dir=tmp_path / "b")
    assert first.lockbox_written and not second.lockbox_written
    assert first.lockbox_digest == second.lockbox_digest == read_lockbox(lockbox)
    assert first.lockbox_digest == read_lockbox(LOCKBOX_FILE)
    for name in ("datathon_data_gold", "datathon_data_silver", "silver_quarantine"):
        a = pd.read_parquet(tmp_path / "a" / f"{name}.parquet")
        b = pd.read_parquet(tmp_path / "b" / f"{name}.parquet")
        pd.testing.assert_frame_equal(
            a.drop(columns="_ingest_ts", errors="ignore"),
            b.drop(columns="_ingest_ts", errors="ignore"),
        )


def test_changed_lockbox_fails(tmp_path):
    lockbox = tmp_path / "lockbox.sha256"
    write_lockbox(lockbox, "0" * 64)
    with pytest.raises(SplitViabilityError):
        pipeline_local.run(lockbox_path=lockbox, out_dir=None)
