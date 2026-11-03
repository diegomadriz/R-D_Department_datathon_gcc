"""Shared fixtures: one local medallion run on the real CSV (no files written)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gcc_strength import pipeline_local

FIXED_TS = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="session")
def medallion(tmp_path_factory):
    lockbox = tmp_path_factory.mktemp("lockbox") / "lockbox.sha256"
    return pipeline_local.run(lockbox_path=lockbox, out_dir=None, run_id="test", now=FIXED_TS)
