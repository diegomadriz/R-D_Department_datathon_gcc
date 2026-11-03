"""Synthetic Bronze frames for unit tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from gcc_strength.audit import EXPECTED_COLUMNS

FIXED_TS = datetime(2026, 1, 1, tzinfo=UTC)


def make_bronze(rows: list[dict]) -> pd.DataFrame:
    """Synthetic Bronze frame: missing cells are empty strings, as in the raw CSV."""
    data = pd.DataFrame([{c: str(r.get(c, "")) for c in EXPECTED_COLUMNS} for r in rows])
    data.insert(0, "_row_id", range(len(data)))
    data["_ingest_ts"] = pd.Timestamp(FIXED_TS)
    data["_source_file"] = "synthetic.csv"
    data["_file_hash"] = "0" * 64
    return data
