"""Bronze layer: the raw CSV exactly as delivered, every column as a string, plus lineage.

No value is trimmed, parsed or replaced: empty cells stay as empty strings. The only additions
are the lineage columns:

- ``_row_id``: 0-based position of the data record in the file (0 = first data row). Every
  order-dependent step downstream sorts by it, because Spark does not guarantee read order.
- ``_ingest_ts``: UTC ingestion timestamp (same value for the whole load).
- ``_source_file``: path or URI the file was read from.
- ``_file_hash``: SHA-256 of the file bytes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gcc_strength.audit import sha256

LINEAGE_COLUMNS = ["_row_id", "_ingest_ts", "_source_file", "_file_hash"]


def expected_sha256(checksums_path: Path | str, filename: str) -> str:
    """Digest listed for ``filename`` in a ``sha256sum``-style checksums file."""
    for line in Path(checksums_path).read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        if name in {filename, f"data/{filename}"}:
            return digest
    raise KeyError(f"{filename} is not listed in {checksums_path}")


def read_bronze(
    path: Path | str,
    *,
    source_file: str | None = None,
    ingest_ts: datetime | None = None,
) -> pd.DataFrame:
    """Read the CSV with every column as a string and append the lineage columns."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    ts = pd.Timestamp(ingest_ts or datetime.now(UTC))
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = raw.copy()
    out.insert(0, "_row_id", pd.Series(range(len(raw)), dtype="int64"))
    out["_ingest_ts"] = pd.Series([ts] * len(raw), dtype="datetime64[ns, UTC]")
    out["_source_file"] = source_file if source_file is not None else str(path)
    out["_file_hash"] = sha256(path)
    return out


def data_columns(bronze: pd.DataFrame) -> list[str]:
    """Columns that come from the CSV (everything except lineage)."""
    return [c for c in bronze.columns if c not in LINEAGE_COLUMNS]
