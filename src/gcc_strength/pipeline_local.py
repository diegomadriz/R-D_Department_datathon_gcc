"""Run the whole medallion locally (pandas) and write Parquet files.

Usage::

    uv run python -m gcc_strength.pipeline_local                 # outputs/medallion/*.parquet
    uv run python -m gcc_strength.pipeline_local --out /tmp/x    # another output folder

The lockbox hash (``conf/lockbox.sha256``) is written on the first run. Later runs must
reproduce it exactly; otherwise the run fails. ``--rewrite-lockbox`` replaces it and is only
for a re-assignment approved in an ADR.

The summary prints counts only (rows and groups per side); no target or feature values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gcc_strength import bronze as bronze_layer
from gcc_strength import dq
from gcc_strength.config import (
    CHECKSUMS_FILE,
    DATA_DIR,
    LOCKBOX_FILE,
    MEDALLION_CONF,
    ROOT,
    load_medallion_conf,
)
from gcc_strength.gold import build_gold, gold_lockbox_digest, gold_lockbox_size
from gcc_strength.silver import REASONS, build_silver
from gcc_strength.split import GROUP_COL, SPLIT_COL, lockbox_check, read_lockbox, write_lockbox

DEFAULT_OUT = ROOT / "outputs" / "medallion"


@dataclass
class MedallionResult:
    bronze: pd.DataFrame
    silver: pd.DataFrame
    quarantine: pd.DataFrame
    gold: pd.DataFrame
    dq_results: pd.DataFrame
    lockbox_digest: str
    lockbox_written: bool = False
    counts: dict = field(default_factory=dict)


def run(
    csv_path: Path | str | None = None,
    *,
    conf_path: Path | str = MEDALLION_CONF,
    lockbox_path: Path | str = LOCKBOX_FILE,
    out_dir: Path | str | None = DEFAULT_OUT,
    rewrite_lockbox: bool = False,
    run_id: str | None = None,
    now: datetime | None = None,
) -> MedallionResult:
    conf = load_medallion_conf(conf_path)
    csv_path = Path(csv_path or DATA_DIR / conf["raw_file"])
    run_id = run_id or dq.new_run_id()
    now = now or datetime.now(UTC)

    rows: list[dict] = []
    bronze = bronze_layer.read_bronze(csv_path, source_file=csv_path.name, ingest_ts=now)
    file_hash = bronze["_file_hash"].iloc[0]
    if csv_path.name == conf["raw_file"] and CHECKSUMS_FILE.is_file():
        expected_hash = bronze_layer.expected_sha256(CHECKSUMS_FILE, csv_path.name)
        rows.append(
            dq.dq_row(
                "bronze",
                "source_sha256_matches_checksums",
                "error",
                len(bronze),
                0 if file_hash == expected_hash else len(bronze),
                "fail",
            )
        )
    rows.append(dq.dq_row("bronze", "rows_ingested", "info", len(bronze), 0, "none"))

    silver, quarantine, silver_rows = build_silver(bronze)
    rows += silver_rows
    gold, gold_rows = build_gold(silver, conf["split"])
    rows += gold_rows

    digest = gold_lockbox_digest(gold)
    n_test = gold_lockbox_size(gold)
    stored = read_lockbox(lockbox_path)
    written = False
    if stored is None or rewrite_lockbox:
        write_lockbox(lockbox_path, digest)
        stored, written = digest, True
    rows += lockbox_check(digest, stored, n_test)

    by_reason = {r: int((quarantine["reason"] == r).sum()) for r in REASONS}
    dups = dq.rows_failed(rows, "silver", "exact_duplicate")
    rows += dq.run_reconciliation(
        len(bronze), len(silver), by_reason, dups, file_hash, conf.get("expected_counts")
    )
    dq.assert_no_errors(rows)
    dq_results = dq.to_frame(rows, run_id, now)

    counts = {
        "bronze": len(bronze),
        "silver": len(silver),
        "quarantine": {k: v for k, v in by_reason.items() if v},
        "duplicates_removed": dups,
        "gold": len(gold),
        "rows": gold[SPLIT_COL].value_counts().to_dict(),
        "groups": gold.groupby(SPLIT_COL)[GROUP_COL].nunique().to_dict(),
    }
    result = MedallionResult(bronze, silver, quarantine, gold, dq_results, digest, written, counts)
    if out_dir is not None:
        write_parquet(result, Path(out_dir))
    return result


def write_parquet(result: MedallionResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "datathon_data_bronze": result.bronze,
        "datathon_data_silver": result.silver,
        "silver_quarantine": result.quarantine,
        "datathon_data_gold": result.gold,
        "dq_results": result.dq_results,
    }
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rewrite-lockbox", action="store_true")
    args = parser.parse_args()
    res = run(args.csv, out_dir=args.out, rewrite_lockbox=args.rewrite_lockbox)
    c = res.counts
    print(
        f"bronze={c['bronze']} silver={c['silver']} quarantine={c['quarantine']} "
        f"duplicates_removed={c['duplicates_removed']}"
    )
    print(f"gold rows by split={c['rows']} groups by split={c['groups']}")
    state = "written" if res.lockbox_written else "verified"
    print(f"lockbox sha256={res.lockbox_digest} ({state})")
    print(f"parquet -> {args.out}")


if __name__ == "__main__":
    main()
