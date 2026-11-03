"""Bronze keeps the CSV verbatim (strings) and adds lineage."""

import hashlib

import pandas as pd

from gcc_strength.audit import EXPECTED_COLUMNS
from gcc_strength.bronze import LINEAGE_COLUMNS, data_columns, expected_sha256, read_bronze
from gcc_strength.config import CHECKSUMS_FILE, DATA_DIR


def test_real_csv_all_strings_with_lineage():
    path = DATA_DIR / "datathon_data_gcc.csv"
    b = read_bronze(path, source_file="datathon_data_gcc.csv")
    assert len(b) == 791
    assert data_columns(b) == EXPECTED_COLUMNS
    assert set(LINEAGE_COLUMNS) <= set(b.columns)
    assert b["_row_id"].tolist() == list(range(791))
    assert all(b[c].map(type).eq(str).all() for c in EXPECTED_COLUMNS)
    assert b["_file_hash"].nunique() == 1
    assert b["_file_hash"].iloc[0] == expected_sha256(CHECKSUMS_FILE, path.name)
    assert b["_ingest_ts"].dt.tz is not None and b["_ingest_ts"].nunique() == 1


def test_no_transformation(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("ProductType,Strength1\n  type 1 ,\nType 2,1.50\n", encoding="utf-8")
    b = read_bronze(csv)
    assert b["ProductType"].tolist() == ["  type 1 ", "Type 2"]
    assert b["Strength1"].tolist() == ["", "1.50"]
    assert b["_file_hash"].iloc[0] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert b["_source_file"].iloc[0] == str(csv)
    assert isinstance(b["_ingest_ts"].iloc[0], pd.Timestamp)
