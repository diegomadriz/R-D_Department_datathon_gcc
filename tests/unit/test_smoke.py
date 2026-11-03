"""Smoke tests: package imports and the raw CSV is intact."""

import hashlib
from pathlib import Path

import gcc_strength

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CSV_PATH = DATA_DIR / "datathon_data_gcc.csv"
CHECKSUMS_PATH = DATA_DIR / "CHECKSUMS.txt"


def _expected_sha256(filename: str) -> str:
    for line in CHECKSUMS_PATH.read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if name == filename:
            return digest
    raise AssertionError(f"{filename} not listed in {CHECKSUMS_PATH}")


def test_package_imports():
    assert gcc_strength.__version__


def test_raw_csv_exists():
    assert CSV_PATH.is_file()


def test_raw_csv_sha256_matches_checksums():
    actual = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert actual == _expected_sha256(CSV_PATH.name)
