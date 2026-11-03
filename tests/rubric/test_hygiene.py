"""Hygiene checks of the evaluation rubric that must hold from Phase 0 onwards.

- The raw CSV matches ``data/CHECKSUMS.txt``.
- No target is listed as a feature.
- Every fixed seed in project code is 42.
- Versionable files contain no references to the build process tooling and no
  workspace hosts, e-mail addresses or access tokens.

The forbidden vocabulary is stored reversed so that this file never matches itself.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gcc_strength import audit

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "datathon_data_gcc.csv"
CHECKSUMS_PATH = REPO_ROOT / "data" / "CHECKSUMS.txt"
TARGETS = {"Strength1", "Strength7", "Strength28"}
SEED = 42


def _rev(words: list[str]) -> list[str]:
    return [w[::-1] for w in words]


# Tool and vendor names, instruction files and commit trailers (reversed).
_FORBIDDEN_TERMS = _rev(
    [
        "edualc",
        "xedoc",
        "ciporhtna",
        "ianepo",
        "inimeg",
        "tpgtahc",
        "tolipoc",
        "ytivargitna",
        "dm.stnega",
        "dm.edualc",
        "dm.inimeg",
        ":yb-derohtua-oc",
        "htiw detareneg",
        "etnega",
        "tnegabus",
    ]
)
# Whole words: the Spanish/English acronym for the tooling and the short CLI name.
_FORBIDDEN_WORDS = _rev(["ai", "sai", "yga", "mll", "smll", "tnega", "stnega"])
# "AI/BI" is a Databricks product name, not a reference to the build process.
_PRODUCT_NAME_EXCEPTIONS = re.compile(r"\bAI/BI\b", re.IGNORECASE)

_FORBIDDEN_RE = re.compile(
    "|".join(
        [re.escape(t) for t in _FORBIDDEN_TERMS]
        + [rf"\b{re.escape(w)}\b" for w in _FORBIDDEN_WORDS]
    ),
    re.IGNORECASE,
)

_HOST_SUFFIXES = _rev(["moc.skcirbatad.duolc", "ten.skcirbataderuza", "moc.skcirbatad.pcg"])
_SENSITIVE_RE = re.compile(
    "|".join(
        [
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",  # e-mail
            r"\bdapi[0-9a-f]{32}\b",  # personal access token
            r"^\s*\[(?:DEFAULT|[A-Za-z0-9_-]+)\]\s*$\n^\s*host\s*=",  # CLI profile block
        ]
        + [re.escape(s) for s in _HOST_SUFFIXES]
    ),
    re.MULTILINE,
)

# The official challenge statement is third-party text and may mention the generic acronym.
_GENERIC_WORD_ALLOWLIST = {"docs/README.md"}
# Third-party package names in the lock file are not project text.
_SKIP = {"uv.lock"}


def _versionable_files() -> list[str]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("git repository not available")
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({p for p in out.splitlines() if p and (REPO_ROOT / p).is_file()})


def _read_text(rel: str) -> str | None:
    try:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None  # binary file (e.g. the rubric workbook)


def test_raw_csv_matches_checksum():
    expected = None
    for line in CHECKSUMS_PATH.read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if name in {CSV_PATH.name, f"data/{CSV_PATH.name}"}:
            expected = digest
    assert expected, "the raw CSV is not listed in CHECKSUMS.txt"
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == expected


def test_no_target_in_features():
    assert TARGETS.isdisjoint(audit.FEATURES)
    assert not any(t in f for f in audit.FEATURES for t in TARGETS)
    assert set(audit.TARGETS) == TARGETS


def test_partition_metadata_is_not_a_feature():
    assert not any("group" in f.lower() for f in audit.FEATURES)


def test_product_type_is_not_numeric_feature():
    assert audit.CATEGORICAL == "ProductType"
    assert audit.CATEGORICAL not in audit.FEATURES


def test_fixed_seeds_are_42():
    pattern = re.compile(r"\b(?:random_state|seed|random_seed)\s*[=:]\s*(\d+)")
    bad = []
    for rel in _versionable_files():
        if not rel.endswith((".py", ".yaml", ".yml", ".json", ".sql")) or rel in _SKIP:
            continue
        text = _read_text(rel) or ""
        for m in pattern.finditer(text):
            if int(m.group(1)) != SEED:
                bad.append(f"{rel}: {m.group(0)}")
    assert not bad, f"seeds other than {SEED}: {bad}"


def test_versionable_files_do_not_mention_build_tooling():
    hits = []
    for rel in _versionable_files():
        if rel in _SKIP:
            continue
        text = _read_text(rel)
        if text is None:
            continue
        text = _PRODUCT_NAME_EXCEPTIONS.sub(" ", text)
        for m in _FORBIDDEN_RE.finditer(text):
            word = m.group(0).lower()
            if rel in _GENERIC_WORD_ALLOWLIST and word in _FORBIDDEN_WORDS:
                continue
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}: {m.group(0)!r}")
    assert not hits, "forbidden references in versionable files:\n" + "\n".join(hits)


def test_versionable_files_have_no_sensitive_data():
    hits = []
    for rel in _versionable_files():
        if rel in _SKIP:
            continue
        text = _read_text(rel)
        if text is None:
            continue
        for m in _SENSITIVE_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{rel}:{line}")
    assert not hits, "possible secrets or hosts in versionable files:\n" + "\n".join(hits)
