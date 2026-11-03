"""Project paths and YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf"
DATA_DIR = ROOT / "data"
MEDALLION_CONF = CONF_DIR / "medallion.yaml"
LOCKBOX_FILE = CONF_DIR / "lockbox.sha256"
CHECKSUMS_FILE = DATA_DIR / "CHECKSUMS.txt"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_medallion_conf(path: Path | str = MEDALLION_CONF) -> dict[str, Any]:
    """Medallion configuration (split parameters and expected counts)."""
    return load_yaml(path)
