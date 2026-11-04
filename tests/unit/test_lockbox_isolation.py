"""AST guard against adding lockbox readers to training or analysis modules."""

import ast
import re
from pathlib import Path

import pytest

from gcc_strength.config import ROOT

ALLOWED = {"lockbox.py", "split.py", "gold.py", "dq.py"}


def violations(source):
    tree = ast.parse(source)
    failures = []
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update({n.asname or n.name: n.name for n in node.names})
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if aliases.get(name, name) in {
                "evaluate_lockbox",
                "load_lockbox",
                "replay_lockbox",
                "_read_replay_gold",
            }:
                failures.append((node.lineno, "lockbox call"))
        # Inspect comparisons, pandas filter calls, reader predicates and subscripts.
        if isinstance(node, (ast.Compare, ast.Call, ast.Subscript, ast.Assign)):
            for child in ast.walk(node):
                if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                    continue
                if child.value == "test":
                    # Descriptive report dict keys named test are not partition filters.
                    is_dict_key = any(
                        isinstance(parent, ast.Dict) and child in parent.keys
                        for parent in ast.walk(node)
                    )
                    if not is_dict_key:
                        failures.append((child.lineno, "test partition literal"))
                elif re.search(
                    r"(?:split\s*(?:==|!=|=)|\bsplit\s+in).*['\"]test['\"]", child.value
                ):
                    failures.append((child.lineno, "query filter"))
    return failures


def test_modules_do_not_read_or_filter_lockbox():
    for path in sorted((ROOT / "src/gcc_strength").rglob("*.py")):
        if path.name not in ALLOWED:
            assert not violations(path.read_text()), str(path.relative_to(ROOT))


@pytest.mark.parametrize(
    "source",
    [
        'df.loc[df["split"] == "test"]',
        'df[df.split.eq("test")]',
        'pd.read_parquet(path, filters=[("split", "==", "test")])',
        "df.query(\"split == 'test'\")",
        "evaluate_lockbox()",
        "module.load_lockbox()",
        "replay_lockbox()",
        "module._read_replay_gold()",
        "spark.table(name).where(\"split = 'test'\").toPandas()",
        "from gcc_strength.lockbox import evaluate_lockbox as evaluate\nevaluate()",
    ],
)
def test_detector_rejects_forbidden_patterns(source):
    assert violations(source)


def test_detector_allows_train_and_report_keys():
    assert not violations('part = df.loc[df.split.eq("train")]\nreport = {"test": name}')


def test_official_lockbox_not_part_of_bundle():
    assert "07_lockbox" not in (ROOT / "databricks.yml").read_text()
    assert Path(ROOT / "notebooks/07_lockbox.py").is_file()
    assert "07_evaluate.py" in (ROOT / "databricks.yml").read_text()
