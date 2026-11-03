"""Data dictionary, layer tags and SQL builders for Unity Catalog."""

import pandas as pd
import pytest

from gcc_strength import catalog as uc
from gcc_strength.spark_io import spark_ddl


def _frames(m):
    return {
        "bronze": m.bronze,
        "silver": m.silver,
        "quarantine": m.quarantine,
        "gold": m.gold,
        "dq": m.dq_results,
    }


def test_every_column_has_a_comment(medallion):
    for key, df in _frames(medallion).items():
        for col in df.columns:
            assert uc.column_comment(key, col)


def test_undocumented_column_raises():
    with pytest.raises(KeyError):
        uc.column_comment("gold", "mystery_feature")


def test_governance_statements(medallion):
    stmts = uc.governance_statements("datathon", "gcc", "gold", medallion.gold.columns)
    assert stmts[0].startswith("COMMENT ON TABLE `datathon`.`gcc`.`datathon_data_gold`")
    assert stmts[-1].endswith("SET TAGS ('layer' = 'gold')")
    assert len(stmts) == len(medallion.gold.columns) + 2
    assert uc.LAYER_TAG == {
        "bronze": "bronze",
        "silver": "silver",
        "quarantine": "silver",
        "gold": "gold",
        "dq": "dq",
    }


def test_sql_escaping_and_identifiers():
    assert uc.sql_string("it's") == "'it\\'s'"
    with pytest.raises(ValueError):
        uc.quote_ident("gcc; DROP TABLE x")
    assert uc.volume_path("datathon", "gcc") == "/Volumes/datathon/gcc/raw"


def test_bootstrap_is_idempotent_sql():
    stmts = uc.bootstrap_statements("datathon", "gcc")
    assert all("IF NOT EXISTS" in s for s in stmts)
    assert [s.split()[1] for s in stmts] == ["CATALOG", "SCHEMA", "VOLUME"]


def test_reset_requires_explicit_confirmation():
    with pytest.raises(PermissionError):
        uc.reset_statements("datathon", "gcc", ["t"], confirm="")
    with pytest.raises(PermissionError):
        uc.reset_statements("datathon", "gcc", ["t"], confirm="yes")
    stmts = uc.reset_statements("datathon", "gcc", ["b", "a"], confirm="datathon.gcc")
    assert stmts == [
        "DROP TABLE IF EXISTS `datathon`.`gcc`.`a`",
        "DROP TABLE IF EXISTS `datathon`.`gcc`.`b`",
    ]
    schema = uc.reset_statements("datathon", "gcc", [], "datathon.gcc", mode="schema")
    assert schema == ["DROP SCHEMA IF EXISTS `datathon`.`gcc` CASCADE"]


def test_spark_ddl(medallion):
    ddl = spark_ddl(medallion.gold)
    assert "`_row_id` BIGINT" in ddl and "`split` STRING" in ddl
    assert "`Strength28` DOUBLE" in ddl and "`flag_r1_suspect` BOOLEAN" in ddl
    assert "`_ingest_ts` TIMESTAMP" in spark_ddl(medallion.bronze)
    assert spark_ddl(pd.DataFrame({"a": pd.Series([], dtype="float64")})) == "`a` DOUBLE"
