"""Asset Bundle contract: variables, job order, environment and no account details."""

import tomllib

import yaml

from gcc_strength.config import ROOT

BUNDLE = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
JOB = BUNDLE["resources"]["jobs"]["gcc_end_to_end"]


def test_variables():
    assert BUNDLE["variables"]["catalog"]["default"] == "datathon"
    assert BUNDLE["variables"]["schema"]["default"] == "gcc"


def test_targets_have_no_host_or_profile():
    targets = BUNDLE["targets"]
    assert set(targets) == {"diego", "ricardo", "christopher"}
    assert targets["diego"].get("default") is True
    for t in targets.values():
        assert "workspace" not in t or not {"host", "profile"} & set(t["workspace"])
    assert "workspace" not in BUNDLE or not {"host", "profile"} & set(BUNDLE["workspace"])


def test_job_task_chain():
    keys = [t["task_key"] for t in JOB["tasks"]]
    assert keys == ["bootstrap", "bronze", "silver", "gold", "reconcile"]
    for prev, task in zip(JOB["tasks"], JOB["tasks"][1:], strict=False):
        assert task["depends_on"] == [{"task_key": prev["task_key"]}]


def test_notebooks_exist_and_use_env6():
    envs = {e["environment_key"]: e["spec"] for e in JOB["environments"]}
    assert all(spec["environment_version"] == "6" for spec in envs.values())
    for job in BUNDLE["resources"]["jobs"].values():
        for task in job["tasks"]:
            assert (ROOT / task["notebook_task"]["notebook_path"]).is_file()
            assert task["environment_key"] in {e["environment_key"] for e in job["environments"]}


def test_environment_pins_match_pyproject():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    for env in JOB["environments"]:
        for dep in env["spec"].get("dependencies", []):
            assert dep in deps, dep


def test_reset_job_is_guarded():
    reset = BUNDLE["resources"]["jobs"]["gcc_reset_schema"]
    params = {p["name"]: p["default"] for p in reset["parameters"]}
    assert params["confirm_reset"] == ""
