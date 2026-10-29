"""Settings overrides: CLI parsing, type coercion, hand-down to worker processes. No data needed."""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def run_py(code, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "TANUKI_CONFIG_JSON"}
    env["PYTHONPATH"] = SRC
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_defaults_have_no_personal_paths(tmp_path):
    line = run_py("from tanuki import config as c; print(c.MASTER_DIR.name, c.CSV_PATH.name, c.BONSAI_REPO)")
    assert line == "data BIGmatched_peaks_raw.csv None"


def test_overrides_coerce_types_and_derive(tmp_path):
    code = ("from tanuki import config as c\n"
            f"c.configure(MASTER_DIR=r'{tmp_path}', USE_MORPHOLOGY_FEATURES=False, safe_ram_workers=1,\n"
            "            STAGE2_AUDIT_CHANNEL_COLORS={'0': [1, 1, 1]}, CELLPOSE_STRETCH_PERCENTILES=[1, 99])\n"
            "print(c.CSV_PATH.parent == c.MASTER_DIR, c.FEATURE_TAG, c.MAX_WORKERS_STAGE1,\n"
            "      c.STAGE2_AUDIT_CHANNEL_COLORS, c.CELLPOSE_STRETCH_PERCENTILES)")
    assert run_py(code) == "True mass_only 1 {0: (1, 1, 1)} (1, 99)"


def test_unknown_setting_is_an_error():
    code = ("from tanuki import config as c\n"
            "try:\n    c.configure(NOT_A_SETTING=1)\nexcept KeyError:\n    print('raised')")
    assert run_py(code) == "raised"


def test_worker_processes_inherit_overrides(tmp_path):
    """what a spawned stage 1 / stage 3 worker sees: a fresh import with TANUKI_CONFIG_JSON set"""
    env = {"TANUKI_CONFIG_JSON": json.dumps({"MASTER_DIR": str(tmp_path), "N_CLUSTERS": 9})}
    line = run_py("from tanuki import config as c; print(c.MASTER_DIR == __import__('pathlib').Path(r'%s'), "
                  "c.N_CLUSTERS)" % tmp_path, env)
    assert line == "True 9"


def test_cli_show_config(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"N_CLUSTERS": 5, "COMBAT_ENABLED": True}))
    env = {k: v for k, v in os.environ.items() if k != "TANUKI_CONFIG_JSON"}
    env["PYTHONPATH"] = SRC
    out = subprocess.run([sys.executable, "-m", "tanuki", "--input_dir", str(tmp_path), "--config",
                          str(tmp_path / "s.json"), "--set", "N_CLUSTERS=6", "--show-config"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "N_CLUSTERS = 6" in out and "COMBAT_ENABLED = True" in out
