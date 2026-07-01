"""
Tests for the PBDS activation / triggering logic (configurable path + silent skip).

Verifies that:
  * a missing workbook -> the tool stays off (load_pbds_manager returns None), no raise;
  * the path activates the tool when a readable file exists;
  * the path is overloadable three ways, mirroring the rest of the config:
      - an explicit workbook_path argument,
      - configure(overrides={"pbds": {"workbook_path": ...}}),
      - the QMIX_REPORT_PBDS_WORKBOOK environment variable.

Uses the sample workbook at the repo root as a "file that exists"; it does not
build the graph, so no Ollama and no valid formulas are required. Each test
resets the cached config and the env var so nothing leaks into other tests.

Run:  .venv/Scripts/python.exe tests/test_pbds_activation.py
"""

import os
import sys

sys.path.insert(0, ".")

from qmix_report_writer.utils import config as cfg
from qmix_report_writer.tools.pbds import PBDSManager, load_pbds_manager, pbds_available

SAMPLE_XLSX = "parameter names and descriptions.xlsx"   # exists at the repo root
ABSENT_XLSX = "definitely_absent_pbds_workbook.xlsx"    # must NOT exist


def _reset_config():
    cfg._config = None
    os.environ.pop(cfg.ENV_PBDS_WORKBOOK, None)


def test_missing_workbook_is_silent_skip():
    _reset_config()
    try:
        cfg.configure(overrides={"pbds": {"workbook_path": ABSENT_XLSX}})
        assert pbds_available() is False
        assert load_pbds_manager() is None            # no exception, just off
    finally:
        _reset_config()
    print("PASS  test_missing_workbook_is_silent_skip")


def test_explicit_path_argument():
    _reset_config()
    try:
        assert load_pbds_manager(workbook_path=ABSENT_XLSX) is None
        mgr = load_pbds_manager(workbook_path=SAMPLE_XLSX)
        assert isinstance(mgr, PBDSManager), mgr
        assert os.path.samefile(mgr.workbook_path, SAMPLE_XLSX)
    finally:
        _reset_config()
    print("PASS  test_explicit_path_argument")


def test_configure_override_activates():
    _reset_config()
    try:
        cfg.configure(overrides={"pbds": {"workbook_path": SAMPLE_XLSX}})
        assert pbds_available() is True
        mgr = load_pbds_manager()
        assert isinstance(mgr, PBDSManager)
        assert os.path.samefile(mgr.workbook_path, SAMPLE_XLSX)
    finally:
        _reset_config()
    print("PASS  test_configure_override_activates")


def test_env_var_override_activates():
    _reset_config()
    try:
        # Config leaves workbook_path unset, so the env var takes effect.
        os.environ[cfg.ENV_PBDS_WORKBOOK] = SAMPLE_XLSX
        cfg.configure()
        assert pbds_available() is True
        assert isinstance(load_pbds_manager(), PBDSManager)
    finally:
        _reset_config()
    print("PASS  test_env_var_override_activates")


def test_default_is_off_when_no_file_present():
    _reset_config()
    try:
        # Clear any configured path so we exercise the default-filename fallback
        # ('pbds_parameters.xlsx'), which is absent at the repo root. (default.yaml
        # may point workbook_path at a real puppet file, so null it explicitly.)
        cfg.configure(overrides={"pbds": {"workbook_path": None}})
        assert pbds_available() is False
        assert load_pbds_manager() is None
    finally:
        _reset_config()
    print("PASS  test_default_is_off_when_no_file_present")


def _run_all():
    tests = [
        test_missing_workbook_is_silent_skip,
        test_explicit_path_argument,
        test_configure_override_activates,
        test_env_var_override_activates,
        test_default_is_off_when_no_file_present,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {test.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())
