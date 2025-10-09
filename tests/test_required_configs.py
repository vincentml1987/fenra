import os

from config.checks import check_required_configs
from config.required_configs import REQUIRED_CONFIGS


def test_check_required_configs_identifies_missing(tmp_path):
    conf_dir = tmp_path / "confs"
    conf_dir.mkdir()
    for name in REQUIRED_CONFIGS[:-1]:
        (conf_dir / name).write_text("{}", encoding="utf-8")

    all_present, missing = check_required_configs(str(conf_dir))

    assert not all_present
    assert missing == [REQUIRED_CONFIGS[-1]]


def test_check_required_configs_all_present(tmp_path):
    conf_dir = tmp_path / "confs"
    conf_dir.mkdir()
    for name in REQUIRED_CONFIGS:
        (conf_dir / name).write_text("{}", encoding="utf-8")

    all_present, missing = check_required_configs(str(conf_dir))

    assert all_present
    assert missing == []


def test_check_required_configs_missing_directory(tmp_path):
    conf_dir = tmp_path / "missing"

    all_present, missing = check_required_configs(str(conf_dir))

    assert not all_present
    assert missing == list(REQUIRED_CONFIGS)
