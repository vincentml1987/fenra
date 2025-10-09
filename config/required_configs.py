"""Centralized list of configuration files required for running Fenra."""
from __future__ import annotations

import os
from typing import Final, Iterable, List, Tuple

CONF_DIR: Final[str] = "confs"

REQUIRED_CONFIGS: Final[Tuple[str, ...]] = (
    "globals.json",
    "pdvs.json",
    "agent_classes.json",
    "agents.json",
)


def iter_required_paths(conf_dir: str = CONF_DIR) -> Iterable[str]:
    """Yield absolute paths for each required configuration file."""

    base = os.path.abspath(conf_dir)
    for name in REQUIRED_CONFIGS:
        yield os.path.join(base, name)


def required_config_names() -> List[str]:
    """Return the ordered list of required configuration filenames."""

    return list(REQUIRED_CONFIGS)
