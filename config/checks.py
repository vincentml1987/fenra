"""Utilities for validating required configuration files."""
from __future__ import annotations

import os
from typing import List, Tuple

from .required_configs import CONF_DIR, REQUIRED_CONFIGS


def check_required_configs(conf_dir: str = CONF_DIR) -> Tuple[bool, List[str]]:
    """Return whether all required config files exist and a list of missing ones."""

    missing: List[str] = []
    for name in REQUIRED_CONFIGS:
        path = os.path.join(conf_dir, name)
        if not os.path.isfile(path):
            missing.append(name)
    return len(missing) == 0, missing
