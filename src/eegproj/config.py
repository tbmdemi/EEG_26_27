"""Locate the repository and load the shared configuration (configs/config.yaml)."""
from pathlib import Path

import yaml

CONFIG_RELPATH = Path('configs/config.yaml')


def find_root(start=None):
    """Return the first directory at or above `start` (default: cwd) containing configs/config.yaml."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / CONFIG_RELPATH).exists():
            return candidate
    raise FileNotFoundError(f'{CONFIG_RELPATH} not found at or above {start}')


def load_config(root):
    """Load configs/config.yaml from the repository root."""
    return yaml.safe_load((Path(root) / CONFIG_RELPATH).read_text(encoding='utf-8'))
