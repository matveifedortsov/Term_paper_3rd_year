"""Configuration loader for the term-paper pipeline.

Single point of truth: config/default.yaml at the repo root. Modules
that need a constant call:

    from src.config import config
    cfg = config()
    block_c = cfg["lomn_real"]["block_constant"]

The result is a plain dict-of-dicts (structurally compatible with
JSON), with attribute-style access enabled by Box() if needed.

Override path with TERMPAPER_CONFIG env var or by passing path=...
to load_config().
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config; defaults to config/default.yaml at repo root.

    Environment variable TERMPAPER_CONFIG, if set, takes precedence over
    the default path.
    """
    if path is None:
        env_path = os.environ.get("TERMPAPER_CONFIG")
        path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    """Cached default config — first call reads YAML, subsequent are O(1)."""
    return load_config()


def reload_config() -> dict[str, Any]:
    """Force a re-read (for tests / interactive use after editing YAML)."""
    config.cache_clear()
    return config()
