"""Filesystem layout and scope resolution.

Atlas keeps two stores. The global one holds knowledge that is true everywhere
(your tooling habits, how you like tests written); the project one holds
knowledge about a single repo. Recall reads both and ranks project first.
Writes pick exactly one.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_SCOPE = "project"
GLOBAL_SCOPE = "global"
SCOPES = (PROJECT_SCOPE, GLOBAL_SCOPE)

# A directory holding one of these is treated as a project root. `.atlas` wins
# over `.git` in the same directory so you can scope Atlas to a subtree of a
# monorepo by running `atlas init` there.
ROOT_MARKERS = (".atlas", ".git")

STORE_DIRNAME = ".atlas"
PROJECT_DB = "store.db"
GLOBAL_DB = "global.db"


def atlas_home() -> Path:
    """Directory holding the global store. Override with ATLAS_HOME."""
    env = os.environ.get("ATLAS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".atlas"


def global_db_path() -> Path:
    return atlas_home() / GLOBAL_DB


def project_db_path(root: Path) -> Path:
    return Path(root) / STORE_DIRNAME / PROJECT_DB


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a project marker.

    Returns None when called from somewhere that isn't a project, in which case
    Atlas operates on the global store alone.
    """
    env = os.environ.get("ATLAS_PROJECT")
    if env:
        root = Path(env).expanduser().resolve()
        return root if root.is_dir() else None

    here = Path(start or Path.cwd()).expanduser().resolve()
    for directory in (here, *here.parents):
        for marker in ROOT_MARKERS:
            if (directory / marker).exists():
                return directory
    return None


def project_name(root: Path | None) -> str:
    return root.name if root else "(no project)"
