"""Load credentials from a .env file into the environment.

Every entry point calls this at startup so `.env` actually does something --
without it the API-key error message would be telling you to create a file
nothing reads.

Existing environment variables always win, so an exported key beats the file.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path | None = None) -> bool:
    """Returns True if a .env file was found and read."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False

    try:                                  # prefer python-dotenv when installed
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return True
    except ImportError:
        pass

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True
