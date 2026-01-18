"""Environment variable utilities."""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file. If None, uses .env in current directory.
    """
    if env_path is None:
        env_path = Path.cwd() / ".env"

    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


def get_env_var(
    key: str,
    default: Optional[str] = None,
    load_dotenv: bool = True,
) -> Optional[str]:
    """
    Get an environment variable, optionally loading from .env first.

    Args:
        key: Environment variable name
        default: Default value if not found
        load_dotenv: Whether to load from .env if not already set

    Returns:
        The environment variable value or default
    """
    value = os.getenv(key)
    if value is not None:
        return value

    if load_dotenv:
        load_env_file()
        value = os.getenv(key)
        if value is not None:
            return value

    return default
