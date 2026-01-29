"""API key resolution utilities."""

import os
from pathlib import Path
from typing import Optional


def resolve_api_key(key_name: str, provided: Optional[str] = None) -> Optional[str]:
    """Resolve an API key from multiple sources.

    Resolution order:
    1. Provided value (if not None)
    2. Environment variable
    3. .env file in current directory

    Args:
        key_name: Name of the environment variable (e.g., "OPENROUTER_API_KEY")
        provided: Explicitly provided API key value

    Returns:
        The resolved API key, or None if not found
    """
    # 1. Use provided value if given
    if provided:
        return provided

    # 2. Try environment variable
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    # 3. Try .env file
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == key_name:
                        return value.strip()

    return None
