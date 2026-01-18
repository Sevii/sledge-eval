"""Centralized configuration for sledge-eval."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Configuration for llama-server connection."""

    url: str = Field(default="http://localhost:8080", description="Server URL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    debug: bool = Field(default=False, description="Enable debug logging")


class GeminiConfig(BaseModel):
    """Configuration for Gemini API connection."""

    model: str = Field(default="gemini-2.5-flash-lite", description="Gemini model to use")
    api_key: Optional[str] = Field(default=None, description="API key (defaults to env var)")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    debug: bool = Field(default=False, description="Enable debug logging")

    def get_api_key(self) -> Optional[str]:
        """Get API key from config, environment variables, or .env file."""
        if self.api_key:
            return self.api_key

        # Try environment variables
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("APIKey")
        if api_key:
            return api_key

        # Try .env file
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "APIKey":
                            return value.strip()

        return None


class TestSuiteConfig(BaseModel):
    """Configuration for test suite locations."""

    base_path: Path = Field(default=Path("tests/test_data"), description="Base path for test data")
    example_suite: str = Field(default="example_test_suite.json", description="Example test suite filename")
    anki_suite: str = Field(default="anki_large_toolset_suite.json", description="Anki test suite filename")
    text_suite: str = Field(default="comprehensive_text_suite.json", description="Text evaluation suite filename")

    @property
    def example_suite_path(self) -> Path:
        """Get full path to example test suite."""
        return self.base_path / self.example_suite

    @property
    def anki_suite_path(self) -> Path:
        """Get full path to Anki test suite."""
        return self.base_path / self.anki_suite

    @property
    def text_suite_path(self) -> Path:
        """Get full path to text evaluation test suite."""
        return self.base_path / self.text_suite


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    base_path: Path = Field(default=Path.cwd(), description="Base path for reports")
    reports_dir: str = Field(default="reports", description="Reports directory name")
    include_hardware_info: bool = Field(default=True, description="Include hardware info in reports")

    @property
    def reports_path(self) -> Path:
        """Get full path to reports directory."""
        return self.base_path / self.reports_dir


class EvalConfig(BaseModel):
    """Main configuration for sledge-eval."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    test_suites: TestSuiteConfig = Field(default_factory=TestSuiteConfig)
    reports: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_env(cls) -> "EvalConfig":
        """Create configuration from environment variables."""
        return cls(
            server=ServerConfig(
                url=os.getenv("SLEDGE_SERVER_URL", "http://localhost:8080"),
                timeout=int(os.getenv("SLEDGE_TIMEOUT", "120")),
                debug=os.getenv("SLEDGE_DEBUG", "").lower() == "true",
            ),
            gemini=GeminiConfig(
                model=os.getenv("SLEDGE_GEMINI_MODEL", "gemini-2.5-flash-lite"),
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("APIKey"),
                timeout=int(os.getenv("SLEDGE_TIMEOUT", "120")),
                debug=os.getenv("SLEDGE_DEBUG", "").lower() == "true",
            ),
        )


# Default tool definitions for voice command evaluation
DEFAULT_VOICE_COMMAND_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "control_lights",
            "description": "Control smart lights in a specific room",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "The room where the lights are located",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "dim", "brighten"],
                        "description": "The action to perform on the lights",
                    },
                },
                "required": ["room", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_temperature",
            "description": "Set the thermostat to a specific temperature",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "The target temperature",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["temperature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Play music from a specific playlist or artist",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist": {
                        "type": "string",
                        "description": "Name of the playlist to play",
                    },
                    "artist": {
                        "type": "string",
                        "description": "Name of the artist",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_volume",
            "description": "Adjust the volume level",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["increase", "decrease", "mute", "unmute"],
                        "description": "Volume adjustment action",
                    },
                    "level": {
                        "type": "number",
                        "description": "Specific volume level (0-100)",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["now", "today", "tomorrow", "week"],
                        "description": "Time period for weather",
                    },
                },
            },
        },
    },
]
