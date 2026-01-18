"""Default tool definitions for voice command evaluation."""

from typing import Any, Dict, List

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


def get_default_tools() -> List[Dict[str, Any]]:
    """
    Get default tool definitions for common voice commands.

    Returns a copy to prevent accidental modification.

    Returns:
        List of tool definitions in OpenAI format
    """
    import copy
    return copy.deepcopy(DEFAULT_VOICE_COMMAND_TOOLS)
