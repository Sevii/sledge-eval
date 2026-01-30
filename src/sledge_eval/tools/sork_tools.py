"""Sork game tool definitions for voice command evaluation."""

import copy
from typing import Any, Dict, List

# Sork game tool definitions extracted from gameplay logs
SORK_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "go_north",
            "description": "Move to the room to the north. Use when player says 'go north', 'head north', 'north', 'walk north', or 'move north'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_south",
            "description": "Move to the room to the south. Use when player says 'go south', 'head south', 'south', 'walk south', or 'move south'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_east",
            "description": "Move to the room to the east. Use when player says 'go east', 'head east', 'east', 'walk east', or 'move east'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_west",
            "description": "Move to the room to the west. Use when player says 'go west', 'head west', 'west', 'walk west', or 'move west'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_up",
            "description": "Move up to the room above. Use when player says 'go up', 'climb up', 'up', 'ascend', or 'climb ladder'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_down",
            "description": "Move down to the room below. Use when player says 'go down', 'climb down', 'down', 'descend', or 'go downstairs'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take",
            "description": "Pick up an item from the current room and add it to inventory. Use when player says 'take', 'pick up', 'grab', 'get', or 'collect' followed by an item name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The name of the item to pick up (e.g., 'crowbar', 'keycard', 'medkit', 'flashlight')."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drop",
            "description": "Drop an item from inventory into the current room. Use when player says 'drop', 'put down', 'leave', or 'discard' followed by an item name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The name of the item to drop from inventory."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "use",
            "description": "Use an item from inventory, optionally on a target. Use when player says 'use', 'activate', 'operate', or 'apply' followed by an item and optionally a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The name of the item to use."
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target to use the item on."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "examine",
            "description": "Look closely at the room, an item, object, or creature to get more details. Use when player says 'examine', 'look at', 'inspect', 'check', 'study', or 'describe' followed by a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "What to examine - can be 'room'/'surroundings' for the current location, or an item, object, or creature."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search an object in the room for hidden items. Use when player says 'search', 'look in', 'look inside', 'check inside', or 'rummage through'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The object to search (e.g., 'locker', 'desk', 'crate', 'cabinet')."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Look around and describe the current room. Use when player says 'look', 'look around', 'where am I', 'describe room', or 'what do I see'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inventory",
            "description": "List all items the player is carrying. Use when player says 'inventory', 'what am I carrying', 'show items', 'check inventory', or 'my items'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "attack",
            "description": "Attack a target creature or enemy. Use when player says 'attack', 'fight', 'hit', 'strike', or 'kill' followed by a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The creature or enemy to attack."
                    },
                    "weapon": {
                        "type": "string",
                        "description": "Optional weapon to use for the attack."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flee",
            "description": "Run away from combat or danger. Use when player says 'flee', 'run', 'escape', 'run away', or 'retreat'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "help",
            "description": "Show available commands and help information. Use when player explicitly asks for help by saying 'help', 'what can I do', 'commands', or 'how do I play'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def get_sork_tools() -> List[Dict[str, Any]]:
    """
    Get Sork game tool definitions for voice command evaluation.

    Returns a copy to prevent accidental modification.

    Returns:
        List of 16 Sork tool definitions in OpenAI format
    """
    return copy.deepcopy(SORK_TOOLS)
