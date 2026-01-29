"""Default tool definitions for voice command evaluation."""

import copy
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
    return copy.deepcopy(DEFAULT_VOICE_COMMAND_TOOLS)


# Anki MCP tool definitions for large tool set testing
ANKI_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "sync",
            "description": "Synchronizes local Anki collections with AnkiWeb. Should be called at the START of a review session (before getting cards) and at the END when user indicates they are done.",
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
            "name": "get_due_cards",
            "description": "Retrieve cards that are due for review from Anki. IMPORTANT: Use sync tool FIRST before getting cards to ensure latest data. After getting cards, use present_card to show them one by one to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deck_name": {
                        "type": "string",
                        "description": "Specific deck name to get cards from. If not specified, gets cards from all decks"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of cards to return",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "present_card",
            "description": "Retrieve a card's content for review. WORKFLOW: 1) Show question, 2) Wait for user answer, 3) Show answer with show_answer=true, 4) Evaluate and suggest rating (1-4), 5) Wait for user confirmation ('ok'/'next' = accept, or they provide different rating), 6) Only then use rate_card",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "number",
                        "description": "The ID of the card to retrieve"
                    },
                    "show_answer": {
                        "type": "boolean",
                        "description": "Whether to include the answer/back content in the response",
                        "default": False
                    }
                },
                "required": ["card_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rate_card",
            "description": "Submit a rating for a card to update Anki's spaced repetition scheduling. Use this ONLY after the user confirms or modifies your suggested rating. Do not rate automatically without user input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "number",
                        "description": "The identifier for the card being rated"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Rating value: 1 = Again (failed), 2 = Hard, 3 = Good, 4 = Easy",
                        "minimum": 1,
                        "maximum": 4
                    }
                },
                "required": ["card_id", "rating"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_decks",
            "description": "List all available Anki decks, optionally with statistics. Remember to sync first at the start of a review session for latest data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "Include card count statistics for each deck",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_deck",
            "description": "Creates a new empty Anki deck with support for hierarchical naming using parent::child structure (maximum 2 levels). The tool will not overwrite existing decks. Only creates empty decks - should not add cards unless explicitly requested by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deck_name": {
                        "type": "string",
                        "description": "The deck name, supporting '::' notation for nested structures (e.g., 'Japanese' or 'Japanese::Tokyo'). Maximum 2 nesting levels allowed",
                        "minLength": 1
                    }
                },
                "required": ["deck_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "addNote",
            "description": "Add a new note to Anki. Use modelNames to see available note types and modelFieldNames to see required fields. Returns the note ID on success. IMPORTANT: Only create notes that were explicitly requested by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deckName": {
                        "type": "string",
                        "description": "The deck where the note will be stored",
                        "minLength": 1
                    },
                    "modelName": {
                        "type": "string",
                        "description": "The note type/model to use (e.g., 'Basic', 'Cloze')",
                        "minLength": 1
                    },
                    "fields": {
                        "type": "object",
                        "description": "Field values as key-value pairs (e.g., {'Front': 'question', 'Back': 'answer'})",
                        "additionalProperties": {
                            "type": "string"
                        }
                    },
                    "tags": {
                        "type": "array",
                        "description": "Optional tags for organizing the note",
                        "items": {
                            "type": "string"
                        }
                    },
                    "allowDuplicate": {
                        "type": "boolean",
                        "description": "Whether to permit duplicate notes",
                        "default": False
                    },
                    "duplicateScope": {
                        "type": "string",
                        "description": "Scope for duplicate detection",
                        "enum": ["deck", "collection"]
                    },
                    "duplicateScopeOptions": {
                        "type": "object",
                        "description": "Advanced duplicate checking settings including specific deck name, child deck checking, and cross-model checking"
                    }
                },
                "required": ["deckName", "modelName", "fields"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "findNotes",
            "description": "Search for notes using Anki query syntax. Use queries like 'deck:DeckName', 'tag:tagname', 'is:due', 'is:new', 'is:review', 'front:text', 'back:text', or combine with spaces for AND, OR for alternatives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Anki search query using Anki query syntax",
                        "minLength": 1
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notesInfo",
            "description": "Retrieves detailed information about specific notes including all fields, tags, model info, and CSS styling. This tool should be used after findNotes to obtain complete note data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "description": "Array of note IDs to get information for (max 100 at once for performance). Get these IDs from findNotes tool",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 1,
                        "maxItems": 100
                    }
                },
                "required": ["notes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "updateNoteFields",
            "description": "Updates existing note fields with support for HTML content and CSS preservation. Key warnings: avoid note viewing during updates and only modify notes the user explicitly requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "object",
                        "description": "Note object containing id and fields to update",
                        "properties": {
                            "id": {
                                "type": "number",
                                "description": "The note's unique identifier, obtainable via findNotes or notesInfo"
                            },
                            "fields": {
                                "type": "object",
                                "description": "Key-value pairs representing fields to modify. Only changed fields needed. Accepts HTML formatting",
                                "additionalProperties": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["id", "fields"]
                    }
                },
                "required": ["note"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deleteNotes",
            "description": "Delete notes by their IDs. This will permanently remove the notes and ALL associated cards. This action cannot be undone unless you have a backup. CRITICAL: This is destructive and permanent - only delete notes the user explicitly confirmed for deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "array",
                        "description": "Array of note IDs to delete (max 100 at once for safety). Get these IDs from findNotes tool. ALL cards associated with these notes will be deleted",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 1,
                        "maxItems": 100
                    },
                    "confirmDeletion": {
                        "type": "boolean",
                        "description": "Must be set to true to confirm you want to permanently delete these notes and their cards"
                    }
                },
                "required": ["notes", "confirmDeletion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modelNames",
            "description": "Get a list of all available note type (model) names in Anki. Use this to see what note types are available before creating notes.",
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
            "name": "modelFieldNames",
            "description": "Get the field names for a specific note type (model). Use this to know what fields are required when creating notes of this type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modelName": {
                        "type": "string",
                        "description": "The name of the model/note type to get fields for",
                        "minLength": 1
                    }
                },
                "required": ["modelName"]
            }
        }
    }
]


def get_anki_tools() -> List[Dict[str, Any]]:
    """
    Get Anki MCP tool definitions for large tool set testing.

    Returns a copy to prevent accidental modification.

    Returns:
        List of 13 Anki tool definitions in OpenAI format
    """
    return copy.deepcopy(ANKI_TOOLS)
