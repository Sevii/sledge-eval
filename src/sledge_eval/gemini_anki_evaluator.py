"""Anki Large Tool Set Evaluator for Google Gemini API."""

from typing import Any, Dict, List, Optional

from .gemini_evaluator import GeminiEvaluator


class GeminiAnkiEvaluator(GeminiEvaluator):
    """Evaluator that tests Gemini model performance with large tool sets using Anki MCP tools."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the Gemini Anki large tool set evaluator.

        Args:
            api_key: Google API key. If not provided, reads from GEMINI_API_KEY or APIKey env var
            model: Gemini model to use (default: gemini-2.5-flash-lite)
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        # Define the comprehensive Anki tool set (13 tools)
        anki_tools = [
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

        super().__init__(
            api_key=api_key,
            model=model,
            available_tools=anki_tools,
            timeout=timeout,
            debug=debug,
        )

    def get_tool_count(self) -> int:
        """Get the number of available tools."""
        return len(self.available_tools)

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return [tool["function"]["name"] for tool in self.available_tools]
