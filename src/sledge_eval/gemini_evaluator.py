"""Gemini API evaluator for Google's Gemini models."""

import json
import os
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .evaluator import EvaluationResult, Evaluator, ToolCall, VoiceCommandTest
from .text_evaluator import TextEvaluator


class GeminiEvaluator(Evaluator):
    """Evaluator that uses Google Gemini API for tool calling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        available_tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the Gemini evaluator.

        Args:
            api_key: Google API key. If not provided, reads from GEMINI_API_KEY or APIKey env var
            model: Gemini model to use (default: gemini-2.5-flash-lite-preview-06-17)
            available_tools: List of tool definitions in OpenAI format (will be converted)
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        # Get API key from parameter, env vars, or .env file
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("APIKey")
        if not self.api_key:
            # Try to load from .env file
            self._load_env_file()
            self.api_key = os.getenv("APIKey")

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set via api_key parameter, "
                "GEMINI_API_KEY env var, or APIKey in .env file"
            )

        self.model = model
        self.available_tools = available_tools or self._get_default_tools()
        self.timeout = timeout
        self.debug = debug

        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)

        super().__init__(model_client=self.client)

    def _load_env_file(self):
        """Load environment variables from .env file if present."""
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

    def _get_default_tools(self) -> List[Dict[str, Any]]:
        """
        Get default tool definitions for common voice commands.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [
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

    def _convert_to_gemini_tools(self, openai_tools: List[Dict[str, Any]]) -> types.Tool:
        """
        Convert OpenAI-format tool definitions to Gemini format.

        Args:
            openai_tools: List of tool definitions in OpenAI format

        Returns:
            Gemini Tool object with function declarations
        """
        function_declarations = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            params = func.get("parameters", {})

            # Convert properties to Gemini Schema format
            gemini_properties = {}
            for prop_name, prop_def in params.get("properties", {}).items():
                prop_type = prop_def.get("type", "STRING").upper()
                if prop_type == "NUMBER":
                    prop_type = "NUMBER"
                elif prop_type == "INTEGER":
                    prop_type = "INTEGER"
                elif prop_type == "BOOLEAN":
                    prop_type = "BOOLEAN"
                elif prop_type == "ARRAY":
                    prop_type = "ARRAY"
                elif prop_type == "OBJECT":
                    prop_type = "OBJECT"
                else:
                    prop_type = "STRING"

                schema_kwargs = {
                    "type": prop_type,
                    "description": prop_def.get("description", ""),
                }

                # Handle enum values
                if "enum" in prop_def:
                    schema_kwargs["enum"] = prop_def["enum"]

                # Handle items for array types
                if prop_type == "ARRAY" and "items" in prop_def:
                    items_def = prop_def["items"]
                    items_type = items_def.get("type", "STRING").upper()
                    schema_kwargs["items"] = types.Schema(type=items_type)

                gemini_properties[prop_name] = types.Schema(**schema_kwargs)

            # Build parameter schema
            param_schema = types.Schema(
                type="OBJECT",
                properties=gemini_properties,
                required=params.get("required", []),
            )

            # Create function declaration
            func_decl = types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=param_schema,
            )
            function_declarations.append(func_decl)

        return types.Tool(function_declarations=function_declarations)

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """
        Evaluate a single test case using Gemini API.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        start_time = time.time()

        try:
            # Convert tools to Gemini format
            gemini_tools = self._convert_to_gemini_tools(self.available_tools)

            # Build the prompt
            system_prompt = "You are a helpful assistant that interprets voice commands and calls appropriate functions."
            user_message = test.voice_command

            if self.debug:
                print("\n" + "=" * 60)
                print("DEBUG: REQUEST TO GEMINI API")
                print("=" * 60)
                print(f"Model: {self.model}")
                print(f"System: {system_prompt}")
                print(f"User: {user_message}")
                print(f"Tools: {len(self.available_tools)} function(s)")
                print("=" * 60)

            # Create the request with tools
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"{system_prompt}\n\nUser command: {user_message}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    tools=[gemini_tools],
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )

            if self.debug:
                print("\n" + "=" * 60)
                print("DEBUG: RESPONSE FROM GEMINI API")
                print("=" * 60)
                print(f"Response: {response}")
                print("=" * 60 + "\n")

            # Parse tool calls from response
            predicted_tool_calls = []

            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                func_call = part.function_call
                                # Convert args to dict
                                args = {}
                                if func_call.args:
                                    # func_call.args is a dict-like object
                                    args = dict(func_call.args)

                                predicted_tool_calls.append(
                                    ToolCall(name=func_call.name, arguments=args)
                                )

            # Compare predicted vs expected tool calls
            passed = self._compare_tool_calls(
                predicted_tool_calls, test.expected_tool_calls
            )

            # Calculate evaluation time
            evaluation_time_ms = (time.time() - start_time) * 1000

            return EvaluationResult(
                test_id=test.id,
                passed=passed,
                predicted_tool_calls=predicted_tool_calls,
                expected_tool_calls=test.expected_tool_calls,
                evaluation_time_ms=evaluation_time_ms,
                voice_command=test.voice_command,
                test_description=test.description,
                tags=test.tags,
            )

        except Exception as e:
            # Calculate evaluation time even for errors
            evaluation_time_ms = (time.time() - start_time) * 1000

            return EvaluationResult(
                test_id=test.id,
                passed=False,
                predicted_tool_calls=[],
                expected_tool_calls=test.expected_tool_calls,
                error=str(e),
                evaluation_time_ms=evaluation_time_ms,
                voice_command=test.voice_command,
                test_description=test.description,
                tags=test.tags,
            )

    def _compare_tool_calls(
        self, predicted: List[ToolCall], expected: List[ToolCall]
    ) -> bool:
        """
        Compare predicted and expected tool calls.

        Args:
            predicted: List of predicted tool calls
            expected: List of expected tool calls

        Returns:
            True if they match, False otherwise
        """
        if len(predicted) != len(expected):
            return False

        # Create a more flexible comparison that checks names and key arguments
        for pred, exp in zip(predicted, expected):
            if pred.name != exp.name:
                return False

            # Check that all expected arguments are present with correct values
            for key, value in exp.arguments.items():
                if key not in pred.arguments:
                    return False
                if pred.arguments[key] != value:
                    return False

        return True


class GeminiTextEvaluator(TextEvaluator):
    """Text evaluator that uses Google Gemini API for text generation (no tools)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the Gemini text evaluator.

        Args:
            api_key: Google API key. If not provided, reads from GEMINI_API_KEY or APIKey env var
            model: Gemini model to use (default: gemini-2.5-flash-lite)
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        # Get API key from parameter, env vars, or .env file
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("APIKey")
        if not self.api_key:
            # Try to load from .env file
            self._load_env_file()
            self.api_key = os.getenv("APIKey")

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set via api_key parameter, "
                "GEMINI_API_KEY env var, or APIKey in .env file"
            )

        self.model = model
        self.timeout = timeout
        self.debug = debug

        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)

        super().__init__(model_client=self.client)

    def _load_env_file(self):
        """Load environment variables from .env file if present."""
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()

    def _get_model_response(self, question: str) -> str:
        """
        Get response from the Gemini model for a given question.

        Args:
            question: The question to ask the model

        Returns:
            The model's text response
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("DEBUG: TEXT REQUEST TO GEMINI API")
            print("=" * 60)
            print(f"Model: {self.model}")
            print(f"Question: {question}")
            print("=" * 60)

        # Create the request without tools
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=question)]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )

        if self.debug:
            print("\n" + "=" * 60)
            print("DEBUG: TEXT RESPONSE FROM GEMINI API")
            print("=" * 60)
            print(f"Response: {response}")
            print("=" * 60 + "\n")

        # Extract text from response
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                return parts[0].text

        return ""
