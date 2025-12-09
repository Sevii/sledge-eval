"""Ministral-specific evaluator for tool calling with llama.cpp."""

import json
from typing import Any, Dict, List, Optional

from llama_cpp import Llama

from .evaluator import EvaluationResult, Evaluator, ToolCall, VoiceCommandTest


class MinistralEvaluator(Evaluator):
    """Evaluator using Ministral models via llama-cpp-python for tool calling."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        available_tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the Ministral evaluator.

        Args:
            model_path: Path to the Ministral GGUF model file
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all, 0 for CPU only)
            verbose: Enable verbose output from llama.cpp
            available_tools: List of tool definitions in OpenAI format
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.available_tools = available_tools or self._get_default_tools()
        super().__init__(model_client=self.llm)

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

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """
        Evaluate a single test case using Ministral.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        try:
            # Build messages for the model
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that interprets voice commands and calls appropriate functions.",
                },
                {"role": "user", "content": test.voice_command},
            ]

            # First pass: get model response with tools
            response = self.llm.create_chat_completion(
                messages=messages, tools=self.available_tools, tool_choice="auto"
            )

            predicted_tool_calls = []
            choice = response["choices"][0]

            # Check if model made tool calls
            if choice["finish_reason"] == "tool_calls":
                tool_calls = choice["message"]["tool_calls"]

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    predicted_tool_calls.append(
                        ToolCall(name=function_name, arguments=arguments)
                    )

            # Compare predicted vs expected tool calls
            passed = self._compare_tool_calls(
                predicted_tool_calls, test.expected_tool_calls
            )

            return EvaluationResult(
                test_id=test.id,
                passed=passed,
                predicted_tool_calls=predicted_tool_calls,
                expected_tool_calls=test.expected_tool_calls,
            )

        except Exception as e:
            return EvaluationResult(
                test_id=test.id,
                passed=False,
                predicted_tool_calls=[],
                expected_tool_calls=test.expected_tool_calls,
                error=str(e),
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
