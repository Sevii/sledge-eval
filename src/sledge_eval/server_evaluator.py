"""Server-based evaluator that uses llama-server HTTP API."""

import json
import time
import requests
from typing import Any, Dict, List, Optional

from .evaluator import EvaluationResult, Evaluator, ToolCall, VoiceCommandTest


class ServerEvaluator(Evaluator):
    """Evaluator that uses llama-server HTTP API for tool calling."""

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        available_tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the server evaluator.

        Args:
            server_url: URL of the llama-server instance
            available_tools: List of tool definitions in OpenAI format
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        self.server_url = server_url.rstrip('/')
        self.available_tools = available_tools or self._get_default_tools()
        self.timeout = timeout
        self.debug = debug
        super().__init__(model_client=None)

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

    def _check_server_health(self) -> bool:
        """
        Check if the llama-server is healthy and responding.

        Returns:
            True if server is healthy, False otherwise
        """
        # Try multiple endpoints that llama-server might expose
        endpoints = [
            "/health",
            "/v1/models", 
            "/",
            "/models",
        ]
        
        for endpoint in endpoints:
            try:
                url = f"{self.server_url}{endpoint}"
                if self.debug:
                    print(f"🐛 DEBUG: Health check trying: {url}")
                else:
                    print(f"   Trying: {url}")
                response = requests.get(url, timeout=5)
                if self.debug:
                    print(f"🐛 DEBUG: Health check response: {response.status_code}")
                else:
                    print(f"   Response: {response.status_code}")
                if response.status_code in [200, 404]:  # 404 is OK, server is responding
                    return True
            except requests.exceptions.ConnectionError as e:
                if "refused" in str(e).lower():
                    print(f"   Connection refused to {url}")
                else:
                    print(f"   Connection error to {url}: {e}")
            except requests.exceptions.Timeout:
                print(f"   Timeout connecting to {url}")
            except requests.exceptions.RequestException as e:
                print(f"   Error connecting to {url}: {e}")
        
        return False

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """
        Evaluate a single test case using the llama-server.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        start_time = time.time()
        
        try:
            # Check server health first
            if not self._check_server_health():
                raise Exception(f"llama-server is not responding at {self.server_url}")

            if self.debug:
                print("🐛 DEBUG: Server health check passed, building request...")

            # Build request payload
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that interprets voice commands and calls appropriate functions.",
                    },
                    {"role": "user", "content": test.voice_command},
                ],
                "tools": self.available_tools,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 512,
            }

            # Debug logging for request
            if self.debug:
                print("\n" + "=" * 60)
                print("🐛 DEBUG: REQUEST TO LLM SERVER")
                print("=" * 60)
                print(f"URL: {self.server_url}/v1/chat/completions")
                print(f"Method: POST")
                print(f"Headers: {{'Content-Type': 'application/json'}}")
                print(f"Timeout: {self.timeout}s")
                print("\nRequest Payload:")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                print("=" * 60)

            # Make request to chat completions endpoint
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            # Debug logging for response
            if self.debug:
                print("\n" + "=" * 60)
                print("🐛 DEBUG: RESPONSE FROM LLM SERVER")
                print("=" * 60)
                print(f"Status Code: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")
                print("\nResponse Body:")
                try:
                    response_json = response.json()
                    print(json.dumps(response_json, indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(f"Raw response text: {response.text}")
                print("=" * 60 + "\n")

            if response.status_code != 200:
                raise Exception(f"Server returned {response.status_code}: {response.text}")

            response_data = response.json()
            predicted_tool_calls = []

            # Parse tool calls from response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                message = choice.get("message", {})
                
                # Check for tool calls
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            function_name = function.get("name", "")
                            
                            # Parse arguments
                            arguments_str = function.get("arguments", "{}")
                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                arguments = {}

                            predicted_tool_calls.append(
                                ToolCall(name=function_name, arguments=arguments)
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