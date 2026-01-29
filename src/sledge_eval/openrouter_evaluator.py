"""OpenRouter API evaluator for models hosted on OpenRouter."""

import json
import os
import time
import requests
from typing import Any, Dict, List, Optional

from .evaluator import EvaluationResult, Evaluator, ToolCall, VoiceCommandTest


class OpenRouterEvaluator(Evaluator):
    """Evaluator that uses OpenRouter API for tool calling."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 120,
        debug: bool = False,
        site_url: Optional[str] = None,
        app_name: str = "sledge-eval",
    ):
        """
        Initialize the OpenRouter evaluator.

        Args:
            model: OpenRouter model ID (e.g., 'anthropic/claude-3-haiku')
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var
            available_tools: List of tool definitions in OpenAI format
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
            site_url: Optional site URL for OpenRouter ranking
            app_name: App name for OpenRouter ranking (default: sledge-eval)
        """
        # Get API key from parameter, env vars, or .env file
        self.api_key = api_key or self._resolve_api_key()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set via api_key parameter, "
                "OPENROUTER_API_KEY env var, or OPENROUTER_API_KEY in .env file"
            )

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.available_tools = available_tools or self._get_default_tools()
        self.timeout = timeout
        self.debug = debug
        self.site_url = site_url
        self.app_name = app_name

        super().__init__(model_client=None)

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key from environment variables or .env file."""
        # Try environment variable first
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            return api_key

        # Try .env file
        from pathlib import Path
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == "OPENROUTER_API_KEY":
                            return value.strip()

        return None

    def _get_default_tools(self) -> List[Dict[str, Any]]:
        """
        Get default tool definitions for common voice commands.

        Returns:
            List of tool definitions in OpenAI format
        """
        from .tools.defaults import get_default_tools
        return get_default_tools()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including authentication and optional ranking headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Optional ranking headers
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        return headers

    def check_api_key(self) -> bool:
        """
        Verify that the API key is valid by making a test request.

        Returns:
            True if the API key is valid, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """
        Evaluate a single test case using the OpenRouter API.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        start_time = time.time()

        try:
            # Build request payload
            payload = {
                "model": self.model,
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
                print("DEBUG: REQUEST TO OPENROUTER API")
                print("=" * 60)
                print(f"URL: {self.base_url}/chat/completions")
                print(f"Model: {self.model}")
                print(f"Method: POST")
                print(f"Timeout: {self.timeout}s")
                print("\nRequest Payload:")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                print("=" * 60)

            # Make request to chat completions endpoint
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            # Debug logging for response
            if self.debug:
                print("\n" + "=" * 60)
                print("DEBUG: RESPONSE FROM OPENROUTER API")
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

            # Handle specific error codes
            if response.status_code == 401:
                raise Exception("Invalid API key (401 Unauthorized)")
            elif response.status_code == 402:
                raise Exception("Payment required - check your OpenRouter account balance (402)")
            elif response.status_code == 429:
                raise Exception("Rate limited - too many requests (429)")
            elif response.status_code != 200:
                raise Exception(f"OpenRouter API returned {response.status_code}: {response.text}")

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

    # _compare_tool_calls is inherited from the base Evaluator class
