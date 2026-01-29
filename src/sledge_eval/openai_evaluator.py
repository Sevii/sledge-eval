"""Unified base evaluator for OpenAI-compatible APIs."""

import time
from typing import Any, Dict, List, Optional

from .evaluator import EvaluationResult, Evaluator, ToolCall, VoiceCommandTest
from .http_openai_client import OpenAIClientConfig, OpenAIHTTPClient
from .tools.defaults import get_default_tools


class OpenAICompatibleEvaluator(Evaluator):
    """Base evaluator for OpenAI-compatible APIs.

    This class provides a unified implementation of evaluate_test() that works
    with any OpenAI-compatible API (llama-server, OpenRouter, etc.).

    Subclasses only need to configure the client appropriately for their
    specific provider.
    """

    def __init__(
        self,
        client_config: OpenAIClientConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
        check_health: bool = True,
        system_prompt: str = "You are a helpful assistant that interprets voice commands and calls appropriate functions.",
    ):
        """Initialize the evaluator.

        Args:
            client_config: Configuration for the HTTP client
            tools: List of tool definitions in OpenAI format. If None, uses default tools.
            check_health: Whether to check API health before evaluating tests
            system_prompt: System prompt to use for all evaluations
        """
        self.client = OpenAIHTTPClient(client_config)
        self.available_tools = tools if tools is not None else get_default_tools()
        self.check_health = check_health
        self.system_prompt = system_prompt

        # Store config for subclass access
        self._config = client_config

        super().__init__(model_client=None)

    @property
    def debug(self) -> bool:
        """Get debug mode setting."""
        return self._config.debug

    @property
    def timeout(self) -> int:
        """Get timeout setting."""
        return self._config.timeout

    def _check_server_health(self) -> bool:
        """Check if the API is healthy and responding.

        Returns:
            True if healthy, False otherwise
        """
        return self.client.health_check()

    def evaluate_test(self, test: VoiceCommandTest) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            test: The test case to evaluate

        Returns:
            EvaluationResult with pass/fail and details
        """
        start_time = time.time()

        try:
            # Check server health first if enabled
            if self.check_health:
                if not self._check_server_health():
                    raise Exception(f"API is not responding at {self.client.base_url}")

                if self.debug:
                    print("DEBUG: Server health check passed, building request...")

            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": test.voice_command},
            ]

            # Make API request
            response_data = self.client.chat_completion(
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=512,
            )

            # Parse tool calls from response
            predicted_tool_calls = self.client.parse_tool_calls(response_data)

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

    def get_tool_count(self) -> int:
        """Get the number of available tools."""
        return len(self.available_tools)

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return [tool["function"]["name"] for tool in self.available_tools]
