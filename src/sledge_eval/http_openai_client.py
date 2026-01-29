"""HTTP client for OpenAI-compatible APIs."""

import json
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evaluator import ToolCall


@dataclass
class OpenAIClientConfig:
    """Configuration for OpenAI-compatible HTTP client.

    Args:
        base_url: Base URL for the API (e.g., "https://openrouter.ai/api/v1" or "http://localhost:8080/v1")
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds
        debug: Enable debug logging of requests and responses
        headers: Additional headers to include in requests
        model: Model ID (required for some providers like OpenRouter)
    """
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 120
    debug: bool = False
    headers: Dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None


class OpenAIHTTPClient:
    """HTTP client for OpenAI-compatible chat completion APIs.

    This client provides a unified interface for making requests to various
    OpenAI-compatible APIs (OpenRouter, llama-server, etc.).
    """

    def __init__(self, config: OpenAIClientConfig):
        """Initialize the HTTP client.

        Args:
            config: Client configuration
        """
        self.config = config
        self.base_url = config.base_url.rstrip('/')

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers including authentication."""
        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Merge additional headers from config
        headers.update(self.config.headers)

        return headers

    def _debug_request(self, url: str, payload: dict):
        """Log request details for debugging."""
        if not self.config.debug:
            return

        print("\n" + "=" * 60)
        print("DEBUG: REQUEST TO API")
        print("=" * 60)
        print(f"URL: {url}")
        if self.config.model:
            print(f"Model: {self.config.model}")
        print(f"Method: POST")
        print(f"Timeout: {self.config.timeout}s")
        print("\nRequest Payload:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 60)

    def _debug_response(self, response: requests.Response):
        """Log response details for debugging."""
        if not self.config.debug:
            return

        print("\n" + "=" * 60)
        print("DEBUG: RESPONSE FROM API")
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

    def health_check(self) -> bool:
        """Check if the API is healthy and responding.

        Returns:
            True if API is healthy, False otherwise
        """
        # Try multiple endpoints that might be available
        endpoints = [
            "/health",
            "/models",
            "/",
        ]

        for endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                if self.config.debug:
                    print(f"DEBUG: Health check trying: {url}")
                else:
                    print(f"   Trying: {url}")

                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=5
                )

                if self.config.debug:
                    print(f"DEBUG: Health check response: {response.status_code}")
                else:
                    print(f"   Response: {response.status_code}")

                # 200 means success, 404 is OK (server responding but endpoint not found)
                if response.status_code in [200, 404]:
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

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> dict:
        """Make a chat completion request.

        Args:
            messages: List of message dictionaries with role and content
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Tool selection strategy ("auto", "none", or specific tool)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            The API response as a dictionary

        Raises:
            Exception: If the request fails
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add model if configured
        if self.config.model:
            payload["model"] = self.config.model

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        self._debug_request(url, payload)

        response = requests.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=self.config.timeout,
        )

        self._debug_response(response)

        # Handle error responses
        self._handle_error_response(response)

        return response.json()

    def _handle_error_response(self, response: requests.Response):
        """Handle HTTP error responses with descriptive messages.

        Args:
            response: The HTTP response

        Raises:
            Exception: With descriptive error message
        """
        if response.status_code == 200:
            return

        if response.status_code == 401:
            raise Exception("Invalid API key (401 Unauthorized)")
        elif response.status_code == 402:
            raise Exception("Payment required - check your account balance (402)")
        elif response.status_code == 429:
            raise Exception("Rate limited - too many requests (429)")
        else:
            raise Exception(f"API returned {response.status_code}: {response.text}")

    def parse_tool_calls(self, response_data: dict) -> List[ToolCall]:
        """Parse tool calls from an API response.

        Args:
            response_data: The API response dictionary

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        if "choices" not in response_data or len(response_data["choices"]) == 0:
            return tool_calls

        choice = response_data["choices"][0]
        message = choice.get("message", {})

        if "tool_calls" not in message:
            return tool_calls

        for tool_call in message["tool_calls"]:
            if tool_call.get("type") != "function":
                continue

            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            # Parse arguments
            arguments_str = function.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}

            tool_calls.append(
                ToolCall(name=function_name, arguments=arguments)
            )

        return tool_calls

    def extract_text_content(self, response_data: dict) -> str:
        """Extract text content from an API response.

        Args:
            response_data: The API response dictionary

        Returns:
            The text content from the response
        """
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            return ""

        message = response_data["choices"][0].get("message", {})
        return message.get("content", "").strip()
