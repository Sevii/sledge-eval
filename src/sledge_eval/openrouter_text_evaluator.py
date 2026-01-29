"""OpenRouter text evaluator (no tools) for models hosted on OpenRouter."""

import json
import os
import requests
from typing import Optional

from .text_evaluator import TextEvaluator


class OpenRouterTextEvaluator(TextEvaluator):
    """Text evaluator that uses OpenRouter API for text generation (no tools)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        debug: bool = False,
        site_url: Optional[str] = None,
        app_name: str = "sledge-eval",
    ):
        """
        Initialize the OpenRouter text evaluator.

        Args:
            model: OpenRouter model ID (e.g., 'anthropic/claude-3-haiku')
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var
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

    def _get_headers(self) -> dict:
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

    def _get_model_response(self, question: str) -> str:
        """
        Get response from the OpenRouter model for a given question.

        Args:
            question: The question to ask the model

        Returns:
            The model's text response
        """
        if self.debug:
            print("\n" + "=" * 60)
            print("DEBUG: TEXT REQUEST TO OPENROUTER API")
            print("=" * 60)
            print(f"Model: {self.model}")
            print(f"Question: {question}")
            print("=" * 60)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": question}
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._get_headers(),
            timeout=self.timeout,
        )

        if self.debug:
            print("\n" + "=" * 60)
            print("DEBUG: TEXT RESPONSE FROM OPENROUTER API")
            print("=" * 60)
            print(f"Status Code: {response.status_code}")
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

        # Extract text from response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            return message.get("content", "")

        return ""
