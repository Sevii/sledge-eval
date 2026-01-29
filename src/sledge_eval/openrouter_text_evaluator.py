"""OpenRouter text evaluator (no tools) for models hosted on OpenRouter."""

from typing import Dict, Optional

from .http_openai_client import OpenAIClientConfig, OpenAIHTTPClient
from .text_evaluator import TextEvaluator
from .utils.api_keys import resolve_api_key


class OpenRouterTextEvaluator(TextEvaluator):
    """Text evaluator that uses OpenRouter API for text generation (no tools).

    Uses the shared OpenAIHTTPClient for making requests to OpenRouter.
    """

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
        # Resolve API key
        resolved_key = resolve_api_key("OPENROUTER_API_KEY", api_key)

        if not resolved_key:
            raise ValueError(
                "OpenRouter API key required. Set via api_key parameter, "
                "OPENROUTER_API_KEY env var, or OPENROUTER_API_KEY in .env file"
            )

        # Store for compatibility with existing code
        self.model = model
        self.api_key = resolved_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = site_url
        self.app_name = app_name
        self._debug = debug

        # Build headers for OpenRouter
        headers: Dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        # Build client configuration
        config = OpenAIClientConfig(
            base_url=self.base_url,
            api_key=resolved_key,
            timeout=timeout,
            debug=debug,
            headers=headers,
            model=model,
        )

        self.client = OpenAIHTTPClient(config)

        super().__init__(model_client=None)

    @property
    def debug(self) -> bool:
        """Get debug mode setting."""
        return self._debug

    @property
    def timeout(self) -> int:
        """Get timeout setting."""
        return self.client.config.timeout

    def _get_model_response(self, question: str) -> str:
        """
        Get response from the OpenRouter model for a given question.

        Args:
            question: The question to ask the model

        Returns:
            The model's text response
        """
        messages = [
            {"role": "user", "content": question}
        ]

        response_data = self.client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )

        content = self.client.extract_text_content(response_data)

        if not content and self.debug:
            print("Warning: Model returned empty response")

        return content
