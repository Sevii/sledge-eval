"""OpenRouter API evaluator for models hosted on OpenRouter."""

from typing import Any, Dict, List, Optional

from .http_openai_client import OpenAIClientConfig
from .openai_evaluator import OpenAICompatibleEvaluator
from .utils.api_keys import resolve_api_key


class OpenRouterEvaluator(OpenAICompatibleEvaluator):
    """Evaluator that uses OpenRouter API for tool calling.

    This is a thin wrapper around OpenAICompatibleEvaluator configured
    for OpenRouter's API.
    """

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

        super().__init__(
            client_config=config,
            tools=available_tools,
            check_health=False,  # OpenRouter doesn't need health checks
        )

    def check_api_key(self) -> bool:
        """Verify that the API key is valid by making a test request.

        Returns:
            True if the API key is valid, False otherwise
        """
        import requests

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
