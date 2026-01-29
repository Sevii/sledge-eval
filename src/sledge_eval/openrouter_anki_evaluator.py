"""Anki Large Tool Set Evaluator for OpenRouter API."""

from typing import List, Optional

from .openrouter_evaluator import OpenRouterEvaluator
from .tools.defaults import get_anki_tools


class OpenRouterAnkiEvaluator(OpenRouterEvaluator):
    """Evaluator that tests OpenRouter model performance with large tool sets using Anki MCP tools.

    This evaluator uses the 13 Anki MCP tools to test how well models hosted on
    OpenRouter handle larger tool sets.
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
        Initialize the OpenRouter Anki large tool set evaluator.

        Args:
            model: OpenRouter model ID (e.g., 'anthropic/claude-3-haiku')
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
            site_url: Optional site URL for OpenRouter ranking
            app_name: App name for OpenRouter ranking (default: sledge-eval)
        """
        super().__init__(
            model=model,
            api_key=api_key,
            available_tools=get_anki_tools(),
            timeout=timeout,
            debug=debug,
            site_url=site_url,
            app_name=app_name,
        )

    def get_tool_count(self) -> int:
        """Get the number of available tools."""
        return len(self.available_tools)

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return [tool["function"]["name"] for tool in self.available_tools]
