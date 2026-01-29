"""Server-based evaluator that uses llama-server HTTP API."""

from typing import Any, Dict, List, Optional

from .http_openai_client import OpenAIClientConfig
from .openai_evaluator import OpenAICompatibleEvaluator


class ServerEvaluator(OpenAICompatibleEvaluator):
    """Evaluator that uses llama-server HTTP API for tool calling.

    This is a thin wrapper around OpenAICompatibleEvaluator configured
    for local llama-server instances.
    """

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
        # Store for compatibility with existing code
        self.server_url = server_url.rstrip('/')

        # Build client configuration
        config = OpenAIClientConfig(
            base_url=f"{self.server_url}/v1",
            timeout=timeout,
            debug=debug,
        )

        super().__init__(
            client_config=config,
            tools=available_tools,
            check_health=True,
        )

    def _get_default_tools(self) -> List[Dict[str, Any]]:
        """Get default tool definitions for common voice commands.

        Returns:
            List of tool definitions in OpenAI format
        """
        from .tools.defaults import get_default_tools
        return get_default_tools()
