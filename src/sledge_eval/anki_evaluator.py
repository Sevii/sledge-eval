"""Anki MCP Server Large Tool Set Evaluator."""

from typing import List, Optional

from .server_evaluator import ServerEvaluator
from .tools.defaults import get_anki_tools


class AnkiLargeToolSetEvaluator(ServerEvaluator):
    """Evaluator that tests model performance with large tool sets using Anki MCP tools.

    This evaluator uses the 13 Anki MCP tools to test how well models handle
    larger tool sets.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the Anki large tool set evaluator.

        Args:
            server_url: URL of the llama-server instance
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        super().__init__(
            server_url=server_url,
            available_tools=get_anki_tools(),
            timeout=timeout,
            debug=debug,
        )

    def get_tool_count(self) -> int:
        """Get the number of available tools."""
        return len(self.available_tools)

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return [tool["function"]["name"] for tool in self.available_tools]
