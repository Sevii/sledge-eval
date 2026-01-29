"""Server-based text evaluator implementation."""

from typing import Optional

from .http_openai_client import OpenAIClientConfig, OpenAIHTTPClient
from .text_evaluator import TextEvaluator


class TextServerEvaluator(TextEvaluator):
    """Text evaluator that works with llama-server HTTP API.

    Uses the shared OpenAIHTTPClient for making requests to the server.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        timeout: int = 30,
        debug: bool = False,
    ):
        """
        Initialize the text server evaluator.

        Args:
            server_url: URL of the llama-server instance
            timeout: Request timeout in seconds
            debug: Enable debug logging of requests and responses
        """
        self.server_url = server_url.rstrip('/')

        # Build client configuration
        config = OpenAIClientConfig(
            base_url=f"{self.server_url}/v1",
            timeout=timeout,
            debug=debug,
        )

        self.client = OpenAIHTTPClient(config)
        self._debug = debug

        super().__init__(model_client=None)  # We don't use the model_client for HTTP requests

    def health_check(self) -> bool:
        """Check if the server is running and responsive."""
        return self.client.health_check()

    def _get_model_response(self, question: str) -> str:
        """
        Get response from the model server for a given question.

        Args:
            question: The question to ask the model

        Returns:
            The model's response text

        Raises:
            Exception: If the server request fails
        """
        messages = [
            {"role": "user", "content": question}
        ]

        response_data = self.client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=100,
        )

        return self.client.extract_text_content(response_data)


def create_letter_counting_test_file(output_path: str = "tests/test_data/letter_counting_suite.json"):
    """
    Create a JSON test file for letter counting evaluations.

    Args:
        output_path: Path where to save the test file
    """
    import json
    from pathlib import Path
    from .text_evaluator import create_letter_counting_test_suite

    # Create the test suite
    test_suite = create_letter_counting_test_suite()

    # Ensure directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(test_suite.model_dump(), f, indent=2)

    print(f"Created letter counting test file at: {output_file}")
    return output_file
