"""Server-based text evaluator implementation."""

import requests
import time
from typing import Dict, Any, Optional

from .text_evaluator import TextEvaluator


class TextServerEvaluator(TextEvaluator):
    """Text evaluator that works with llama-server HTTP API."""

    def __init__(self, server_url: str = "http://localhost:8080", timeout: int = 30):
        """
        Initialize the text server evaluator.

        Args:
            server_url: URL of the llama-server instance
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.timeout = timeout
        super().__init__(model_client=None)  # We don't use the model_client for HTTP requests

    def health_check(self) -> bool:
        """Check if the server is running and responsive."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

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
        # Prepare the chat completion request
        payload = {
            "messages": [
                {
                    "role": "user", 
                    "content": question
                }
            ],
            "temperature": 0.1,  # Low temperature for more deterministic responses
            "max_tokens": 100,   # Short responses for text evaluation
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Server request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid response format: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")


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