"""Server-based evaluation runner."""

import subprocess
from typing import Any, Dict, List, Optional

import requests

from .runner import EvaluationRunner
from ..server_evaluator import ServerEvaluator
from ..anki_evaluator import AnkiLargeToolSetEvaluator
from ..text_server_evaluator import TextServerEvaluator
from ..hardware_detector import HardwareDetector


class ServerRunner(EvaluationRunner):
    """Evaluation runner for llama-server backend."""

    def __init__(
        self,
        server_url: str,
        model_name: Optional[str] = None,
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the server runner.

        Args:
            server_url: URL of the llama-server instance
            model_name: Override model name (auto-detected if not provided)
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._evaluator: Optional[ServerEvaluator] = None
        self._anki_evaluator: Optional[AnkiLargeToolSetEvaluator] = None
        self._text_evaluator: Optional[TextServerEvaluator] = None

        # Auto-detect model name if not provided
        resolved_model_name = model_name or self._extract_model_name()

        super().__init__(model_name=resolved_model_name, debug=debug)

    def _extract_model_name(self) -> str:
        """Extract model name from server or use URL as fallback."""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].get("id", "unknown")
                elif "models" in data and len(data["models"]) > 0:
                    return data["models"][0].get("model", "unknown")
        except Exception:
            pass

        # Fallback to URL-based name
        return f"server_{self.server_url.replace('http://', '').replace('https://', '').replace(':', '_').replace('/', '_')}"

    def get_evaluator(self) -> ServerEvaluator:
        """Get the primary evaluator instance."""
        if self._evaluator is None:
            self._evaluator = ServerEvaluator(
                server_url=self.server_url,
                timeout=self.timeout,
                debug=self.debug,
            )
        return self._evaluator

    def get_server_url(self) -> Optional[str]:
        """Get the server URL."""
        return self.server_url

    def check_connection(self) -> bool:
        """Check if the server is available."""
        print(f"\nConnecting to llama-server at: {self.server_url}")
        if self.debug:
            print("🐛 Debug mode enabled")

        evaluator = self.get_evaluator()
        if not evaluator._check_server_health():
            self._print_connection_error()
            return False

        # Display hardware information
        hardware_detector = HardwareDetector()
        hardware_summary = hardware_detector.get_hardware_summary()
        if hardware_summary:
            print(f"\n🖥️ Hardware Information:")
            for key, value in hardware_summary.items():
                print(f"   {key}: {value}")

        return True

    def _print_connection_error(self) -> None:
        """Print detailed connection error message."""
        print("❌ Server is not responding. Make sure llama-server is running.")
        print(f"   Base URL: {self.server_url}")

        # Check if anything is listening on the port
        try:
            port = self.server_url.split(":")[-1]
            result = subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"   Something is listening on port {port}:")
                print("   " + result.stdout.replace("\n", "\n   "))
            else:
                print(f"   Nothing is listening on port {port}")
        except Exception:
            pass

        print("   Possible issues:")
        print("   - Server not started yet")
        print("   - Wrong port number")
        print("   - Server crashed during startup")
        print("   - Firewall blocking connection")
        print("   - Server doesn't support expected API endpoints")

    def get_custom_evaluator(self, custom_tools: List[Dict[str, Any]]) -> ServerEvaluator:
        """Get an evaluator with custom tools."""
        return ServerEvaluator(
            server_url=self.server_url,
            available_tools=custom_tools,
            timeout=self.timeout,
            debug=self.debug,
        )

    def get_anki_evaluator(self) -> Optional[AnkiLargeToolSetEvaluator]:
        """Get an Anki evaluator."""
        if self._anki_evaluator is None:
            self._anki_evaluator = AnkiLargeToolSetEvaluator(
                server_url=self.server_url,
                timeout=self.timeout,
                debug=self.debug,
            )
        return self._anki_evaluator

    def get_text_evaluator(self) -> Optional[TextServerEvaluator]:
        """Get a text evaluator."""
        if self._text_evaluator is None:
            self._text_evaluator = TextServerEvaluator(
                server_url=self.server_url,
                timeout=self.timeout,
            )
        return self._text_evaluator
