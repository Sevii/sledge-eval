"""Gemini API-based evaluation runner."""

from typing import Any, Dict, List, Optional

from .runner import EvaluationRunner
from .report_generator import ReportGenerator


class GeminiRunner(EvaluationRunner):
    """Evaluation runner for Google Gemini API backend."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        timeout: int = 120,
        debug: bool = False,
    ):
        """
        Initialize the Gemini runner.

        Args:
            model: Gemini model to use
            api_key: Google API key (defaults to env vars)
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._evaluator = None
        self._anki_evaluator = None
        self._text_evaluator = None

        # Use model name as the model identifier
        super().__init__(
            model_name=model,
            debug=debug,
            report_generator=ReportGenerator(include_hardware_info=False),  # Cloud API
        )

    def get_evaluator(self):
        """Get the primary evaluator instance."""
        if self._evaluator is None:
            from ..gemini_evaluator import GeminiEvaluator

            self._evaluator = GeminiEvaluator(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout,
                debug=self.debug,
            )
        return self._evaluator

    def get_server_url(self) -> Optional[str]:
        """Get the server URL (Gemini API endpoint)."""
        return "https://generativelanguage.googleapis.com"

    def check_connection(self) -> bool:
        """Check if the Gemini API is available."""
        print(f"\nInitializing Gemini evaluator...")
        if self.debug:
            print("🐛 Debug mode enabled")

        try:
            # Initialize the evaluator (which validates the API key)
            evaluator = self.get_evaluator()
            print(f"✅ Connected to Gemini API with model: {self.model}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Gemini API: {e}")
            return False

    def get_custom_evaluator(self, custom_tools: List[Dict[str, Any]]):
        """Get an evaluator with custom tools."""
        from ..gemini_evaluator import GeminiEvaluator

        return GeminiEvaluator(
            api_key=self.api_key,
            model=self.model,
            available_tools=custom_tools,
            timeout=self.timeout,
            debug=self.debug,
        )

    def get_anki_evaluator(self):
        """Get an Anki evaluator."""
        if self._anki_evaluator is None:
            try:
                from ..gemini_anki_evaluator import GeminiAnkiEvaluator

                self._anki_evaluator = GeminiAnkiEvaluator(
                    api_key=self.api_key,
                    model=self.model,
                    timeout=self.timeout,
                    debug=self.debug,
                )
            except ImportError:
                return None
        return self._anki_evaluator

    def get_text_evaluator(self):
        """Get a text evaluator."""
        if self._text_evaluator is None:
            try:
                from ..gemini_evaluator import GeminiTextEvaluator

                self._text_evaluator = GeminiTextEvaluator(
                    api_key=self.api_key,
                    model=self.model,
                    timeout=self.timeout,
                    debug=self.debug,
                )
            except ImportError:
                return None
        return self._text_evaluator
