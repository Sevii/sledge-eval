"""OpenRouter API-based evaluation runner."""

from typing import Any, Dict, List, Optional

from .runner import EvaluationRunner
from .report_generator import ReportGenerator


class OpenRouterRunner(EvaluationRunner):
    """Evaluation runner for OpenRouter API backend."""

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
        Initialize the OpenRouter runner.

        Args:
            model: OpenRouter model ID (e.g., 'anthropic/claude-3-haiku')
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            debug: Enable debug logging
            site_url: Optional site URL for OpenRouter ranking
            app_name: App name for OpenRouter ranking (default: sledge-eval)
        """
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.site_url = site_url
        self.app_name = app_name
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
            from ..openrouter_evaluator import OpenRouterEvaluator

            self._evaluator = OpenRouterEvaluator(
                model=self.model,
                api_key=self.api_key,
                timeout=self.timeout,
                debug=self.debug,
                site_url=self.site_url,
                app_name=self.app_name,
            )
        return self._evaluator

    def get_server_url(self) -> Optional[str]:
        """Get the server URL (OpenRouter API endpoint)."""
        return "https://openrouter.ai/api/v1"

    def get_hosting_provider(self) -> Optional[str]:
        """Get the hosting provider name."""
        return "OpenRouter"

    def check_connection(self) -> bool:
        """Check if the OpenRouter API is available."""
        print(f"\nInitializing OpenRouter evaluator...")
        if self.debug:
            print("Debug mode enabled")

        try:
            # Initialize the evaluator (which validates the API key)
            evaluator = self.get_evaluator()

            # Verify API key is valid
            if evaluator.check_api_key():
                print(f"Connected to OpenRouter API with model: {self.model}")
                return True
            else:
                print("Failed to verify OpenRouter API key")
                return False
        except Exception as e:
            print(f"Failed to connect to OpenRouter API: {e}")
            return False

    def get_custom_evaluator(self, custom_tools: List[Dict[str, Any]]):
        """Get an evaluator with custom tools."""
        from ..openrouter_evaluator import OpenRouterEvaluator

        return OpenRouterEvaluator(
            model=self.model,
            api_key=self.api_key,
            available_tools=custom_tools,
            timeout=self.timeout,
            debug=self.debug,
            site_url=self.site_url,
            app_name=self.app_name,
        )

    def get_anki_evaluator(self):
        """Get an Anki evaluator."""
        if self._anki_evaluator is None:
            try:
                from ..openrouter_anki_evaluator import OpenRouterAnkiEvaluator

                self._anki_evaluator = OpenRouterAnkiEvaluator(
                    model=self.model,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    debug=self.debug,
                    site_url=self.site_url,
                    app_name=self.app_name,
                )
            except ImportError:
                return None
        return self._anki_evaluator

    def get_text_evaluator(self):
        """Get a text evaluator."""
        if self._text_evaluator is None:
            try:
                from ..openrouter_text_evaluator import OpenRouterTextEvaluator

                self._text_evaluator = OpenRouterTextEvaluator(
                    model=self.model,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    debug=self.debug,
                    site_url=self.site_url,
                    app_name=self.app_name,
                )
            except ImportError:
                return None
        return self._text_evaluator
