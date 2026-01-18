"""JSON report renderer for evaluation results."""

import json
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import EvaluationReport


class JsonRenderer:
    """Renders evaluation reports to JSON format."""

    def __init__(self, indent: int = 2, sort_keys: bool = False):
        """
        Initialize the JSON renderer.

        Args:
            indent: Number of spaces for indentation
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys

    def render(self, report: "EvaluationReport") -> str:
        """
        Render an evaluation report to JSON.

        Args:
            report: The evaluation report to render

        Returns:
            JSON string
        """
        return json.dumps(
            report.model_dump(),
            indent=self.indent,
            default=str,
            sort_keys=self.sort_keys,
        )

    def render_to_dict(self, report: "EvaluationReport") -> Dict[str, Any]:
        """
        Render an evaluation report to a dictionary.

        Args:
            report: The evaluation report to render

        Returns:
            Dictionary representation of the report
        """
        return report.model_dump()
