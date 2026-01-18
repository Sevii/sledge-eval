"""Shared report generation logic for CLI tools."""

from pathlib import Path
from typing import Dict, List, Optional

from ..evaluator import EvaluationReport, EvaluationResult
from ..hardware_detector import HardwareDetector


class ReportGenerator:
    """Generates and saves evaluation reports."""

    def __init__(self, include_hardware_info: bool = True):
        """
        Initialize the report generator.

        Args:
            include_hardware_info: Whether to include hardware detection info in reports
        """
        self.include_hardware_info = include_hardware_info

    def generate_report(
        self,
        results: List[EvaluationResult],
        model_name: str,
        mode: str,
        server_url: Optional[str] = None,
        test_suite_name: Optional[str] = None,
        base_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Generate and save a comprehensive evaluation report.

        Args:
            results: List of evaluation results
            model_name: Name of the model being evaluated
            mode: Evaluation mode (single, suite, custom, all, anki, text)
            server_url: URL of the server (for server-based evaluations)
            test_suite_name: Name of the test suite
            base_path: Base path for saving reports (defaults to current directory)

        Returns:
            Dict with 'json' and 'markdown' keys containing the file paths
        """
        base_path = base_path or Path.cwd()

        # Calculate total evaluation time
        total_time = sum(r.evaluation_time_ms or 0 for r in results)

        # Detect hardware information if enabled
        hardware_info = None
        if self.include_hardware_info:
            hardware_detector = HardwareDetector()
            hardware_info = hardware_detector.extract_hardware_info()

        # Create report
        report = EvaluationReport(
            model_name=model_name,
            server_url=server_url,
            evaluation_mode=mode,
            test_suite_name=test_suite_name,
            hardware_info=hardware_info,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            pass_rate=0.0,
            total_evaluation_time_ms=total_time,
            test_results=[],
        )

        # Add all results
        for result in results:
            report.add_result(result)

        # Save report
        report_paths = report.save_to_file(base_path)

        return report_paths

    def print_report_paths(self, report_paths: Dict[str, Path]) -> None:
        """Print the paths of saved reports."""
        print(f"\n📊 Reports saved:")
        print(f"   JSON: {report_paths['json']}")
        print(f"   Markdown: {report_paths['markdown']}")
