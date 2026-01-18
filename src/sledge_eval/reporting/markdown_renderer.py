"""Markdown report renderer for evaluation results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import EvaluationReport, EvaluationResult
    from ..hardware_detector import HardwareInfo


class MarkdownRenderer:
    """Renders evaluation reports to Markdown format."""

    def render(self, report: "EvaluationReport") -> str:
        """
        Render an evaluation report to Markdown.

        Args:
            report: The evaluation report to render

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.extend(self._render_header(report))

        # Hardware Information
        if report.hardware_info:
            lines.extend(self._render_hardware_info(report.hardware_info))

        # Executive Summary
        lines.extend(self._render_executive_summary(report))

        # Performance by Tag
        if report.tag_performance:
            lines.extend(self._render_tag_performance(report.tag_performance))

        # Detailed Test Results
        lines.extend(self._render_test_results(report.test_results))

        # Error Summary
        if report.errors:
            lines.extend(self._render_errors(report.errors))

        # Technical Details
        lines.extend(self._render_technical_details(report))

        # Footer
        lines.extend(self._render_footer())

        return "\n".join(lines)

    def _render_header(self, report: "EvaluationReport") -> List[str]:
        """Render the report header."""
        lines = []
        lines.append(f"# Evaluation Report: {report.display_name}")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp.strftime('%B %d, %Y at %I:%M:%S %p')}")
        lines.append(f"**Model:** `{report.display_name}`")
        lines.append(f"**Server URL:** {report.server_url or 'N/A'}")
        lines.append(f"**Evaluation Mode:** {report.evaluation_mode}")
        if report.test_suite_name:
            lines.append(f"**Test Suite:** {report.test_suite_name}")
        lines.append("")
        return lines

    def _render_hardware_info(self, hardware_info: "HardwareInfo") -> List[str]:
        """Render hardware information section."""
        lines = []
        lines.append("## 🖥️ Hardware Information")
        lines.append("")

        # Create hardware summary table
        lines.append("| Component | Details |")
        lines.append("|-----------|---------|")

        if hardware_info.gpu_name:
            gpu_details = hardware_info.gpu_name
            if hardware_info.gpu_family:
                gpu_details += f" ({hardware_info.gpu_family})"
            lines.append(f"| **GPU** | {gpu_details} |")

        if hardware_info.processor:
            lines.append(f"| **CPU** | {hardware_info.processor} |")

        if hardware_info.metal_backend:
            lines.append(f"| **Compute Backend** | Metal |")

        if hardware_info.gpu_memory_mb:
            lines.append(f"| **GPU Memory** | {hardware_info.gpu_memory_mb:.0f} MB |")

        if hardware_info.total_memory_mb:
            lines.append(f"| **Total Memory** | {hardware_info.total_memory_mb:.0f} MB |")

        if hardware_info.model_memory_mb:
            lines.append(f"| **Model Memory** | {hardware_info.model_memory_mb:.0f} MB |")

        if hardware_info.context_memory_mb:
            lines.append(f"| **Context Memory** | {hardware_info.context_memory_mb:.0f} MB |")

        if hardware_info.model_size_gb:
            lines.append(f"| **Model Size** | {hardware_info.model_size_gb:.2f} GB |")

        if hardware_info.model_params_b:
            lines.append(f"| **Model Parameters** | {hardware_info.model_params_b:.1f}B |")

        if hardware_info.n_threads:
            thread_info = f"{hardware_info.n_threads}"
            if hardware_info.n_threads_batch:
                thread_info += f" (batch: {hardware_info.n_threads_batch})"
            lines.append(f"| **Threads** | {thread_info} |")

        if hardware_info.context_size:
            lines.append(f"| **Context Size** | {hardware_info.context_size:,} |")

        if hardware_info.os_name and hardware_info.architecture:
            lines.append(f"| **System** | {hardware_info.os_name} {hardware_info.architecture} |")

        # GPU Capabilities
        capabilities = []
        if hardware_info.has_unified_memory:
            capabilities.append("Unified Memory")
        if hardware_info.has_bfloat:
            capabilities.append("BFloat16")
        if hardware_info.has_tensor is False:
            capabilities.append("No Tensor API")

        if capabilities:
            lines.append(f"| **GPU Features** | {', '.join(capabilities)} |")

        if hardware_info.llama_cpp_build and hardware_info.llama_cpp_commit:
            build_info = f"Build {hardware_info.llama_cpp_build} ({hardware_info.llama_cpp_commit[:8]})"
            lines.append(f"| **llama.cpp** | {build_info} |")

        lines.append("")
        return lines

    def _render_executive_summary(self, report: "EvaluationReport") -> List[str]:
        """Render executive summary section."""
        lines = []
        lines.append("## 📊 Executive Summary")
        lines.append("")

        # Pass rate with emoji
        if report.pass_rate >= 90:
            status_emoji = "🟢"
        elif report.pass_rate >= 70:
            status_emoji = "🟡"
        else:
            status_emoji = "🔴"

        lines.append(
            f"- **Overall Score:** {status_emoji} {report.pass_rate:.1f}% "
            f"({report.passed_tests}/{report.total_tests} tests passed)"
        )
        lines.append(f"- **Total Evaluation Time:** {report.total_evaluation_time_ms:.1f}ms")

        if report.test_results:
            avg_time = sum(r.evaluation_time_ms or 0 for r in report.test_results) / len(report.test_results)
            lines.append(f"- **Average Test Time:** {avg_time:.1f}ms")

        lines.append("")
        return lines

    def _render_tag_performance(self, tag_performance: Dict[str, Dict[str, Any]]) -> List[str]:
        """Render performance by tag section."""
        lines = []
        lines.append("## 🏷️ Performance by Category")
        lines.append("")
        lines.append("| Category | Passed | Failed | Total | Pass Rate |")
        lines.append("|----------|--------|--------|-------|-----------|")

        for tag, perf in sorted(tag_performance.items()):
            pass_rate = (perf["passed"] / perf["total"] * 100) if perf["total"] > 0 else 0
            if pass_rate >= 90:
                emoji = "🟢"
            elif pass_rate >= 70:
                emoji = "🟡"
            else:
                emoji = "🔴"
            lines.append(
                f"| {emoji} {tag} | {perf['passed']} | {perf['failed']} | {perf['total']} | {pass_rate:.1f}% |"
            )
        lines.append("")
        return lines

    def _render_test_results(self, test_results: List["EvaluationResult"]) -> List[str]:
        """Render detailed test results section."""
        lines = []
        lines.append("## 📋 Detailed Test Results")
        lines.append("")

        # Group by pass/fail
        passed_tests = [r for r in test_results if r.passed]
        failed_tests = [r for r in test_results if not r.passed]

        if passed_tests:
            lines.append("### ✅ Passed Tests")
            lines.append("")
            for result in passed_tests:
                lines.extend(self._render_single_result(result))

        if failed_tests:
            lines.append("### ❌ Failed Tests")
            lines.append("")
            for result in failed_tests:
                lines.extend(self._render_single_result(result, show_failure_details=True))

        return lines

    def _render_single_result(
        self, result: "EvaluationResult", show_failure_details: bool = False
    ) -> List[str]:
        """Render a single test result."""
        lines = []
        lines.append(f"#### {result.test_id}")
        lines.append("")

        if result.voice_command:
            lines.append(f"**Command:** `{result.voice_command}`")
        if result.test_description:
            lines.append(f"**Description:** {result.test_description}")
        if result.tags:
            lines.append(f"**Tags:** {', '.join(result.tags)}")
        if result.evaluation_time_ms:
            lines.append(f"**Execution Time:** {result.evaluation_time_ms:.1f}ms")

        if show_failure_details and result.error:
            lines.append("")
            lines.append(f"**Error:** `{result.error}`")

        lines.append("")
        lines.append("**Expected Tool Calls:**")
        for tc in result.expected_tool_calls:
            lines.append(f"- `{tc.name}({tc.arguments})`")

        lines.append("")
        lines.append("**Predicted Tool Calls:**")
        if result.predicted_tool_calls:
            for tc in result.predicted_tool_calls:
                lines.append(f"- `{tc.name}({tc.arguments})`")
        else:
            lines.append("- *No tool calls predicted*")

        lines.append("")
        return lines

    def _render_errors(self, errors: List[str]) -> List[str]:
        """Render error summary section."""
        lines = []
        lines.append("## ⚠️ Errors Encountered")
        lines.append("")
        for i, error in enumerate(errors, 1):
            lines.append(f"{i}. `{error}`")
        lines.append("")
        return lines

    def _render_technical_details(self, report: "EvaluationReport") -> List[str]:
        """Render technical details section."""
        lines = []
        lines.append("## 🔧 Technical Details")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Model Name | `{report.model_name}` |")
        lines.append(f"| Server URL | {report.server_url or 'N/A'} |")
        lines.append(f"| Evaluation Mode | {report.evaluation_mode} |")
        lines.append(f"| Timestamp | {report.timestamp.isoformat()} |")
        lines.append(f"| Total Tests | {report.total_tests} |")
        lines.append(f"| Passed Tests | {report.passed_tests} |")
        lines.append(f"| Failed Tests | {report.failed_tests} |")
        lines.append(f"| Pass Rate | {report.pass_rate:.2f}% |")
        lines.append(f"| Total Time | {report.total_evaluation_time_ms:.1f}ms |")

        # Add hardware summary to technical details
        if report.hardware_info:
            if report.hardware_info.model_size_gb and report.hardware_info.model_memory_mb:
                efficiency = (report.hardware_info.model_size_gb * 1024) / report.hardware_info.model_memory_mb
                lines.append(f"| Memory Efficiency | {efficiency:.2f}x |")

            if report.total_evaluation_time_ms > 0 and report.hardware_info.n_threads:
                throughput = (
                    report.total_tests / (report.total_evaluation_time_ms / 1000) * report.hardware_info.n_threads
                )
                lines.append(f"| Throughput (tests/sec/thread) | {throughput:.2f} |")

        lines.append("")
        return lines

    def _render_footer(self) -> List[str]:
        """Render report footer."""
        lines = []
        lines.append("---")
        lines.append("*This report was automatically generated by Sledge Eval*")
        return lines
