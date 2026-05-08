"""Markdown report renderer for prompt comparison results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import PromptComparisonReport, PromptComparisonResult


class ComparisonRenderer:
    """Renders prompt comparison reports to Markdown format."""

    def render(self, report: "PromptComparisonReport") -> str:
        """
        Render a prompt comparison report to Markdown.

        Args:
            report: The prompt comparison report to render

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.extend(self._render_header(report))

        # Summary Table
        lines.extend(self._render_summary_table(report))

        # Per-Test Comparison
        lines.extend(self._render_per_test_comparison(report))

        # Variable Results Section
        if report.variable_test_ids:
            lines.extend(self._render_variable_results(report))

        # Prompt Details
        lines.extend(self._render_prompt_details(report))

        # Footer
        lines.extend(self._render_footer())

        return "\n".join(lines)

    def _render_header(self, report: "PromptComparisonReport") -> List[str]:
        """Render the report header."""
        lines = []
        lines.append(f"# Prompt Comparison Report: {report.display_name}")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp.strftime('%B %d, %Y at %I:%M:%S %p')}")
        lines.append(f"**Model:** `{report.display_name}`")
        if report.hosting_provider:
            lines.append(f"**Provider:** {report.hosting_provider}")
        lines.append(f"**Server URL:** {report.server_url or 'N/A'}")
        if report.test_suite_name:
            lines.append(f"**Test Suite:** {report.test_suite_name}")
        if report.prompts_file_name:
            lines.append(f"**Prompts File:** {report.prompts_file_name}")
        lines.append(f"**Number of Prompts:** {len(report.prompt_results)}")
        lines.append("")
        return lines

    def _render_summary_table(self, report: "PromptComparisonReport") -> List[str]:
        """Render the summary comparison table."""
        lines = []
        lines.append("## Summary")
        lines.append("")
        lines.append("| Prompt | Pass Rate | Passed | Failed | Avg Time |")
        lines.append("|--------|-----------|--------|--------|----------|")

        # Sort by pass rate (descending), then by average time (ascending)
        sorted_results = sorted(
            report.prompt_results,
            key=lambda x: (-x.pass_rate, x.avg_time_ms)
        )

        for pr in sorted_results:
            # Add best marker
            if pr.prompt_id == report.best_prompt_id:
                name_display = f"**{pr.prompt_name}** (Best)"
            else:
                name_display = pr.prompt_name

            # Add emoji based on pass rate
            if pr.pass_rate >= 90:
                emoji = ""
            elif pr.pass_rate >= 70:
                emoji = ""
            else:
                emoji = ""

            lines.append(
                f"| {emoji} {name_display} | {pr.pass_rate:.1f}% | "
                f"{pr.passed_tests} | {pr.failed_tests} | {pr.avg_time_ms:.1f}ms |"
            )

        lines.append("")

        # Best prompt callout
        if report.best_prompt_name:
            best_result = next(
                (pr for pr in report.prompt_results if pr.prompt_id == report.best_prompt_id),
                None
            )
            if best_result:
                lines.append(f"> **Best Performing Prompt:** {report.best_prompt_name} "
                           f"with {best_result.pass_rate:.1f}% pass rate")
                lines.append("")

        return lines

    def _render_per_test_comparison(self, report: "PromptComparisonReport") -> List[str]:
        """Render the per-test comparison table."""
        lines = []
        lines.append("## Per-Test Comparison")
        lines.append("")

        if not report.prompt_results:
            lines.append("*No results available*")
            lines.append("")
            return lines

        # Build header
        header_parts = ["| Test ID"]
        for pr in report.prompt_results:
            header_parts.append(f" {pr.prompt_name}")
        header_parts.append(" |")
        lines.append("".join(header_parts))

        # Build separator
        separator_parts = ["|--------"]
        for _ in report.prompt_results:
            separator_parts.append("--------")
        separator_parts.append("|")
        lines.append("|".join(separator_parts))

        # Build rows - collect all test IDs
        all_test_ids = set()
        for pr in report.prompt_results:
            for result in pr.test_results:
                all_test_ids.add(result.test_id)

        for test_id in sorted(all_test_ids):
            row_parts = [f"| {test_id}"]
            for pr in report.prompt_results:
                # Find result for this test
                result = next(
                    (r for r in pr.test_results if r.test_id == test_id),
                    None
                )
                if result:
                    status = "PASS" if result.passed else "FAIL"
                else:
                    status = "N/A"
                row_parts.append(f" {status}")
            row_parts.append(" |")
            lines.append("".join(row_parts))

        lines.append("")
        return lines

    def _render_variable_results(self, report: "PromptComparisonReport") -> List[str]:
        """Render the variable results section."""
        lines = []
        lines.append("## Variable Results")
        lines.append("")
        lines.append("These tests passed with some prompts but failed with others, "
                    "indicating areas where prompt wording makes a difference.")
        lines.append("")

        for test_id in report.variable_test_ids:
            lines.append(f"### {test_id}")
            lines.append("")

            # Find the test details from first result
            test_command = None
            test_description = None
            for pr in report.prompt_results:
                for result in pr.test_results:
                    if result.test_id == test_id:
                        test_command = result.voice_command
                        test_description = result.test_description
                        break
                if test_command:
                    break

            if test_command:
                lines.append(f"**Command:** `{test_command}`")
            if test_description:
                lines.append(f"**Description:** {test_description}")
            lines.append("")

            lines.append("| Prompt | Result | Time |")
            lines.append("|--------|--------|------|")

            for pr in report.prompt_results:
                result = next(
                    (r for r in pr.test_results if r.test_id == test_id),
                    None
                )
                if result:
                    status = "PASS" if result.passed else "FAIL"
                    time_str = f"{result.evaluation_time_ms:.1f}ms" if result.evaluation_time_ms else "N/A"
                    lines.append(f"| {pr.prompt_name} | {status} | {time_str} |")

            lines.append("")

        return lines

    def _render_prompt_details(self, report: "PromptComparisonReport") -> List[str]:
        """Render the prompt details section."""
        lines = []
        lines.append("## Prompt Details")
        lines.append("")

        for pr in report.prompt_results:
            best_marker = " (Best)" if pr.prompt_id == report.best_prompt_id else ""
            lines.append(f"### {pr.prompt_name}{best_marker}")
            lines.append("")
            lines.append(f"**ID:** `{pr.prompt_id}`")
            lines.append(f"**Pass Rate:** {pr.pass_rate:.1f}%")
            lines.append(f"**Tests:** {pr.passed_tests} passed, {pr.failed_tests} failed")
            lines.append(f"**Total Time:** {pr.total_time_ms:.1f}ms")
            lines.append(f"**Average Time:** {pr.avg_time_ms:.1f}ms")
            lines.append("")
            lines.append("**System Prompt:**")
            lines.append("```")
            lines.append(pr.system_prompt)
            lines.append("```")
            lines.append("")

        return lines

    def _render_footer(self) -> List[str]:
        """Render report footer."""
        lines = []
        lines.append("---")
        lines.append("*This comparison report was automatically generated by Sledge Eval*")
        return lines
