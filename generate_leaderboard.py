#!/usr/bin/env python3
"""Generate a markdown leaderboard from evaluation report JSON files."""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def clean_model_name(name: str) -> str:
    """
    Clean up model name by removing paths, extensions, and common prefixes.

    Args:
        name: Raw model name from report

    Returns:
        Cleaned model name
    """
    import re

    # Remove quotes
    name = name.strip('"\'')

    # If it's a path, extract the filename or meaningful part
    if '/' in name:
        parts = name.split('/')
        # Look for meaningful parts (not snapshots, not cache dirs)
        for part in reversed(parts):
            if part and not part.startswith('.') and 'snapshots' not in part and 'cache' not in part:
                # Check if it looks like a model name (has letters)
                if re.search(r'[a-zA-Z]', part):
                    name = part
                    break

    # Remove common file extensions
    for ext in ['.gguf', '.bin', '.safetensors', '.pt', '.pth']:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]

    # Remove common prefixes like "../"
    name = name.lstrip('./')

    # Clean up HuggingFace-style paths (org/model -> model or keep org/model)
    # but prefer the directory name from reports if it's cleaner
    if name.startswith('models--'):
        # Extract from HuggingFace cache path format: models--org--model
        match = re.match(r'models--([^-]+)--(.+)', name)
        if match:
            org, model = match.groups()
            name = f"{org}/{model}"

    return name.strip()


def find_latest_reports(reports_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Find the most recent report for each model in the reports directory.

    Args:
        reports_dir: Path to the reports directory

    Returns:
        Dictionary mapping model names to their latest report data
    """
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        print(f"Error: Reports directory '{reports_dir}' not found")
        sys.exit(1)

    model_reports: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}

    # Scan all subdirectories for JSON reports
    for model_dir in reports_path.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue

        for json_file in model_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    report = json.load(f)

                # Parse timestamp
                timestamp_str = report.get("timestamp", "")
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        timestamp = datetime.min

                # Get model name (clean it up if needed)
                model_name = report.get("model_name", model_dir.name)
                model_name = clean_model_name(model_name)

                # Keep the latest report for each model
                if model_name not in model_reports or timestamp > model_reports[model_name][0]:
                    model_reports[model_name] = (timestamp, report)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
                continue

    # Return just the report data (without timestamps)
    return {name: data for name, (_, data) in model_reports.items()}


def calculate_category_scores(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate pass rates for major test categories.

    Args:
        report: The report data

    Returns:
        Dictionary of category scores
    """
    tag_perf = report.get("tag_performance", {})

    # Define category groupings
    categories = {
        "Tool Calling": ["smart_home", "lights", "thermostat", "music", "weather"],
        "Anki (Large Toolset)": ["anki"],
        "Letter Counting": ["letter_counting"],
        "Theory of Mind": ["theory_of_mind"],
        "Multi-Step": ["multi_action", "multi_step", "sequential"],
    }

    category_scores = {}

    for category, tags in categories.items():
        passed = 0
        total = 0
        for tag in tags:
            if tag in tag_perf:
                passed += tag_perf[tag]["passed"]
                total += tag_perf[tag]["total"]

        if total > 0:
            category_scores[category] = {
                "passed": passed,
                "total": total,
                "rate": (passed / total) * 100
            }

    return category_scores


def format_pass_rate(passed: int, total: int) -> str:
    """Format pass rate as 'X/Y (Z%)'."""
    if total == 0:
        return "N/A"
    rate = (passed / total) * 100
    return f"{passed}/{total} ({rate:.1f}%)"


def generate_leaderboard(reports: Dict[str, Dict[str, Any]], output_file: Optional[str] = None) -> str:
    """
    Generate a markdown leaderboard from report data.

    Args:
        reports: Dictionary of model names to report data
        output_file: Optional file path to write the markdown output

    Returns:
        The generated markdown string
    """
    if not reports:
        return "No reports found."

    # Sort models by pass rate (descending)
    sorted_models = sorted(
        reports.items(),
        key=lambda x: x[1].get("pass_rate", 0),
        reverse=True
    )

    lines = []

    # Header
    lines.append("# Sledge Eval Leaderboard")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Overall Rankings Table
    lines.append("## Overall Rankings")
    lines.append("")
    lines.append("| Rank | Model | Pass Rate | Passed | Failed | Total | Avg Time (ms) |")
    lines.append("|------|-------|-----------|--------|--------|-------|---------------|")

    for rank, (model_name, report) in enumerate(sorted_models, 1):
        pass_rate = report.get("pass_rate", 0)
        passed = report.get("passed_tests", 0)
        failed = report.get("failed_tests", 0)
        total = report.get("total_tests", 0)
        total_time = report.get("total_evaluation_time_ms", 0)
        avg_time = total_time / total if total > 0 else 0

        # Add medal emoji for top 3
        rank_str = str(rank)
        if rank == 1:
            rank_str = "🥇 1"
        elif rank == 2:
            rank_str = "🥈 2"
        elif rank == 3:
            rank_str = "🥉 3"

        lines.append(f"| {rank_str} | {model_name} | {pass_rate:.1f}% | {passed} | {failed} | {total} | {avg_time:.1f} |")

    lines.append("")

    # Category Breakdown Table
    lines.append("## Performance by Category")
    lines.append("")

    # Get all categories
    all_categories = set()
    model_category_scores = {}
    for model_name, report in sorted_models:
        scores = calculate_category_scores(report)
        model_category_scores[model_name] = scores
        all_categories.update(scores.keys())

    # Sort categories for consistent ordering
    categories_list = sorted(all_categories)

    # Build category table header
    header = "| Model |"
    separator = "|-------|"
    for cat in categories_list:
        header += f" {cat} |"
        separator += "--------|"

    lines.append(header)
    lines.append(separator)

    for model_name, report in sorted_models:
        scores = model_category_scores[model_name]
        row = f"| {model_name} |"
        for cat in categories_list:
            if cat in scores:
                s = scores[cat]
                row += f" {format_pass_rate(s['passed'], s['total'])} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # Hardware Info Section (for local models)
    lines.append("## Hardware Information")
    lines.append("")
    lines.append("*Hardware info shown for locally-run models only.*")
    lines.append("")

    has_hardware_info = False
    for model_name, report in sorted_models:
        hw = report.get("hardware_info")
        if hw:
            has_hardware_info = True
            lines.append(f"### {model_name}")
            lines.append("")

            if hw.get("gpu_name"):
                lines.append(f"- **GPU:** {hw['gpu_name']}")
            if hw.get("processor"):
                lines.append(f"- **CPU:** {hw['processor']}")
            if hw.get("total_memory_mb"):
                lines.append(f"- **System RAM:** {hw['total_memory_mb'] / 1024:.1f} GB")
            if hw.get("gpu_memory_mb"):
                lines.append(f"- **GPU Memory:** {hw['gpu_memory_mb'] / 1024:.1f} GB")
            if hw.get("os_name"):
                lines.append(f"- **OS:** {hw['os_name']} {hw.get('os_version', '')}")

            lines.append("")

    if not has_hardware_info:
        lines.append("*No hardware information available (API-based models).*")
        lines.append("")

    # Test Suite Info
    lines.append("## Test Suite Information")
    lines.append("")

    # Get unique tags across all reports
    all_tags = set()
    for report in reports.values():
        all_tags.update(report.get("tags_tested", []))

    lines.append(f"- **Total unique test categories:** {len(all_tags)}")
    lines.append(f"- **Models evaluated:** {len(reports)}")
    lines.append("")
    lines.append("### Test Categories")
    lines.append("")

    # Group tags nicely
    tag_list = sorted(all_tags)
    lines.append(", ".join(f"`{tag}`" for tag in tag_list))
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This leaderboard is automatically generated from evaluation reports in the `reports/` directory.*")

    markdown = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(markdown)
        print(f"Leaderboard written to: {output_file}")

    return markdown


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate leaderboard from evaluation reports")
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing model report subdirectories (default: reports)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="LEADERBOARD.md",
        help="Output markdown file (default: LEADERBOARD.md)"
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print leaderboard to stdout"
    )

    args = parser.parse_args()

    # Find all reports
    reports = find_latest_reports(args.reports_dir)

    if not reports:
        print("No valid reports found in the reports directory.")
        sys.exit(1)

    print(f"Found {len(reports)} model reports")

    # Generate leaderboard
    markdown = generate_leaderboard(reports, args.output)

    if args.print:
        print("\n" + markdown)


if __name__ == "__main__":
    main()
