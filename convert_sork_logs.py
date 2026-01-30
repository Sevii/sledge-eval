#!/usr/bin/env python3
"""Convert Sork game voice command logs to sledge-eval test suites.

Usage:
    python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs
    python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs --success-only
    python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs --dedupe normalized --debug
"""

import argparse
import sys
from pathlib import Path

from src.sledge_eval.converters.sork_converter import SorkConverter


def main():
    parser = argparse.ArgumentParser(
        description="Convert Sork voice command logs to sledge-eval test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs
  python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs --success-only
  python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs --dedupe normalized --debug
  python convert_sork_logs.py --logs-dir ~/Projects/sork-experience/logs --extract-tools
"""
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory containing Sork JSONL log files"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/test_data/sork_voice_commands_suite.json"),
        help="Output path for the test suite JSON (default: tests/test_data/sork_voice_commands_suite.json)"
    )

    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include successful commands in the test suite"
    )

    parser.add_argument(
        "--dedupe",
        choices=["exact", "normalized", "none"],
        default="exact",
        help="Deduplication strategy: exact (default), normalized (case-insensitive), or none"
    )

    parser.add_argument(
        "--extract-tools",
        action="store_true",
        help="Also extract and display tool definitions from LLM logs"
    )

    parser.add_argument(
        "--name",
        default="Sork Voice Commands",
        help="Name for the generated test suite"
    )

    parser.add_argument(
        "--description",
        default=None,
        help="Description for the generated test suite"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output"
    )

    args = parser.parse_args()

    # Expand user path
    logs_dir = args.logs_dir.expanduser()

    # Validate logs directory exists
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    if not logs_dir.is_dir():
        print(f"Error: Not a directory: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # Create converter
    converter = SorkConverter(debug=args.debug)

    # Extract tool definitions if requested
    if args.extract_tools:
        print("\n=== Extracting Tool Definitions ===")
        tools = converter.extract_tool_definitions(logs_dir)
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            func = tool.get('function', {})
            print(f"  - {func.get('name')}: {func.get('description', '')[:60]}...")

    # Run conversion
    print(f"\n=== Converting Sork Logs ===")
    print(f"Source: {logs_dir}")
    print(f"Output: {args.output}")
    print(f"Success only: {args.success_only}")
    print(f"Deduplication: {args.dedupe}")

    suite = converter.convert(
        logs_dir=logs_dir,
        output_path=args.output,
        success_only=args.success_only,
        dedupe_strategy=args.dedupe,
        suite_name=args.name,
        suite_description=args.description
    )

    # Print summary
    print(f"\n=== Conversion Complete ===")
    print(f"Test suite: {suite.name}")
    print(f"Total tests: {len(suite.tests)}")

    # Count by tags
    tag_counts = {}
    for test in suite.tests:
        for tag in test.get('tags', []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\nTests by tag:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    # Count successful vs edge cases
    edge_cases = sum(1 for t in suite.tests if 'edge_case' in t.get('tags', []))
    print(f"\nSuccessful commands: {len(suite.tests) - edge_cases}")
    print(f"Edge cases: {edge_cases}")

    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
