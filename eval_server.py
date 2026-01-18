#!/usr/bin/env python3
"""
Sledge Eval CLI for llama-server

Entry point for running voice command evaluations against llama-server.

Usage:
    python eval_server.py --port 8080
    python eval_server.py --server-url http://localhost:8080 --mode suite
    python eval_server.py --port 8080 --mode custom
"""

import argparse
from pathlib import Path

from src.sledge_eval.cli import ServerRunner


def main():
    """Main entry point for sledge-eval server CLI."""
    parser = argparse.ArgumentParser(
        description="Sledge Eval - Voice command evaluation using llama-server",
        prog="python eval_server.py",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="URL of the llama-server instance (default: http://localhost:PORT)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port of the llama-server instance (default: 8080)",
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        help="Path to test suite JSON file (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "suite", "custom", "all", "anki", "text"],
        default="all",
        help="Evaluation mode: single test, test suite, custom tools, all tests combined, anki large toolset, or text evaluation (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging of requests and responses",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name for report generation (will auto-detect from server if not provided)",
    )

    args = parser.parse_args()

    # Determine server URL
    if args.server_url:
        server_url = args.server_url
    else:
        server_url = f"http://localhost:{args.port}"

    print(f"🚀 Sledge Eval Server Client")
    print(f"Server URL: {server_url}")
    print(f"Mode: {args.mode}")
    print(f"Timeout: {args.timeout}s")
    if args.debug:
        print(f"Debug: ENABLED 🐛")
    print()

    try:
        # Create runner
        runner = ServerRunner(
            server_url=server_url,
            model_name=args.model_name,
            timeout=args.timeout,
            debug=args.debug,
        )

        # Parse test file path if provided
        test_file = Path(args.test_suite) if args.test_suite else None

        # Run the appropriate mode
        success = False
        if args.mode == "single":
            success = runner.run_single_test()
        elif args.mode == "suite":
            success = runner.run_test_suite(test_file)
        elif args.mode == "custom":
            success = runner.run_custom_tools()
        elif args.mode == "anki":
            success = runner.run_anki_tests()
        elif args.mode == "text":
            success = runner.run_text_evaluation(test_file)
        elif args.mode == "all":
            success = runner.run_all_tests(test_file)

        if success:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
